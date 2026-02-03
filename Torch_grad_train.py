import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
from datetime import datetime
import cv2  # For Grad-CAM visualization overlay
import torch.nn.functional as F
from PIL import Image
import os

# IMPORTANT: import your model defintions from correct directory
from Torch_models.gctch_alexnet import GradCAMReadyAlexNet
from Torch_models.gctch_Resnet18 import GradCAMReadyResNet18
from Torch_models.gctch_mobilenet import GradCAMReadyMobileNetV2
from Torch_models.gctch_cnn import GradCAMReadyCNN


# --- Utility Functions ---

def create_results_folder(model_name, augmentation_type):
    """Creates a timestamped folder for saving results."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{model_name}_{augmentation_type}_{timestamp}"
    results_dir = os.path.join("results", folder_name)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_training_logs(results_folder, history_dict):
    """Saves training metrics to a JSON file."""
    log_path = os.path.join(results_folder, "training_metrics.json")
    with open(log_path, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"Training metrics saved to {log_path}")


def save_loss_curve(results_folder, history):
    """Saves loss and accuracy curves."""
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, "loss_accuracy_curve.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss and accuracy curves saved to {plot_path}")


def plot_confusion_matrix(cm, class_names, results_folder):
    """Plots and saves a confusion matrix heatmap using seaborn."""
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plot_path = os.path.join(results_folder, "confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")


def plot_roc_curve(y_true, y_pred_prob, num_classes, results_folder, class_names):
    """Plots and saves the ROC curve for multi-class classification."""
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(8, 8))

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plot_path = os.path.join(results_folder, "roc_curve.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"ROC curve saved to {plot_path}")


# --- Grad-CAM Implementation ---
"""
    Applies GradCAM directly in this training pipeline,
    It is optinal and can be avoided for pure training purposes and not checking XAI
"""

class GradCAM:
    """
    Implements Grad-CAM for a PyTorch model.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        # This hook captures the output (activations) of the target layer during forward pass
        self.target_layer.register_forward_hook(self._save_activations)
        # This hook captures the gradients flowing back into the target layer during backward pass
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output[0] is the gradient w.r.t. the output of the layer
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, class_idx=None):
        """
        Generates a Grad-CAM heatmap for a given input.

        Args:
            input_tensor (torch.Tensor): The input image tensor (batch_size, channels, H, W).
                                         Ensure this tensor has requires_grad=True if you're not
                                         getting gradients from a prior call.
            class_idx (int, optional): The index of the class to compute Grad-CAM for.
                                       If None, the predicted class will be used.

        Returns:
            np.ndarray: The normalized Grad-CAM heatmap (H, W).
        """
        # Ensure model is in evaluation mode
        self.model.eval()

        # Clear previous gradients and activations
        self.gradients = None
        self.activations = None

        # Ensure gradients are enabled for the input tensor if it's not already
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)

        # Forward pass to get activations and predictions
        conv_features, predictions = self.model(input_tensor)

        if class_idx is None:
            # Use the predicted class if not specified
            class_idx = torch.argmax(predictions, dim=1).item()

        # Get the score for the target class
        score = predictions[:, class_idx]

        # Zero gradients for the model and perform backward pass for the score
        self.model.zero_grad()
        score.backward(retain_graph=True)  # retain_graph=True if you plan subsequent backward calls

        # Get the activations and gradients from the hooks
        gradients = self.gradients
        activations = self.activations

        # Global Average Pooling of gradients
        # Shape of gradients: (batch_size, channels, H, W)
        # We average over H and W for each channel to get channel-wise weights
        pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)  # (batch_size, channels, 1, 1)

        # Weight the activations by the pooled gradients
        # (batch_size, channels, H, W) * (batch_size, channels, 1, 1) -> (batch_size, channels, H, W)
        weighted_activations = activations * pooled_gradients

        # Sum across the channel dimension to get the heatmap
        # Resulting shape: (batch_size, H, W)
        heatmap = torch.sum(weighted_activations, dim=1)

        # Apply ReLU to the heatmap (only positive contributions)
        heatmap = F.relu(heatmap)

        # Normalize the heatmap to 0-1
        # Extract heatmap for the first image in the batch (assuming batch_size=1 for Grad-CAM input)
        heatmap_single = heatmap[0]
        max_val = heatmap_single.max()
        min_val = heatmap_single.min()
        if max_val > min_val:  # Avoid division by zero if heatmap is all same value
            heatmap_single = (heatmap_single - min_val) / (max_val - min_val)
        else:
            heatmap_single = torch.zeros_like(heatmap_single)  # All zeros if no activation change

        return heatmap_single.cpu().detach().numpy()


def visualize_gradcam(original_img_path, heatmap, results_folder, class_name):
    """
    Overlays the Grad-CAM heatmap on the original image and saves it.
    Now includes the image filename in the title and saved path.
    """
    # Load original image using PIL and apply transformations
    original_img_pil_loaded = Image.open(original_img_path)

    transform_for_visualization = transforms.Compose([
        transforms.Resize((256, 256)),
        # Resize to model input size (AlexNet might prefer 224, but keep consistent for vis)
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB for visualization with heatmap
        transforms.ToTensor()  # Converts PIL Image to (C, H, W) Tensor, scales to [0, 1]
    ])

    original_img_tensor_processed = transform_for_visualization(original_img_pil_loaded)
    original_img_np = original_img_tensor_processed.permute(1, 2, 0).numpy()  # Convert to (H, W, C) for OpenCV

    # Scale heatmap to 0-255 and apply colormap
    h, w, _ = original_img_np.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))  # Resize heatmap to image dimensions
    heatmap_resized = np.uint8(255 * heatmap_resized)  # Scale to 0-255
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)  # Apply JET colormap (BGR)

    # Convert original_img_np (which is RGB, 0-1 float) to BGR and 0-255 uint8 for OpenCV
    original_img_np_255_bgr = cv2.cvtColor(np.uint8(original_img_np * 255), cv2.COLOR_RGB2BGR)

    # Overlay heatmap on original image
    superimposed_img = cv2.addWeighted(original_img_np_255_bgr, 0.5, heatmap_colored, 0.5, 0)

    # Convert back to RGB for matplotlib and saving
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # --- ADDITIONS FOR IMAGE NAME ---
    image_filename = os.path.basename(original_img_path)  # Get just the filename
    image_name_without_ext = os.path.splitext(image_filename)[0]  # Get filename without extension
    # --- END ADDITIONS ---

    plt.figure(figsize=(8, 8))
    # Update the title
    plt.title(f"Grad-CAM for Class: {class_name}\nImage: {image_name_without_ext}")
    plt.imshow(superimposed_img_rgb)
    plt.axis("off")
    # Update the saved plot path
    plot_path = os.path.join(results_folder, f"gradcam_{class_name}_{image_name_without_ext}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Grad-CAM visualization saved to {plot_path}")


# --- Main Training and Evaluation Function ---

def main():
    # --- Configuration ---
    input_shape = (256, 256, 1)  # (Height, Width, Channels) - Consistent with your Keras config
    num_classes = 4
    model_name = "grad_AlexNet_Pytorch"
    augmentation_type = "Split_STEM_100"
    epochs = 200
    batch_size = 32
    learning_rate = 0.001

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create results folder
    results_folder = create_results_folder(model_name=model_name, augmentation_type=augmentation_type)
    print(f"Results will be saved in: {results_folder}")

    # --- Data Loading ---
    # Define transformations for training and validation/test data
    # PyTorch expects (C, H, W) format, so resize comes first, then ToTensor converts HWC to CHW
    # Grayscale conversion is crucial for input_channels=1
    train_transform = transforms.Compose([
        transforms.Resize((input_shape[0], input_shape[1])),  # Resize to (H, W)
        transforms.Grayscale(num_output_channels=input_shape[2]),  # Convert to 1 channel for grayscale
        transforms.ToTensor(),  # Converts to (C, H, W) and scales pixel values to [0, 1]
        # Add more augmentations here if needed (e.g., transforms.RandomRotation, transforms.RandomHorizontalFlip)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((input_shape[0], input_shape[1])),
        transforms.Grayscale(num_output_channels=input_shape[2]),  # Convert to 1 channel for grayscale
        transforms.ToTensor(),
    ])

    # Load datasets using ImageFolder
    # Make sure these paths are correct for your system
    #train_data_dir = '/mnt/d/Harsh/Major_Project/Database/Test_Set_SPECT/Split_STEM/train'
    train_data_dir = '/Users/harshyadav/Desktop/pythonProject/Remote/Database/wetransfer_du_original-zip_2025-05-31_0415/Test_Set_SPECT/Split_STEM/train'
    #test_data_dir = '/mnt/d/Harsh/Major_Project/Database/Test_Set_SPECT/Split_STEM/test'
    test_data_dir = '/Users/harshyadav/Desktop/Aiims/cropped_dtgpr'

    full_train_dataset = datasets.ImageFolder(
        root=train_data_dir,
        transform=train_transform
    )

    # Split full_train_dataset into train and validation (80/20 split)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

    test_dataset = datasets.ImageFolder(
        root=test_data_dir,
        transform=test_transform
    )

    # Create DataLoaders
    # num_workers > 0 enables multi-process data loading, which is faster but requires
    # `if __name__ == '__main__':` block for Windows compatibility.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Get class names from the dataset (they are sorted by default by ImageFolder)
    class_names = full_train_dataset.classes  # Use full_train_dataset to get all classes
    print("Class names and their corresponding indices:")
    for i, class_name in enumerate(class_names):
        print(f"Class: {class_name}, Index: {i}")

    # Print basic class distribution (for the full dataset before split)
    print("\nFull Dataset Class Distribution:")
    class_counts = {}
    for _, idx in full_train_dataset.imgs:  # (image_path, class_idx) tuples
        class_name = class_names[idx]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    for class_name, count in class_counts.items():
        print(f"Class: {class_name}, Count: {count}")

    # --- Model, Loss, Optimizer ---
    # `input_channels` for the model should be the channel dimension, which is 1 for grayscale
    # Change model = model_name to train a different model here
    # Example: model = GradCAMReadyCNN(input_channels=1, num_classes=num_classes)
    model = GradCAMReadyAlexNet(input_channels=1, num_classes=num_classes, pretrained= True).to(device)

    # PyTorch's CrossEntropyLoss expects logits (raw outputs) and handles softmax internally
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(results_folder, "best_model_pytorch.pth")

    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients before each batch

            # Forward pass: model returns conv_features and predictions (logits)
            conv_features, predictions = model(inputs)  # conv_features is not used for loss calculation

            # Calculate loss only on the predictions output
            loss = criterion(predictions, labels)

            loss.backward()  # Backward pass: compute gradients
            optimizer.step()  # Update weights

            running_loss += loss.item() * inputs.size(0)

            # Calculate accuracy for the current batch
            _, predicted = torch.max(predictions.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / total_predictions
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)

        # --- Validation ---
        model.eval()  # Set model to evaluation mode
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0

        with torch.no_grad():  # Disable gradient calculation for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                _, predictions = model(inputs)  # Only need predictions for validation
                val_loss = criterion(predictions, labels)

                val_running_loss += val_loss.item() * inputs.size(0)

                _, predicted = torch.max(predictions.data, 1)
                val_total_predictions += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_accuracy = val_correct_predictions / val_total_predictions
        history['val_loss'].append(val_epoch_loss)
        history['val_accuracy'].append(val_epoch_accuracy)

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {epoch_loss:.4f} Train Acc: {epoch_accuracy:.4f} - "
              f"Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_accuracy:.4f}")

        # Early stopping and Model Checkpoint logic
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)  # Save best model weights
            print(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= 30:  # IMPORTANT: CHANGE 30 TO CHANGE AMOUNT OF EPOCHES BEFORE EARLY STOPPING
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    print("\n--- Training Complete ---")

    # Load the best model weights for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)  # Ensure model is on the correct device after loading
    print(f"Best model weights loaded from {best_model_path}")

    # --- Evaluation ---
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_predictions = []
    all_probabilities = []

    print("\n--- Starting Evaluation ---")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            _, predictions = model(inputs)  # Get predictions (logits)

            all_labels.extend(labels.cpu().numpy())
            # Convert logits to probabilities using softmax for ROC curve
            all_probabilities.extend(F.softmax(predictions, dim=1).cpu().numpy())
            _, predicted_classes = torch.max(predictions, 1)  # Get the class with highest logit
            all_predictions.extend(predicted_classes.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plot_confusion_matrix(cm, class_names, results_folder)

    # Classification Report
    class_report_dict = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    report_path = os.path.join(results_folder, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(class_report_dict, f, indent=4)
    print(f"Classification report saved to {report_path}")

    # ROC Curve
    plot_roc_curve(all_labels, all_probabilities, num_classes, results_folder, class_names)

    # Save training history plots
    save_training_logs(results_folder, history)  # save_training_logs expects the history dict directly
    save_loss_curve(results_folder, history)

    print(f"Evaluation metrics and plots saved in: {results_folder}")

    # --- Grad-CAM Visualization Example ---
    print("\n--- Performing Grad-CAM Visualization ---")
    # Identify the target layer for Grad-CAM (e.g., the last convolutional layer)
    # In our GradCAMReadyCNN, it's self.conv3
    #target_layer_for_gradcam = model.conv3 #for CNN
    target_layer_for_gradcam = model.target_layer #for MObileNet

    # Instantiate GradCAM, Comment Out for Trraining Testing only
    grad_cam = GradCAM(model, target_layer_for_gradcam)

    # Get one image from the test set for visualization
    # We need the original image path for visualization overlay
    # test_dataset.samples[0][0] gives the path of the first image
    if len(test_dataset) > 0:
        sample_image_tensor, sample_label = test_dataset[0]
        sample_image_path = test_dataset.samples[0][0]  # Correct for ImageFolder directly
        # Add batch dimension and move to device
        sample_input_for_gradcam = sample_image_tensor.unsqueeze(0).to(device)

        # Get the predicted class name for the sample image
        model.eval()  # Ensure model is in eval mode
        with torch.no_grad():
            _, pred_logits = model(sample_input_for_gradcam)
            predicted_class_idx = torch.argmax(pred_logits, dim=1).item()
        predicted_class_name = class_names[predicted_class_idx]

        # Generate heatmap
        # Ensure gradients are enabled for the input when calling GradCAM
        # The GradCAM.__call__ method handles setting requires_grad for the input tensor
        heatmap = grad_cam(sample_input_for_gradcam, class_idx=predicted_class_idx)

        # Visualize and save Grad-CAM
        visualize_gradcam(sample_image_path, heatmap, results_folder, predicted_class_name)
        print("Grad-CAM visualization complete.")
    else:
        print("No test data available for Grad-CAM visualization.")


if __name__ == "__main__":
    main()