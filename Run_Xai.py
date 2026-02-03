from pyexpat import features
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from datetime import datetime
import random
#To tackle the ssl certificate issue on Mac M1
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import json  # For loading model config if needed

# IMPORTANT: Import your custom model architectures from their new locations
from Torch_models.gctch_cnn import GradCAMReadyCNN
from Torch_models.gctch_alexnet import GradCAMReadyAlexNet
from Torch_models.gctch_mobilenet import GradCAMReadyMobileNetV2
from Torch_models.gctch_Resnet18 import GradCAMReadyResNet18

# IMPORTANT: Import your XAI method implementations from their new location
from Xai_classes.gradcam_classes import GradCAM
from Xai_classes.gcamPP_classes import GradCAMPlusPlus
from Xai_classes.Saliency_maps_class import SaliencyMap
from Xai_classes.Occulsion_classes import Occlusion
from Xai_classes.Smoothgrad_Saliencymaps import SmoothGradSaliencyMap





# --- Utility Functions ---

# create_output_folder uses the base_dir from config if provided,
# otherwise fall back to a default relative path.
def create_output_folder(base_dir, model_name="model", method="gradcam"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{model_name}_{method}_{timestamp}"
    results_dir = os.path.join(base_dir, folder_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"XAI results will be saved in: {results_dir}")
    return results_dir

# Function to create Xai overlay on origianl scans
def visualize_heatmap_overlay_side_by_side(
        original_img_path, heatmap, predicted_class_name, true_class_name,
        output_folder, image_filename, input_img_size=(256, 256)
):
    try:
        original_img_pil = Image.open(original_img_path).convert('RGB')
        original_img_resized = original_img_pil.resize(input_img_size)
        original_img_np = np.array(original_img_resized)

        h, w = input_img_size
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
        superimposed_img_bgr = cv2.addWeighted(original_img_bgr, 0.5, heatmap_colored, 0.5, 0)
        superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(original_img_np)
        plt.title(f"Original (True: {true_class_name})", fontsize=10)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(superimposed_img_rgb)
        plt.title(f"XAI Overlay (Predicted: {predicted_class_name})", fontsize=10)
        plt.axis("off")

        plt.tight_layout()
        # Save filename includes true and predicted class names for clarity
        save_path = os.path.join(output_folder, f"xai_pred_{predicted_class_name}_true_{true_class_name}_{image_filename}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved for {image_filename} to {save_path}")

    except Exception as e:
        print(f"Error processing {original_img_path}: {e}")


# --- Model Specific Configurations ---
# Configurations to help easy permutations of model and xai

def get_target_layer(model, model_name, target_layer_name=None):
    if target_layer_name:
        for name, module in model.named_modules():
            if name == target_layer_name:
                return module
        raise ValueError(f"Specified target layer '{target_layer_name}' not found in {model_name}.")

    if "GradCAMReadyCNN" in model_name.lower():
        return model.conv3
    elif "GradCAMReadyAlexNet" in model_name.lower():
        return model.features[10]
    elif "GradCAMReadyMobileNetV2" in model_name.lower():
        return model.features[18].conv[0]
    elif "gradcamreadyresnet18" in model_name.lower():
        return model.layer4[-1]

    raise ValueError(f"Could not automatically determine target layer for model: {model_name}. "
                     "Please specify `target_layer_name` manually in the config if not using Grad-CAM/Grad-CAM++ or add its logic here.")


# --- XAI Method Registry ---
XAI_METHODS = {
    "GradCAM": GradCAM,
    "GradCAMPlusPlus": GradCAMPlusPlus,
    "SaliencyMap": SaliencyMap,
    "Occlusion": Occlusion,
    "SmoothGradSaliencyMap": SmoothGradSaliencyMap
}


# --- Main XAI Visualization Runner ---

def run_xai_visualization(config):
    # --- Configuration from Dict ---
    model_architecture_class = config['model_architecture_class']
    model_weights_path = config['model_weights_path']
    image_directory_path = config['image_directory_path']
    # num_classes and class_names will be derived dynamically
    input_img_size = config['input_img_size']
    input_channels = config['input_channels']
    xai_method_name = config['xai_method_name']
    target_layer_name = config.get('target_layer_name', None)
    samples_per_class = config['samples_per_class']
    output_base_dir = config['output_base_dir'] # Use this for output folder
    model_name_for_logging = config['model_name_for_logging']

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading for XAI (Initial pass to get class names and num_classes) ---
    xai_transform = transforms.Compose([
        transforms.Resize(input_img_size),
        transforms.Grayscale(num_output_channels=input_channels) if input_channels == 1 else transforms.Lambda(
            lambda x: x),
        transforms.ToTensor(),
        # ADD ANY NORMALIZATION HERE USED DURING TRAINING
        # Example for ImageNet normalization (if input_channels is 3 and model was trained with it):
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Example for your custom grayscale normalization (if input_channels is 1 and model was trained with it):
        # transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    full_dataset = datasets.ImageFolder(
        root=image_directory_path,
        transform=xai_transform
    )

    class_names = full_dataset.classes  # Get class names from directory
    num_classes = len(class_names)      # Get number of classes
    print(f"Detected Classes: {class_names}")
    print(f"Detected Number of Classes: {num_classes}")

    # --- Load Model ---
    if model_architecture_class is GradCAMReadyCNN:
        model = model_architecture_class(input_channels=input_channels, num_classes=num_classes)
    elif model_architecture_class in [GradCAMReadyAlexNet, GradCAMReadyMobileNetV2, GradCAMReadyResNet18]:
        model = model_architecture_class(num_classes=num_classes) # Assuming these take num_classes
    else:
        raise ValueError(f"Unsupported model architecture class: {model_architecture_class.__name__}. "
                         "Please add its loading logic to run_xai_visualization.")

    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    print(f"Model '{model_name_for_logging}' loaded from {model_weights_path}")

    # --- Identify Target Layer (if needed for the XAI method) ---
    xai_generator = None
    if xai_method_name in ["GradCAM", "GradCAMPlusPlus"]:
        target_layer = get_target_layer(model, model_name_for_logging, target_layer_name)
        if target_layer is None:
            raise ValueError(
                f"Could not identify target layer for {xai_method_name} with model {model_name_for_logging}.")
        xai_generator = XAI_METHODS[xai_method_name](model, target_layer)
    elif xai_method_name in ["SaliencyMap"]:
        xai_generator = XAI_METHODS[xai_method_name](model)
    elif xai_method_name in ["SmoothGradSaliencyMap"]:
        xai_generator = XAI_METHODS[xai_method_name](model)
    elif xai_method_name == "Occlusion":
        xai_generator = XAI_METHODS[xai_method_name](model, input_size=input_img_size, device=device)
    else:
        raise ValueError(f"XAI method '{xai_method_name}' is not registered or handled.")

    # Create output folder for this run using the dynamically determined base_dir
    output_folder = create_output_folder(
        base_dir=output_base_dir,
        model_name=model_name_for_logging,
        method=xai_method_name
    )

    # --- Generate Samples for Each Class ---
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (path, label) in enumerate(full_dataset.imgs):
        class_indices[label].append(idx)

    print("\nStarting XAI visualization per class...")
    for class_id in range(num_classes):
        current_class_name = class_names[class_id] # Use current_class_name for clarity
        print(f"Processing class: {current_class_name} (ID: {class_id})")

        if len(class_indices[class_id]) == 0:
            print(f"No samples found for class {current_class_name}. Skipping.")
            continue

        selected_indices = class_indices[class_id]
        if len(selected_indices) > samples_per_class:
            selected_indices = random.sample(selected_indices, samples_per_class)
        else:
            print(
                f"Warning: Not enough samples for class {current_class_name}. Found {len(selected_indices)}, requested {samples_per_class}. Using all available.")

        for idx_in_dataset in selected_indices:
            original_image_path, true_label_id = full_dataset.imgs[idx_in_dataset]
            image_tensor, _ = full_dataset[idx_in_dataset]

            input_for_xai = image_tensor.unsqueeze(0).to(device)
            if xai_method_name in ["GradCAM", "GradCAMPlusPlus", "SaliencyMap"]:
                input_for_xai.requires_grad_(True)

            with torch.no_grad():
                model_output = model(input_for_xai.clone().detach())
                if isinstance(model_output, tuple):
                    pred_logits = model_output[1]
                else:
                    pred_logits = model_output
                predicted_class_id = torch.argmax(pred_logits, dim=1).item()
            predicted_class_name = class_names[predicted_class_id]
            true_class_name = class_names[true_label_id] # Get true class name from ID

            heatmap = xai_generator(input_for_xai, class_idx=predicted_class_id)

            base_filename = os.path.basename(original_image_path)
            name_without_ext = os.path.splitext(base_filename)[0]
            # Unique filename includes predicted and true class names for better organization
            unique_filename = f"{name_without_ext}_pred_{predicted_class_name}_true_{true_class_name}"

            visualize_heatmap_overlay_side_by_side(
                original_image_path, heatmap, predicted_class_name, true_class_name,
                output_folder, unique_filename, input_img_size=input_img_size
            )
    print("\nXAI Visualization Complete!")


# --- Main execution block ---
if __name__ == "__main__":
    # To run a specific combination of model and xai,
    # UNCOMMENT THE RELATIVE FUNCTION AND COMMENT ALL OTHERS
    # Define a common base directory for all results for easier management
    # Using a relative path makes it more portable.
    common_output_base_dir = "PATH_ROOT_RESULT_FOLDER_FOR_CURRENT_RUN"
    os.makedirs(common_output_base_dir, exist_ok=True) # Ensure base dir exists

    # --- Configuration for Custom CNN ---
    custom_cnn_config = {
        'model_architecture_class': GradCAMReadyCNN,
        # IMPORTANT: PUT TRAINED MODEL WEIGHT'S PATH HERE
        'model_weights_path': "/Users/harshyadav/Desktop/pythonProject/Remote/Codebase/Best_weights /grad_CNN_PYtorch_Split_STEM_100_20250527-064818/best_model_pytorch.pth",
        # IMPORTANT: PUT TEST SET(for Xai) PATH HERE
        'image_directory_path': "/Users/harshyadav/Desktop/pythonProject/Remote/Database/Converted_hard_test",
        # 'num_classes': 4, # REMOVED - derived dynamically from ImageFolder
        # 'class_names': ['class_0', 'class_1', 'class_2', 'class_3'], # REMOVED - derived dynamically
        'input_img_size': (256, 256),
        'input_channels': 1,
        'xai_method_name': "SaliencyMap", # TO CHANGE XAI METHOD RELATIVE TO CNN
        'target_layer_name': "conv3",
        'samples_per_class': 2,
        'output_base_dir': common_output_base_dir, # Use the common base dir
        'model_name_for_logging': "Custom_GradCAMReadyCNN"
    }
    #run_xai_visualization(custom_cnn_config) # Uncomment to run

    # --- Configuration for AlexNet (using your custom GradCAMReadyAlexNet) ---
    alexnet_config = {
        'model_architecture_class': GradCAMReadyAlexNet,
        'model_weights_path': "/Users/harshyadav/Desktop/pythonProject/Remote/Codebase/Best_weights /grad_AlexNet_Pytorch_Split_STEM_100_20250527-065259/best_model_pytorch.pth",
        'image_directory_path': "/Users/harshyadav/Desktop/pythonProject/Remote/Database/Converted_hard_test",
        'input_img_size': (224, 224),
        'input_channels': 1,
        'xai_method_name': "Occlusion",
        'target_layer_name': 'model.features.10',
        'samples_per_class': 2,
        'output_base_dir': common_output_base_dir,
        'model_name_for_logging': "AlexNet"
    }
    run_xai_visualization(alexnet_config)

    # --- Configuration for MobileNetV2 (using your custom GradCAMReadyMobileNetV2) ---
    mobilenet_config = {
        'model_architecture_class': GradCAMReadyMobileNetV2,
        'model_weights_path': "/Users/harshyadav/Desktop/pythonProject/Remote/Codebase/Best_weights /grad_MobileNET_Pretrained_PyTorch_Balanced_braintumor_20250527-064225/best_model_pytorch.pth", # REPLACE THIS!
        'image_directory_path': "/Users/harshyadav/Desktop/pythonProject/Remote/Database/PNG_Hard_test_set",
        'input_img_size': (224, 224),
        'input_channels': 1,
        'xai_method_name': "GradCAM",
        'target_layer_name': 'model.features.18.0',
        'samples_per_class': 2,
        'output_base_dir': common_output_base_dir,
        'model_name_for_logging': "MobileNetV2"
    }
    #run_xai_visualization(mobilenet_config)

    # --- Configuration for ResNet18 (using your custom GradCAMReadyResNet18) ---
    resnet18_config = {
        'model_architecture_class': GradCAMReadyResNet18,
        'model_weights_path': "/Users/harshyadav/Desktop/pythonProject/Remote/Codebase/Best_weights /grad_ResNet18_Pytorch_Split_STEM_100_20250531-095911/best_model_pytorch.pth", # REPLACE THIS!
        'image_directory_path': "/Users/harshyadav/Desktop/pythonProject/Remote/Database/wetransfer_du_original-zip_2025-05-31_0415/Test_Set_SPECT/Split_STEM/test",
        'input_img_size': (224, 224),
        'input_channels': 1,
        'xai_method_name': "SaliencyMap",
        'target_layer_name': 'model.layer4.1.conv2',
        'samples_per_class': 2,
        'output_base_dir': common_output_base_dir,
        'model_name_for_logging': "ResNet18"
    }
    #run_xai_visualization(resnet18_config)

    # Example: Run Saliency Map on Custom CNN
    saliency_config_custom_cnn = {
        'model_architecture_class': GradCAMReadyCNN,
        'model_weights_path': "/Users/harshyadav/Desktop/pythonProject/Remote/Codebase/Best_weights /grad_CNN_PYtorch_Split_STEM_100_20250527-064818/best_model_pytorch.pth",
        'image_directory_path': "/Users/harshyadav/Desktop/pythonProject/Remote/Database/wetransfer_du_original-zip_2025-05-31_0415/Test_Set_SPECT/Split_STEM/test",
        'input_img_size': (256, 256),
        'input_channels': 1,
        'xai_method_name': "SaliencyMap",
        'target_layer_name': None,
        'samples_per_class': 2,
        'output_base_dir': common_output_base_dir,
        'model_name_for_logging': "Custom_GradCAMReadyCNN"
    }
    # run_xai_visualization(saliency_config_custom_cnn)

    # Example: Run Occlusion on Custom CNN
    occlusion_config_custom_cnn = {
        'model_architecture_class': GradCAMReadyCNN,
        'model_weights_path': "/Users/harshyadav/Desktop/pythonProject/Remote/Codebase/Best_weights /grad_CNN_PYtorch_Split_STEM_100_20250527-064818/best_model_pytorch.pth",
        'image_directory_path': "/Users/harshyadav/Desktop/pythonProject/Remote/Database/wetransfer_du_original-zip_2025-05-31_0415/Test_Set_SPECT/Split_STEM/test",
        'input_img_size': (256, 256),
        'input_channels': 1,
        'xai_method_name': "Occlusion",
        'target_layer_name': None,
        'samples_per_class': 1,
        'output_base_dir': common_output_base_dir,
        'model_name_for_logging': "Custom_GradCAMReadyCNN"
    }
    # run_xai_visualization(occlusion_config_custom_cnn)