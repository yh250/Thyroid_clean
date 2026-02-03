import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

""" 
   Reusable Test functions for STANDALONE testing of trained model 
   Includes:
            Confusion matrix, Roc-Auc curves, Classification Reports
"""

# --- Reusable Plot Functions ---

def plot_confusion_matrix(cm, class_names, results_folder, filename="confusion_matrix.png"):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plot_path = os.path.join(results_folder, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")


def plot_roc_curve(y_true, y_pred_prob, num_classes, results_folder, class_names, filename="roc_curve.png"):
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
    plot_path = os.path.join(results_folder, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"ROC curve saved to {plot_path}")


# --- Test Bench Function ---

def evaluate_model(model, test_loader, class_names, results_folder, device):
    """
    Evaluate a trained PyTorch model on the test dataset.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test dataset
        class_names: List of class names
        results_folder: Path to save metrics and plots
        device: torch.device
    """

    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = model(inputs)  # Assuming model returns (conv_features, logits)

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plot_confusion_matrix(cm, class_names, results_folder)

    # Classification Report
    report_dict = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    report_path = os.path.join(results_folder, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Classification report saved to {report_path}")

    # ROC Curve
    num_classes = len(class_names)
    plot_roc_curve(all_labels, all_probabilities, num_classes, results_folder, class_names)

    return cm, report_dict, all_probabilities
