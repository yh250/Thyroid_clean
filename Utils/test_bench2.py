from torch.utils.data import DataLoader, Dataset

from test import evaluate_model
import torch
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torchvision import datasets, transforms

# IMPORTANT: USE YOUR SYSTEM DIRECTORY TO IMPORT MODEL ARCHITECTURES ( change root with folder name/directory path)
from ROOT.Torch_models.gctch_cnn import GradCAMReadyCNN
from ROOT.Torch_models.gctch_alexnet import GradCAMReadyAlexNet
from ROOT.Torch_models.gctch_Resnet18 import GradCAMReadyResNet18
from ROOT.Torch_models.gctch_mobilenet import GradCAMReadyMobileNetV2

"""
     STANDALONE TESTING DRIVER FUNCTION FOR TRAINED WEIGHTS 
     Args:
            test_data_dir: PATH TO TEST DATASET
            model: CALL TO MODEL ( IMPORT CORRECT ARCHITECTURE) 
            best_model_path: PATH to trained model weights we are testing
"""

#config
test_data_dir = "/Users/harshyadav/Desktop/Aiims/cropped_dtgpr"

model= GradCAMReadyAlexNet(4,1)
best_model_path= "/Remote/Codebase/Best_weights /grad_AlexNet_Pytorch_Split_STEM_100_20250527-065259/best_model_pytorch.pth"
#"/Users/harshyadav/Desktop/pythonProject/Remote/Codebase/Best_weights /grad_ResNet18_Pytorch_Split_STEM_100_20250531-095911/best_model_pytorch.pth"
#"/Users/harshyadav/Desktop/pythonProject/Remote/Codebase/Best_weights /grad_MobileNET_Pretrained_PyTorch_Balanced_braintumor_20250527-064225/best_model_pytorch.pth"
    #"/Users/harshyadav/Desktop/pythonProject/Remote/Codebase/Best_weights /grad_CNN_PYtorch_Split_STEM_100_20250527-064818/best_model_pytorch.pth"


input_shape = (256, 256, 1)  # (Height, Width, Channels) - Consistent with your Keras config
num_classes = 4
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)


# ---Transformations---
# IMPORTANT: use same transformations as used in training

transform = transforms.Compose([
    transforms.Resize((input_shape[0], input_shape[1])),  # Resize to (H, W)
    transforms.Grayscale(num_output_channels=input_shape[2]),  # Convert to 1 channel for grayscale
    transforms.ToTensor(),
])

# ---Data loader---

test_dataset = datasets.ImageFolder(
    root=test_data_dir,
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ---Class names---
# Get class names from the dataset (they are sorted by default by ImageFolder)
class_names = test_dataset.classes  # Use full_train_dataset to get all classes
print("Class names and their corresponding indices:")
for i, class_name in enumerate(class_names):
    print(f"Class: {class_name}, Index: {i}")

# Print basic class distribution (for the full dataset before split)
print("\nFull Dataset Class Distribution:")
class_counts = {}
for _, idx in test_dataset.imgs:  # (image_path, class_idx) tuples
    class_name = class_names[idx]
    class_counts[class_name] = class_counts.get(class_name, 0) + 1
for class_name, count in class_counts.items():
    print(f"Class: {class_name}, Count: {count}")


cm, report_dict, all_probabilities = evaluate_model(
    model=model,
    test_loader=test_loader,
    class_names=class_names,
    # Name of folder Results will be saved to, Can automate new folder creation for each run.
    results_folder="/Users/harshyadav/Desktop/pythonProject/Aiims_test/Test_result/22jan",
    device=device
)
