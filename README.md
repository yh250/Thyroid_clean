# Thyroid_clean
> **STEM Augmentation | PyTorch Training | Multi-Method XAI | Automated Evaluation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A cleaned-up version of the Thyroid repo with only essential code. 

---

## 📂 Directory Structure

```text
.
├── Torch_models/           # Model architectures (AlexNet, ResNet, MobileNet, CNN)
├── Xai_classes/            # Modular XAI logic (GradCAM, GradCAM++, Saliency, Occlusion)
├── utils/                  # Standalone Evaluation Tools
│   ├── test.py             # Performance metric logic (ROC, Confusion Matrix, JSON logs)
│   └── test_bench2.py      # Standalone driver for validating trained weights
├── STEM2.py               # STEM (Spatio-Temporal Edge Mapping) engine
├── Run_Xai.py              # Configuration-driven post-hoc visualization driver
├── Torch_grad_train.py     # Main training pipeline with integrated Grad-CAM hooks
└── results/                # Auto-generated timestamped metrics and plots
```

---

## 🛠️ Environment & Setup

### **1. Installation**
Ensure you have Python 3.8+ and a CUDA-enabled GPU (optional but recommended).
```bash
# Clone the repository
git clone https://github.com/yh250/Thyroid_clean.git
cd Thyroid_clean

# Install dependencies
pip install torch torchvision numpy opencv-python Pillow matplotlib seaborn scikit-learn
```

### **2. Silicon Mac (M1/M2/M3) Setup**
If you encounter SSL errors while downloading pretrained weights, the scripts include a built-in fix:

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

---

##  Module Documentation

This project is built for **independence**. Each part can be used without the others, though they share a common directory logic.

### **1. STEM Augmentation (`STEM2.py`)**
**Standalone Tool.** Use this to preprocess raw scans into edge-mapped datasets before training.
* **Configuration:** Edit the `train_data_dir` and `output` paths in the script.
* **Run:** `python STEM2.py`

### **2. Training Pipeline (`Torch_grad_train.py`)**
The primary orchestrator for model training.

* **Modular Selection:** Switch models (e.g., `GradCAMReadyAlexNet`) via the `model` instantiation line.
* **Toggling Features:** * **Evaluation:** To disable post-training testing, comment out the `--- Evaluation ---` section.
    * **Grad-CAM:** To disable the integrated visualization, comment out the `Grad-CAM Visualization Example` block.
* **Early Stopping:** Defaults to 30 epochs of patience; adjust via `patience_counter`.
* **Run:** `python Torch_grad_train.py`

### **3. Standalone Testbench (`utils/test_bench2.py`)**
**Standalone Tool.** Validate existing `.pth` weights against any dataset without re-training.
* **Configuration:** Update `best_model_path` and `test_data_dir`.
* **Note:** Ensure the `transform` matches your training settings exactly.
* **Run:** `python utils/test_bench2.py`

### **4. Advanced XAI Driver (`Run_Xai.py`)**
A post-hoc visualization tool to audit model focus regions.

| Method | Type | Description |
| :--- | :--- | :--- |
| **Grad-CAM / ++** | Gradient-based | High-importance spatial regions. |
| **Saliency Maps** | Pixel-based | Identifies pixels influencing the output. |
| **Occlusion** | Perturbation | Masks image sections to check confidence drops. |

* **Usage:** Update the `config` dictionary in the `if __name__ == "__main__":` block.
* **Run:** `python Run_Xai.py`

---

## ⚠️ Critical Troubleshooting (Avoid these common errors)

* **Input Channels:** Most medical scans are grayscale. Always ensure `input_channels=1` is passed to the model constructor.
* **Target Layers:** Gradient-based XAI (Grad-CAM) requires a "target layer" hook. If adding a new architecture, you **must** update the `get_target_layer` function in `Run_Xai.py`.
* **GPU Memory:** If you hit `CUDA out of memory`, reduce the `batch_size` to 16 or 8 in the config sections.
* **Pathing:** Always verify directory paths if moving from Windows (backslashes) to Linux/Mac (forward slashes). Use `os.path.join` for portability.

---

## 📊 Evaluation & Results
All scripts output results to a timestamped directory (e.g., `AlexNet_STEM_2026.../`) containing:
* `classification_report.json`: Precision, Recall, F1 scores.
* `confusion_matrix.png`: Heatmap of classification accuracy.
* `roc_curve.png`: Multi-class sensitivity/specificity analysis.
* `xai_... .png`: Side-by-side heatmaps showing original scans vs. model attention.

---
