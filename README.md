# 🌿 PlantGuard AI: Plant Disease Classification

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/yourusername/plantguard-ai)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)

## 📌 Overview

PlantGuard AI is a deep learning-based system for classifying plant diseases from leaf images. The system implements **two complementary approaches**:

| Approach | Method | Accuracy | Best For |
|----------|--------|----------|----------|
| **Approach 1** | CNN Feature Extractor + SVM Classifier | **96.0%** | Interpretability, research, smaller datasets |
| **Approach 2** | End-to-End Fine-tuning (EfficientNet-B0) | **99.62%** | Production deployment, maximum accuracy |

Using transfer learning with **EfficientNet-B0** pretrained on ImageNet, the system achieves **99.62% accuracy** across **38 disease classes** covering 14 plant species.

### Why Two Approaches?

- **SVM Approach** (Approach 1): Features extracted from frozen CNN backbone → SVM classification. Great for understanding which features matter, faster training, and when computational resources are limited.

- **Deep Learning Approach** (Approach 2): End-to-end fine-tuning of all layers. Maximum accuracy for production use with real-time inference.

## 🎯 Key Results

| Model / Approach | Accuracy | F1-Score (Macro) | Precision | Recall |
|-----------------|----------|------------------|-----------|--------|
| **CNN + SVM** (Approach 1) | **96.0%** | 0.94 | 0.96 | 0.96 |
| **EfficientNet-B0** (Approach 2) | **99.62%** | 0.994 | 0.996 | 0.996 |

### SVM Details (Approach 1)
- **Feature Extractor**: EfficientNet-B0 backbone (frozen)
- **Feature Dimension**: 1280
- **Classifier**: Linear SVM (C=1.0, kernel='linear')
- **Preprocessing**: StandardScaler on extracted features
- **Training Time**: ~15 minutes (feature extraction + SVM)
- **Performance**: 96% accuracy, excellent baseline

### EfficientNet-B0 Details (Approach 2)
- **Architecture**: Compound scaling of width, depth, resolution
- **Total Parameters**: 4,056,226
- **Model Size (fp32)**: 16.2 MB
- **Training Time**: ~108 minutes on Tesla T4 GPU
- **Inference Speed**: ~25 ms per image

## 📁 Project Structure

```
plantguard-ai/
├── app.py                          # Gradio web application (supports both models)
├── requirements.txt                # Python dependencies
├── best_efficientnet_b0.pt        # Trained EfficientNet checkpoint
├── svm_efficientnet.joblib        # Trained SVM model
├── scaler_efficientnet.joblib     # Feature scaler for SVM
├── class_mapping.json              # Class index to name mapping
├── hyperparameters.json            # Optimal hyperparameters
├── README.md                       # Project documentation
├── PlantGuard_Notebook.ipynb       # Complete training notebook
├── checkpoints/                    
│   ├── best_efficientnet_b0_end_to_end.pt
│   ├── svm_efficientnet.joblib
│   └── config.json
├── runs/plantvillage/              # TensorBoard logs
├── data/                           # Dataset (PlantVillage)
└── images/
    ├── demo_screenshot.png
    ├── confusion_matrix.png
    ├── learning_curves.png
    └── grad_cam_sample.png
```

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| **Deep Learning** | PyTorch, Torchvision |
| **Classical ML** | scikit-learn (SVM, StandardScaler) |
| **Architecture** | EfficientNet-B0 |
| **Optimization** | Optuna, AdamW, ReduceLROnPlateau |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, TensorBoard |
| **Web Interface** | Gradio |
| **Cloud Platform** | Hugging Face Spaces, Google Colab |

## 📊 Features

### 1. Dual Approach Architecture

**Approach 1: CNN + SVM (Feature Extraction)**
```python
# Extract features from frozen backbone
backbone = nn.Sequential(*list(model.children())[:-1])
features = backbone(img_tensor).view(1, -1)

# Scale and classify with SVM
features_scaled = scaler.transform(features)
prediction = svm.predict(features_scaled)
probabilities = get_svm_probabilities(svm, features_scaled)
```

**Approach 2: End-to-End Deep Learning**
```python
# Fine-tune all layers
model = build_model("efficientnet_b0", num_classes=38, dropout=0.168)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.000665)
# Train end-to-end with mixed precision
```

### 2. Model Comparison

| Feature | CNN + SVM | EfficientNet-B0 |
|---------|-----------|-----------------|
| **Accuracy** | 96.0% | 99.62% |
| **Training Time** | ~15 min | ~108 min |
| **Inference Speed** | ~20 ms | ~25 ms |
| **Interpretability** | High (linear weights) | Medium (Grad-CAM) |
| **Memory Usage** | Low (SVM + scaler) | Medium (16 MB) |
| **Best For** | Research, baselines | Production |

### 3. Training Pipeline
- ✅ Mixed Precision (AMP) for 2-3x speedup
- ✅ EMA (Exponential Moving Average) for better generalization
- ✅ Gradient Clipping (max_norm=1.0)
- ✅ Early Stopping (patience=7 epochs)
- ✅ Checkpointing (best model + periodic)
- ✅ SVM training on extracted features

### 4. Hyperparameter Optimization (Optuna)

**Approach 2 (End-to-End) Optimal Values:**
| Parameter | Search Space | Optimal Value |
|-----------|--------------|---------------|
| Learning Rate | 1e-5 to 1e-2 (log) | **6.65e-4** |
| Dropout | 0.1 to 0.5 | **0.168** |
| Weight Decay | 1e-5 to 1e-2 (log) | **1e-4** |
| Batch Size | [16, 32, 64] | **32** |

**Approach 1 (SVM) Optimal Values:**
| Parameter | Search Space | Optimal Value |
|-----------|--------------|---------------|
| SVM Kernel | linear, rbf, poly | **linear** |
| C (regularization) | 0.1 to 10 (log) | **1.0** |

### 5. Data Pipeline
- **Augmentation**: Random flips, rotation (±30°), color jitter, random resized crop
- **Stratified Split**: 70/15/15 train/val/test
- **Class Weights**: Handles 36x class imbalance
- **ImageNet Normalization**: μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]

### 6. Debugging & Diagnostics
- **Gradient Flow Analysis**: Detects vanishing/exploding gradients
- **NaN/Inf Detection**: Early failure detection
- **Grad-CAM**: Visual explanations for model predictions
- **Confusion Matrix**: Per-class performance visualization
- **Learning Curves**: Train/val loss and accuracy over epochs

### 7. Web Application
- **Dark Botanical Theme**: Premium UI with glassmorphism effects
- **Real-time Analysis**: Automatic on image upload
- **Dual Model Support**: Toggle between DL and SVM
- **Top-5 Predictions**: Confidence bars for multiple candidates
- **Filename Display**: Shows original uploaded filename
- **Mobile Responsive**: Works on all devices

## 🏃 How to Run Locally

### Prerequisites
- Python 3.9 or higher
- CUDA-capable GPU (optional, CPU works but slower)
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/plantguard-ai.git
cd plantguard-ai
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
```bash
# Option 1: Kaggle API
kaggle datasets download abdallahalidev/plantvillage-dataset -p ./data --unzip

# Option 2: Google Drive (see notebook for file ID)
python download_dataset.py
```

5. **Download trained models**
```bash
# EfficientNet-B0 (Approach 2)
gdown https://drive.google.com/uc?id=1jxrQu-luSzZ6jmZ6MtzC8RSLOoYcI8t3 -O best_efficientnet_b0.pt

# SVM model and scaler (Approach 1)
gdown https://drive.google.com/uc?id=YOUR_SVM_FILE_ID -O svm_efficientnet.joblib
gdown https://drive.google.com/uc?id=YOUR_SCALER_FILE_ID -O scaler_efficientnet.joblib
```

6. **Run the web application**
```bash
python app.py
```

### Training from Scratch

Open `PlantGuard_Notebook.ipynb` in Google Colab or Jupyter and run all cells. The notebook covers:

1. Environment setup and dataset download
2. Configuration and GPU setup
3. Data preparation with augmentation
4. Model architecture definition
5. Gradient diagnostics
6. SVM training (Approach 1)
7. Hyperparameter tuning with Optuna
8. End-to-end training (Approach 2)
9. Evaluation and visualization

## 📈 Results & Visualizations

### Confusion Matrix (EfficientNet-B0)
![Confusion Matrix](images/confusion_matrix.png)

### Learning Curves
![Learning Curves](images/learning_curves.png)

### Grad-CAM Visualization
![Grad-CAM](images/grad_cam_sample.png)

### Web Application Interface
![App Demo](images/demo_screenshot.png)

## 🔬 Dataset

**PlantVillage Dataset**
- **Total Images**: 54,305
- **Classes**: 38
- **Plant Species**: 14 (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato)
- **Format**: RGB color images
- **Source**: [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

### Class Distribution Highlights
| Plant | Healthy Samples | Diseased Samples | Disease Types |
|-------|----------------|------------------|---------------|
| Apple | 2,463 | 2,586 | 3 |
| Tomato | 3,913 | 12,505 | 9 |
| Potato | 1,520 | 2,354 | 2 |
| Grape | 1,703 | 3,435 | 4 |
| Orange | 826 | 4,107 | 1 |

**Imbalance Ratio**: 36.23x (max 5,507 vs min 152 samples)

## 🧠 Methodology

### Approach 1: CNN + SVM Pipeline
```
Input Image (224×224×3)
       ↓
ImageNet Normalization
       ↓
Pretrained EfficientNet-B0 (Frozen Backbone)
       ↓
Feature Extraction (1280-dim vector)
       ↓
StandardScaler Normalization
       ↓
Linear SVM Classifier (C=1.0)
       ↓
Class Prediction (38 classes)
```

### Approach 2: End-to-End Deep Learning Pipeline
```
Input Image (224×224×3)
       ↓
ImageNet Normalization
       ↓
Data Augmentation
       ↓
EfficientNet-B0 (Fine-tuned)
       ↓
Dropout (p=0.168)
       ↓
Linear Layer (1280 → 38)
       ↓
Softmax → Class Probabilities
```

### Training Strategy (Approach 2)
- **Phase 1**: Linear warmup (first 3 epochs)
- **Phase 2**: End-to-end fine-tuning (all layers unfrozen)
- **Learning Rate**: 6.65e-4
- **Batch Size**: 32
- **Epochs**: 30 (early stopping at 18)
- **Optimizer**: AdamW (β1=0.9, β2=0.999, eps=1e-8)
- **Loss Function**: CrossEntropyLoss with class weights

### SVM Training Strategy (Approach 1)
1. Extract features from frozen EfficientNet-B0 backbone
2. Normalize features with StandardScaler
3. Train Linear SVM (C=1.0)
4. Save model and scaler for deployment

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size (16 or 8) |
| NaN loss detected | Lower learning rate or check data normalization |
| Vanishing gradients | Use AdamW, reduce weight decay, check activation functions |
| Slow training | Enable mixed precision (AMP), increase batch size |
| Poor accuracy | Increase epochs, reduce dropout, unfreeze more layers |
| SVM not loading | Check file paths and joblib version compatibility |

### Getting Help
- Check the [GitHub Issues](https://github.com/yourusername/plantguard-ai/issues) page
- Review the training notebook for detailed logging
- Enable TensorBoard for real-time monitoring

## 📝 Future Work

- [ ] **Knowledge Distillation**: Compress EfficientNet-B0 into smaller student model
- [ ] **Real-field Images**: Fine-tune on field-captured (vs. laboratory) images
- [ ] **Multi-disease Detection**: Handle multiple diseases per leaf
- [ ] **Mobile Deployment**: Convert to TensorFlow Lite / ONNX for edge devices
- [ ] **Severity Estimation**: Predict disease progression stage
- [ ] **Treatment Recommendations**: Suggest organic/chemical treatments
- [ ] **Ensemble Methods**: Combine SVM + DL for improved robustness

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to new functions
- Update tests if applicable
- Update README with new features

## 📄 License

Distributed under the MIT License. See `LICENSE` file for more information.

## 🙏 Acknowledgments

- **PlantVillage Dataset** - Hughes, D. P., & Salathé, M. (2015)
- **EfficientNet** - Tan & Le, Google Research, ICML 2019
- **Optuna** - T. Akiba et al., Preferred Networks, KDD 2019
- **Gradio** - Hugging Face for easy ML web apps
- **PyTorch** - Meta AI for deep learning framework
- **scikit-learn** - For SVM implementation and preprocessing

## 📧 Contact

Lujain Hesham - lujainnheshamm@gmail.com

Project Link: [https://github.com/yourusername/plantguard-ai](https://github.com/yourusername/plantguard-ai)

Hugging Face Space: [https://huggingface.co/spaces/yourusername/plantguard-ai](https://huggingface.co/spaces/LujainHesham01/plant-health-scanner)

---

<div align="center">
  <sub>Built with ❤️ for sustainable agriculture and global food security</sub>
</div>
```
   - Included SVM training methodology
3. **Dual Approach Focus** - Clear distinction between Approach 1 (SVM) and Approach 2 (EfficientNet)
4. **Updated Performance Metrics** - Both approaches now clearly presented with their metrics
