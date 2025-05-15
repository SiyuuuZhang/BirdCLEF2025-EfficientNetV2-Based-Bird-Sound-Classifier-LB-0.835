# BirdCLEF2025 EfficientNetV2 Based Bird Sound Classifier LB 0.835

## **Project Overview**
This repository presents an end-to-end solution for the BirdCLEF2025 Kaggle competition, focusing on **audio-based bird species classification**. The proposed method employs **EfficientNetV2** as the backbone model, combined with **Focal Loss optimization**, **spectrogram augmentation**, and **model ensembling** to achieve robust performance on the 2025 dataset containing 5-second birdcall recordings.

## **Methodology**

### **Model Architecture**
- **Backbone**: `tf_efficientnetv2_s.in21k_ft_in1k` (pretrained on ImageNet-21K)
- **Input Representation**:
  - Audio files (`.ogg`) are converted to **Mel-spectrograms** (256×256 resolution)
  - Parameters: `N_FFT=1024`, `HOP_LENGTH=512`, `N_MELS=128`
- **Feature Extraction**:
  - Adaptive average pooling applied to final features
  - Linear classifier with 206 output units (corresponding to primary bird labels)

### **Training Strategy**
- **Loss Function**:
  - Custom **Focal Loss + BCEWithLogits** combination (weights: `bce_weight=0.6`, `focal_weight=1.4`)
  - Addresses class imbalance in multi-label classification
- **Data Augmentation**:
  - **Spectrogram masking** (frequency/time domain)
  - **Mixup** (α=0.5) for improved generalization
- **Optimization**:
  - **AdamW** optimizer (`lr=5e-4`, `weight_decay=1e-5`)
  - **Cosine Annealing** learning rate scheduler

### **Inference Enhancements**
- **Model Ensembling**:
  - Aggregates predictions from multiple folds (folds 0-3)
  - Final probabilities thresholded at **0.7** for submission

## **References**
1. [Kaggle BirdCLEF2025 Dataset](https://www.kaggle.com/competitions/birdclef-2025)
2. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML*.
3. Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. *CVPR*.