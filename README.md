# Steel Surface Defect Detection using Deep Learning

This project demonstrates a **deep learning-based automated visual inspection system** capable of identifying surface defects in **hot-rolled steel strips**. It brings together **mechanical engineering insights** and **computer vision techniques** to support fast and reliable quality inspection in steel manufacturing.

---

## Problem Statement

Surface defects in cold and hot rolling processes can impact the **performance, appearance, and durability** of steel sheets. Industries still rely heavily on **manual visual inspection**, which is slow, subjective, and prone to human error.

To overcome these limitations, a **deep learning classifier** was developed to automatically recognize steel surface defects from images. This solution aims to:

- Reduce downtime and production losses due to defective batches  
- Improve consistency in quality control  
- Enable real-time monitoring in industrial production lines  

---

## Dataset Description

- **Dataset:** NEU Surface Defect Database (NEU-DET) from Kaggle  
- **Total Samples:** 1800 grayscale images (converted to RGB format)  
- **Resolution:** 200 × 200 pixels  

### Defect Types (6 classes)

1. Crazing  
2. Inclusion  
3. Patches  
4. Pitted Surface  
5. Rolled-in Scale  
6. Scratches  

### Data Split

- **Training Set:** 1440 images  
- **Validation Set:** 360 images (60 per category)  

---

## Deep Learning Workflow

- **Framework:** PyTorch  
- **Base Model:** ResNet18 using Transfer Learning (ImageNet weights)  
- **Changes Applied:**  
  - Replaced the final fully connected layer to output 6 classes  
  - Frozen early layers to retain pretrained feature extraction  

- **Loss Function:** Cross Entropy Loss  
- **Optimizer:** Adam  
- **Training Duration:** 10 epochs  
- **Hardware Used:** NVIDIA RTX 3050 GPU  

---

## Model Performance

| Metric | Value |
|--------|--------|
| **Validation Accuracy** | **95.28%** |
| **Average F1-score** | **0.95** |
| **Best Class Accuracy** | **100%** |
| **Lowest F1-score** | 0.86 (Inclusion) |

A full classification report and confusion matrix were generated to analyze per-class behavior and misclassifications.

---

## Download Model

The trained model file (`model_best.pth`) can be downloaded here:  
**Best_Trained_Model**

---

## Steel Surface Defect Detection
Try the deployed model with the link:
https://steel-surface-defect-detection-u58g8a2kjdcg7aw7kql7ca.streamlit.app/

Here you can 
-upload an image
-view confidence scores for all 6 defect categories

---


## Key Features

- Fully automated surface defect classification for steel inspection  
- Transfer learning ensures strong performance with limited data  
- Fast and lightweight inference suitable for real-time use  
- Supports GPU acceleration and optimized data loading  
- Custom augmentations applied to improve robustness  

---

## Sample Inference Code

```python
image_path = "path/to/defective_steel_image.jpg"
prediction = predict_defect(image_path, model, device)
print("Predicted Defect Type:", prediction)
