# 😷 Face Mask Detection using VGG16

A deep learning-based face mask detection system built with **VGG16**, fine-tuned on a custom face mask dataset. The model detects whether a person is wearing a mask or not from images or webcam feed in real-time.

---

## 🚀 Features

- ✅ Binary classification: `Mask` vs `No Mask`
- 🧠 VGG16 (pre-trained on ImageNet) + custom top layers
- 🎥 Real-time face mask detection via webcam (OpenCV)
- 📸 Inference from static images
- 📈 High validation accuracy (~95%)
- 🔄 Data augmentation and preprocessing pipeline

---

## 🗃️ Dataset

- **Main Source**: [Face Mask Dataset by Prajna Bhandary](https://github.com/prajnasb/observations/tree/master/experiements/data)
- Includes two categories:
  - `with_mask`
  - `without_mask`
- Custom data augmentation included
- You can add more images to improve performance

---

## 🧠 Model Architecture

- **Base**: VGG16 (`include_top=False`, `weights='imagenet'`)
- **Added Layers**:
  - `GlobalAveragePooling2D`
  - `Dense(128, activation='relu')`
  - `Dropout(0.5)`
  - `Dense(1, activation='sigmoid')` *(binary classification)*

---

