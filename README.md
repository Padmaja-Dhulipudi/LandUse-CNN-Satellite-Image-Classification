## 🌍 Land use Classification using CNN and Transfer Learning

This project deals with classifying satellite images in different land-use categories using Deep Learning Techniques.  
**Convolutional Neural Networks (CNNs)** and **Transfer Learning using MobileNetV2** are used to train the model on the **EuroSAT dataset**.

The goal of the project is to automatically extract land-use patterns from satellite images for applications like:
- Environmental surveillance- City planning

- Agricultural analyses

Remote sensing research


--- 

# 📌 Characteristics

- Classification of satellite images- CNN deep learning pipeline
- MobileNetV2 based Transfer Learning
- Data augmentation for improved generalization
- Support for training and prediction of models
- Compatible with Google Colab- save and load models

--- 

# 🛠️ Technologies Used

- Python
- TensorFlow 
- NumPy
- PIL
- Google Colab

---
# 📂 Dataset

## EuroSAT Dataset

The project uses the EuroSAT dataset, which contains RGB satellite images grouped into 10 different land-use categories.

### Classes Included
- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

Dataset Link:  
https://www.kaggle.com/datasets/apollo2506/eurosat-dataset

---

# 📁 Dataset Structure

```bash
EuroSAT/
│
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

---

# 🚀 Project Workflow

The overall workflow of the project is:

1. Load and preprocess satellite images
2. Apply data augmentation techniques
3. Train the model using MobileNetV2 transfer learning
4. Validate model performance on unseen data
5. Save the trained model
6. Predict land-use categories for new images

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/Padmaja-Dhulipudi/LandUse-CNN-Satellite-Image-Classification.git
cd LandUse-CNN-Satellite-Image-Classification
```

---

## Install Required Libraries

```bash
pip install tensorflow numpy pillow matplotlib
```

---

# ▶️ Running the Project

## Open in Google Colab

Colab Notebook Link:

https://colab.research.google.com/drive/18FsWq390xDTXfawTFrRytUYdcRnQB_zO

---

## Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Upload Dataset

Upload the EuroSAT dataset folder to your Google Drive and update the dataset path:

```python
DATASET_PATH = "/content/drive/MyDrive/EuroSAT"
```

---

## Train the Model

Run all notebook cells or execute:

```bash
python landuse_model.py
```

---

# 🧠 Model Architecture

This project uses MobileNetV2 with Transfer Learning.

MobileNetV2 was chosen because it:
- is lightweight and efficient
- trains faster compared to larger models
- provides better accuracy for image classification tasks
- works well on limited computational resources like Google Colab

Using pretrained ImageNet weights helped improve the model’s performance and reduced training time.

---

# 📊 Data Augmentation

To improve generalization and reduce overfitting, the following augmentation techniques were applied:

```python
rotation_range=25
zoom_range=0.2
horizontal_flip=True
brightness_range=[0.8,1.2]
```

---

# 💾 Saving the Model

```python
model.save("landuse_model.h5")
```

---

# 🔍 Prediction Example

```python
print("Prediction:", predict("img.jpg"))
```

---

# 📈 Results

The transfer learning model performed significantly better than the basic CNN model.

| Model | Approximate Accuracy |
|------|----------------------|
| Basic CNN | 70% – 80% |
| MobileNetV2 | 85% – 92% |

---

# 📷 Sample Output

```bash
Prediction: Forest
```

---

# 🔮 Future Improvements

Some possible improvements for the project include:

- Deploying the model using Streamlit or Flask
- Adding confusion matrix and performance visualizations
- Experimenting with Vision Transformers (ViT)
- Using multispectral satellite imagery
- Further fine-tuning the pretrained model

---

# 👩‍💻 Author

Padmaja Dhulipudi

---

# 📚 References

- TensorFlow Documentation
- EuroSAT Dataset
- MobileNetV2 Research Paper
- Google Colab
