# Traffic-Sign-Recognition-using-CNN
A deep learning-based traffic sign classification system using Convolutional Neural Networks (CNNs) with TensorFlow and Keras. This project helps autonomous driving systems detect and classify road signs with high accuracy.

# Objective
The Traffic Sign Recognition System enhances road safety by accurately classifying road signs from images. This is crucial for self-driving vehicles and intelligent traffic management systems.

---

# Project Overview

This project is divided into four major components:

1. **Data Collection (`data_collection.py`)** – Collects traffic sign images for training.

2. **Data Preprocessing (`data_preprocessing.py`)** – Normalizes, resizes, and augments images.

3. **Model Training (`train_model.py`)** – Trains a CNN model for traffic sign classification.

4. **Visualization (`visualization.py`)** – Generates graphs and visual representations of model predictions.

---
# ⚙️ System Requirements
Python 3.x

TensorFlow

Keras

OpenCV

NumPy

Matplotlib

---

# Install dependencies:
- pip install -r requirements.txt

---
# Model Architecture

The CNN model consists of:

- Convolutional Layers – Extract features from images

- Max Pooling Layers – Reduce dimensionality

- Dense Layers – Perform classification

- Softmax Activation – Predict probabilities

![Image](https://github.com/user-attachments/assets/1bc2cd44-fe82-4932-8a71-4460a044197c)
---
# Results & Performance

![Image](https://github.com/user-attachments/assets/2042167b-60e6-481d-817f-4a4dbb684729)

# Accuracy & Loss Graphs

![Image](https://github.com/user-attachments/assets/a5f29968-0603-46ee-b3d8-10a212160be1)

# Visualization

The Visualization Module (`visualization.py`) provides graphical representations of:

- Model Accuracy and Loss 
- Traffic Sign Class Predictions 
- Confusion Matrix for Misclassifications 

# Run the script to visualize model performance:
python src/visualization.py

---
# Running the Project

1. Clone the repository
git clone https://github.com/yourusername/Traffic-Sign-Recognition.git
cd Traffic-Sign-Recognition

2. Train the Model 
python src/train_model.py

3. Evaluate the Model 
python src/test_model.py

4. Visualize Results
python src/visualization.py

---
🤝 Contributing
Feel free to contribute! Submit pull requests or report issues.

























