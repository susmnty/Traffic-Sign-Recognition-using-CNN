import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image  

# Load trained model
model = keras.models.load_model("traffic_net_model.h5")

# Define image size
IMG_SIZE = 224  

# Path to test images directory
test_images_dir = r"C:\\Users\\spkus\\OneDrive\\Desktop\\ML vscode\\Traffic-Net\\Traffic-Net\\images\\"

# Class labels (Update these based on your dataset)
class_labels = {
    0: "Speed Limit",
    1: "Stop Sign",
    2: "Pedestrian Crossing",
    3: "Traffic Light",
    4: "No Entry",
    5: "Turn Left",
    6: "Turn Right",
    7: "Roundabout",
    8: "Railway Crossing",
    9: "Yield"
}

# Initialize lists for true and predicted labels
true_labels = []
pred_labels = []

# Process images
for img_name in os.listdir(test_images_dir):
    img_path = os.path.join(test_images_dir, img_name)

    # Skip non-image files
    if not img_name.endswith(('.jpg', '.png', '.jpeg')):
        print(f"Skipping non-image file: {img_name}")
        continue

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Expand to batch format

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    # Get the true class from the filename (Assumes format: '0_image.jpg')
    try:
        true_class = int(img_name.split("_")[0])  
        true_labels.append(true_class)
        pred_labels.append(predicted_class)
        predicted_label = class_labels.get(predicted_class, "Unknown")
        true_label = class_labels.get(true_class, "Unknown")

        print(f"Image: {img_name} → Predicted: {predicted_label} | True: {true_label}")

    except ValueError:
        print(f"Skipping image with incorrect filename format: {img_name}")