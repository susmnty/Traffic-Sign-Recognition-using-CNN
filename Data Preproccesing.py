import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Get dataset path
dataset_dir = r"C:\\Users\\spkus\\OneDrive\\Desktop\\ML vscode\\Traffic-Net\\Traffic-Net"

# Verify dataset exists
if not os.path.exists(dataset_dir):
    raise ValueError(f"Dataset directory not found: {dataset_dir}")

# Check if dataset has subfolders
class_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
if not class_folders:
    raise ValueError("No class subfolders found! Move images into separate class folders inside the dataset directory.")

# Parameters
IMG_SIZE = 64
BATCH_SIZE = 32
NUM_CLASSES = len(class_folders)  # Auto-detect number of classes

# Data augmentation & normalization
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load training and validation data
train_data = datagen.flow_from_directory(dataset_dir, target_size=(IMG_SIZE, IMG_SIZE),
                                         batch_size=BATCH_SIZE, class_mode='categorical', subset='training')

val_data = datagen.flow_from_directory(dataset_dir, target_size=(IMG_SIZE, IMG_SIZE),
                                       batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

# Define CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  # Output layer
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_data, validation_data=val_data, epochs=10)

# Save model
model.save("traffic_sign_model.h5")

print("Training Complete! Model saved as 'traffic_sign_model.h5'")
