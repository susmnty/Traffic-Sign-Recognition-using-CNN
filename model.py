import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Correct dataset path
dataset_dir = r"C:\\Users\\spkus\\OneDrive\\Desktop\\ML vscode\\Traffic-Net\\Traffic-Net\\images\\"

# Data augmentation & normalization
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)  # Reduced validation split

# Load training and validation data
train_data = datagen.flow_from_directory(dataset_dir, target_size=(128, 128),
                                         batch_size=32, class_mode='categorical', subset='training')

val_data = datagen.flow_from_directory(dataset_dir, target_size=(128, 128),
                                       batch_size=32, class_mode='categorical', subset='validation', shuffle=False)

# Print dataset sizes
print(f"🔍 Training set size: {train_data.samples}")
print(f"🔍 Validation set size: {val_data.samples}")

# Define CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_data, validation_data=val_data, epochs=10)

# Save model in new Keras format
model.save("traffic_sign_model.keras")
print("Training Complete! Model saved as 'traffic_sign_model.keras'")

# Evaluate model (only if validation data is available)
if val_data.samples > 0:
    loss, accuracy = model.evaluate(val_data)
    print(f"Validation Accuracy: {accuracy:.2f}")
    print(f"Validation Loss: {loss:.4f}")
else:
    print("Not enough images for validation. Skipping evaluation.")
