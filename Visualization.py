import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Correct dataset path
dataset_dir = r"C:\\Users\\spkus\\OneDrive\\Desktop\\ML vscode\\Traffic-Net\\Traffic-Net\\images\\"

# Data Augmentation & Normalization
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load Training & Validation Data
train_data = datagen.flow_from_directory(dataset_dir, target_size=(128, 128),
                                         batch_size=32, class_mode='categorical', subset='training')

val_data = datagen.flow_from_directory(dataset_dir, target_size=(128, 128),
                                       batch_size=32, class_mode='categorical', subset='validation')

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

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_data, validation_data=val_data, epochs=10)

# Save Model
model.save("traffic_sign_model.keras")
print("Training Complete! Model saved as 'traffic_sign_model.keras'")

# Visualizing Sample Images from Dataset
class_names = list(train_data.class_indices.keys())

images, labels = next(train_data)
plt.figure(figsize=(10, 5))
for i in range(5):  # Display 5 images
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i])
    plt.title(class_names[np.argmax(labels[i])])
    plt.axis("off")
plt.show()

# Plot Accuracy & Loss
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Accuracy")

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Model Loss")

plt.show()

# Model Predictions on Test Data
test_images, test_labels = next(val_data)
predictions = model.predict(test_images)

plt.figure(figsize=(10, 5))
for i in range(5):  # Show 5 sample predictions
    plt.subplot(1, 5, i + 1)
    plt.imshow(test_images[i])
    plt.title(f"Pred: {class_names[np.argmax(predictions[i])]} \nActual: {class_names[np.argmax(test_labels[i])]}")
    plt.axis("off")
plt.show()

# Confusion Matrix
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_data.classes

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()