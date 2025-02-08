import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import string

# Fix Image Rotation for EMNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.rot90(x, k=1, dims=[1, 2])),  # Ensure correct orientation
    transforms.Lambda(lambda x: torch.flip(x, dims=[1]))  # Flip Horizontally
])

# Load EMNIST dataset
train_dataset = torchvision.datasets.EMNIST(root="./data", split="letters", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.EMNIST(root="./data", split="letters", train=False, download=True, transform=transform)

# Convert dataset to NumPy arrays
x_train, y_train = zip(*train_dataset)
x_test, y_test = zip(*test_dataset)

x_train = np.array([img.numpy().squeeze() for img in x_train])
x_test = np.array([img.numpy().squeeze() for img in x_test])
y_train = np.array(y_train)
y_test = np.array(y_test)

# Normalize pixel values to range [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels from 1-26 to 0-25
y_train -= 1
y_test -= 1

# One-hot encoding for labels
y_train = to_categorical(y_train, num_classes=26)
y_test = to_categorical(y_test, num_classes=26)

# Configure CNN Model with Less Regularization & Leaky ReLU
model = Sequential([
    Conv2D(64, (3,3), kernel_initializer='he_normal', input_shape=(28, 28, 1)),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), kernel_initializer='he_normal'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(2,2),
    
    Conv2D(256, (3,3), kernel_initializer='he_normal'),
    LeakyReLU(alpha=0.1),
    
    Flatten(),
    Dense(512, kernel_initializer='he_normal'),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),  # Reduced dropout
    Dense(26, activation='softmax')  # 26 Classes for A-Z
])

# Initial Learning Rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Reduce Learning Rate if Validation Loss Stops Improving
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5)

# Train the Model
history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=64,
                    epochs=20,
                    callbacks=[lr_scheduler])

# Save Model
model.save("cnn_model_letters.h5")

# Evaluate Performance
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# Generate Classification Report
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Convert indices to letters
label_map = list(string.ascii_uppercase)
y_pred_classes = [label_map[i] for i in y_pred_classes]
y_true = [label_map[i] for i in y_true]

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Plot Training History
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Save & Show the plots
plt.savefig("loss_accuracy_graph.png")  # Save plots as an image
plt.show()