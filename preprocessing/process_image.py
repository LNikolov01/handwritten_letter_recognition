import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess the input image minimally to match EMNIST's format.
    - Convert to grayscale
    - Invert colors (black on white to white on black)
    - Resize to 28x28 while preserving aspect ratio
    - Normalize pixel values to [0,1]
    """

    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"❌ Error: Unable to read image from {image_path}")
        return None

    # Invert colors (EMNIST uses white text on black background)
    img = cv2.bitwise_not(img)

    # Resize the image to 28x28
    processed = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize pixel values to range [0,1]
    processed = processed.astype(np.float32) / 255.0

    # Reshape for model input (1, 28, 28, 1)
    processed = np.expand_dims(processed, axis=0)  # Add batch dimension
    processed = np.expand_dims(processed, axis=-1)  # Add channel dimension

    return processed