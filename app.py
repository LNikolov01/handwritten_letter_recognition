import subprocess
import cv2
import numpy as np
import tensorflow as tf
import string
from preprocessing.process_image import preprocess_image

MODEL_PATH = "models/cnn_model_letters.h5"  # Updated to use letter model

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def predict_letter(image_path, model):
    # Preprocess the image
    processed_image = preprocess_image(image_path)  # Should return a NumPy array

    # Debugging: Save preprocessed image
    debug_image = (processed_image[0, :, :, 0] * 255).astype(np.uint8)  # Convert back to uint8
    cv2.imwrite("test_images/debug_preprocessed.png", debug_image)
    print("✅ Saved debug preprocessed image as 'test_images/debug_preprocessed.png'.")

    # Check if preprocessing returned a valid image
    if processed_image is None:
        print("❌ Error: Image preprocessing failed!")
        return None, None

    # Get prediction probabilities
    predictions = model.predict(processed_image)  # Returns an array of probabilities for each letter (A-Z)

    predicted_index = np.argmax(predictions)  # Get the letter with the highest probability
    confidence = np.max(predictions)  # Get the confidence score

    # Convert index to letter (0=A, 1=B, ..., 25=Z)
    predicted_letter = string.ascii_uppercase[predicted_index]

    return predicted_letter, confidence

if __name__ == "__main__":
    print("Launching letter drawing interface...")
    
    # Run draw_letter.py as a subprocess
    subprocess.run(["python3", "drawing_interface/draw_letter.py"])
    
    IMAGE_PATH = "test_images/drawn_letter.png"  # Updated image path
    
    # Load model
    model = load_model()

    # Predict the letter
    predicted_letter, confidence = predict_letter(IMAGE_PATH, model)

    # Display the results in terminal
    print(f"Predicted Letter: {predicted_letter}")
    print(f"Confidence: {confidence * 100:.2f}%")  # Converted to percentage format

    # Create a blank white image
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255

    # Display the predicted letter
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, predicted_letter, (60, 120), font, 3, (0, 0, 0), 5, cv2.LINE_AA)

    # Show the window with the predicted letter
    cv2.imshow("Predicted Letter", img)
    cv2.waitKey(3000)  # Display for 3 seconds
    cv2.destroyAllWindows()