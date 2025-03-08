import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/best_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)  # Image size for Swin Transformer


def predict_image(image_path):
    """Predicts whether a given image contains a tumor or not."""

    # Load and preprocess image
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predicted_prob = model.predict(img_array)[0][0]
    predicted_label = "Tumor" if predicted_prob > 0.5 else "No Tumor"

    # Display image with prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label} ({predicted_prob:.2f})")
    plt.axis('off')
    plt.show()


# Example usage
if __name__ == "__main__":
    image_path = input("Enter image path: ")
    if os.path.exists(image_path):
        predict_image(image_path)
    else:
        print("Invalid image path!")
