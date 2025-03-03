import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import math

from transformers import TFSwinForImageClassification, AutoImageProcessor

# Define paths
train_path = r"D:\ED\braintumor\data\Training"
test_path = r"D:\ED\braintumor\data\Testing"

# Original images are 512x512, so we choose 448x448
# (since 448 is divisible by 7, matching the window size requirement of swin-tiny)
img_size = (448, 448)

# Initialize the image processor; we will pass the desired size during processing
image_processor = AutoImageProcessor.from_pretrained('microsoft/swin-tiny-patch4-window7-224',use_fast = False)

def load_images_from_folder(folder):
    images = []
    labels = []
    # Expecting folder structure: folder/notumor and folder/tumor
    for label, subfolder in enumerate(['notumor', 'tumor']):
        path = os.path.join(folder, subfolder)
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Load and resize the image to our chosen size
                img = load_img(img_path, target_size=img_size)
                img = img_to_array(img)
                # Use the image processor and force resizing to img_size
                img = image_processor(images=img, size=img_size, return_tensors="np")['pixel_values'][0]
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load and preprocess data
train_images, train_labels = load_images_from_folder(train_path)
test_images, test_labels = load_images_from_folder(test_path)

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Load the pre-trained Swin Transformer model and adapt it for binary classification (2 classes)
model = TFSwinForImageClassification.from_pretrained(
    'microsoft/swin-tiny-patch4-window7-224',
    num_labels=2
)

# Compile the model with a small learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Fine-tune the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,      # Adjust the number of epochs based on your needs
    batch_size=16
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

# Make predictions on the test set
y_pred_logits = model.predict(test_images)
# Convert logits to probabilities using softmax and select probability for the "tumor" class (index 1)
y_pred_prob = tf.nn.softmax(y_pred_logits, axis=1).numpy()[:, 1]
# Get predicted labels using argmax
y_pred = np.argmax(y_pred_logits, axis=1)

# Compute performance metrics
auc = roc_auc_score(test_labels, y_pred_prob)
acc = accuracy_score(test_labels, y_pred)
prec = precision_score(test_labels, y_pred)
recall = recall_score(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred)
tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()
specificity = tn / (tn + fp)
sensitivity = recall
gmean = math.sqrt(sensitivity * specificity)

# Create a DataFrame with the computed metrics and display it
metrics_df = pd.DataFrame({
    "Metric": ["AUC", "Accuracy", "Precision", "Recall", "F1 Score", "Specificity", "GMean"],
    "Value": [auc, acc, prec, recall, f1, specificity, gmean]
})
print("\nPerformance Metrics:")
print(metrics_df)

# Save test predictions to CSV
predictions_df = pd.DataFrame({
    "y_test": test_labels,
    "y_pred": y_pred,
    "y_pred_prob": y_pred_prob
})
predictions_df.to_csv("test_predictions_swin_448.csv", index=False)

# Save the fine-tuned model
model.save_pretrained("brain_tumor_swin_model_448")

# Plot a graph comparing true and predicted labels
plt.figure(figsize=(10, 6))
plt.plot(range(len(test_labels)), test_labels, 'bo', label="True Labels")
plt.plot(range(len(test_labels)), y_pred, 'rx', label="Predicted Labels")
plt.xlabel("Sample Index")
plt.ylabel("Label")
plt.legend()
plt.title("True vs. Predicted Labels")
plt.show()

# Plot a random test image with its true and predicted labels
random_idx = random.randint(0, len(test_images) - 1)
random_img = test_images[random_idx]
true_label = test_labels[random_idx]

# Predict using the correct model variable (Swin with softmax, index 1 is "Tumor")
predicted_prob = model.predict(random_img[np.newaxis, ...])[0][1]
predicted_label = 1 if predicted_prob > 0.5 else 0

plt.figure(figsize=(6,6))
plt.imshow(random_img)
plt.title("True: {} | Predicted: {}".format("Tumor" if true_label==1 else "No Tumor",
                                               "Tumor" if predicted_label==1 else "No Tumor"))
plt.axis('off')
plt.show()
