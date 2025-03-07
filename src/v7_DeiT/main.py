import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from transformers import AutoFeatureExtractor, TFAutoModelForImageClassification

# Define paths
train_path = r"D:\ED\braintumor\data\Training"
test_path = r"D:\ED\braintumor\data\Testing"

# Set image size for DeiT (224x224)
img_size = (224, 224)

# Function to load images from folder (expects subfolders: 'notumor' and 'tumor')
def load_images_from_folder(folder):
    images = []
    labels = []
    for label, subfolder in enumerate(['notumor', 'tumor']):
        subfolder_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_path):
            if filename.lower().endswith('.jpg'):
                img_path = os.path.join(subfolder_path, filename)
                img = load_img(img_path, target_size=img_size)
                img = img_to_array(img)  # pixel values in range [0, 255]
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load training and testing images
train_images, train_labels = load_images_from_folder(train_path)
test_images, test_labels = load_images_from_folder(test_path)

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Initialize the feature extractor and DeiT model (ignore mismatched sizes for the classification head)
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/deit-base-distilled-patch16-224")
model = TFAutoModelForImageClassification.from_pretrained(
    "facebook/deit-base-distilled-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)

# Preprocess images using the feature extractor
def preprocess_images(images):
    inputs = feature_extractor(images=list(images.astype("uint8")), return_tensors="tf")
    return inputs["pixel_values"]

inputs_train = preprocess_images(X_train)
inputs_val = preprocess_images(X_val)
inputs_test = preprocess_images(test_images)

# Create tf.data.Dataset objects
batch_size = 16
train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, y_train)).shuffle(1000).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((inputs_val, y_val)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((inputs_test, test_labels)).batch(batch_size)

# Compile the model using TensorFlow's Keras optimizers and losses
optimizer = Adam(learning_rate=5e-5)
loss = SparseCategoricalCrossentropy(from_logits=True)
metrics = ["accuracy"]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Fine-tune the model
epochs = 3
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# Evaluate the model on the test dataset
results = model.evaluate(test_dataset)
print("Test Loss: {:.4f} | Test Accuracy: {:.2f}%".format(results[0], results[1] * 100))

# Predict on the test set
predictions = model.predict(test_dataset)
logits = predictions.logits
pred_probs = tf.nn.softmax(logits, axis=-1).numpy()
pred_labels = np.argmax(pred_probs, axis=-1)

# Compute performance metrics
auc = roc_auc_score(test_labels, pred_probs[:, 1])
acc = accuracy_score(test_labels, pred_labels)
prec = precision_score(test_labels, pred_labels)
recall = recall_score(test_labels, pred_labels)
f1 = f1_score(test_labels, pred_labels)
tn, fp, fn, tp = confusion_matrix(test_labels, pred_labels).ravel()
specificity = tn / (tn + fp)
sensitivity = recall
gmean = math.sqrt(sensitivity * specificity)

metrics_df = pd.DataFrame({
    "Metric": ["AUC", "Accuracy", "Precision", "Recall", "F1 Score", "Specificity", "GMean"],
    "Value": [auc, acc, prec, recall, f1, specificity, gmean]
})
print("\nPerformance Metrics:")
print(metrics_df)

# Save test set predictions to a CSV file
predictions_df = pd.DataFrame({
    "y_test": test_labels.flatten(),
    "y_pred": pred_labels.flatten()
})
predictions_df.to_csv("deit_predictions.csv", index=False)

# Save the fine-tuned model
model.save_pretrained("brain_tumor_deit_model")

# Plot true vs. predicted labels for the test set
plt.figure(figsize=(10, 6))
plt.plot(range(len(test_labels)), test_labels, 'bo', label="True Labels")
plt.plot(range(len(test_labels)), pred_labels, 'rx', label="Predicted Labels")
plt.xlabel("Sample Index")
plt.ylabel("Label")
plt.legend()
plt.title("True vs. Predicted Labels")
plt.show()

# Plot a random test image with its true and predicted labels
random_idx = random.randint(0, len(test_images) - 1)
random_img = test_images[random_idx]
true_label = test_labels[random_idx]

inputs_random = feature_extractor(images=[random_img.astype("uint8")], return_tensors="tf")["pixel_values"]
pred_random = model.predict(inputs_random)
pred_prob = tf.nn.softmax(pred_random.logits, axis=-1)[0]
pred_label = np.argmax(pred_prob)

plt.figure(figsize=(6,6))
plt.imshow(random_img.astype("uint8"))
plt.title("True: {} | Predicted: {}".format("Tumor" if true_label == 1 else "No Tumor",
                                              "Tumor" if pred_label == 1 else "No Tumor"))
plt.axis('off')
plt.show()
