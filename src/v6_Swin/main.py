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

# ---------------------------
# Configuration and Settings
# ---------------------------

# Paths to your data directories
train_path = r"D:\ED\braintumor\data\Training"
test_path = r"D:\ED\braintumor\data\Testing"

# Set image resolution – choose either 224x224 or 128x128
img_size = (224, 224)  # Change to (128, 128) if desired

# Batch size for tf.data pipeline
batch_size = 16

# Initialize the image processor.
# We force the slow processor (use_fast=False) because it returns NumPy arrays, which works smoothly with TensorFlow.
image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224", use_fast=False)

# ---------------------------
# Data Preparation Functions
# ---------------------------

def generate_file_paths_and_labels(root_path):
    """
    Returns lists of file paths and corresponding labels.
    Assumes subfolders named 'notumor' (label 0) and 'tumor' (label 1).
    """
    file_paths = []
    labels = []
    for label, subfolder in enumerate(["notumor", "tumor"]):
        folder = os.path.join(root_path, subfolder)
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(folder, fname))
                labels.append(label)
    return file_paths, labels

def load_and_preprocess_image(file_path, label):
    """
    Loads an image from disk, resizes it, converts it to an array,
    and processes it using the image processor.
    """
    # file_path comes as a bytes string from tf.py_function; convert it to a Python string.
    file_path = file_path.numpy().decode('utf-8')
    # Load image using Keras
    img = load_img(file_path, target_size=img_size)
    img = img_to_array(img)
    # Process the image; returns a NumPy array with shape (H, W, 3)
    processed = image_processor(images=img, size=img_size, return_tensors="np")['pixel_values'][0]
    return processed, label

def tf_load_and_preprocess(file_path, label):
    """
    Wraps the Python image loading function for use in tf.data.
    """
    processed_img, lbl = tf.py_function(func=load_and_preprocess_image, inp=[file_path, label],
                                          Tout=(tf.float32, tf.int32))
    # Set static shape information (channels=3)
    processed_img.set_shape((img_size[0], img_size[1], 3))
    lbl.set_shape(())
    return processed_img, lbl

def create_dataset(file_paths, labels, batch_size=16, shuffle=True):
    """
    Creates a tf.data.Dataset from lists of file paths and labels.
    """
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths))
    ds = ds.map(tf_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------------
# Prepare File Lists and Datasets
# ---------------------------

# Get file paths and labels for training and testing data
train_file_paths, train_labels = generate_file_paths_and_labels(train_path)
test_file_paths, test_labels = generate_file_paths_and_labels(test_path)

# Split training data into training and validation sets (80/20 split)
train_file_paths, val_file_paths, train_labels, val_labels = train_test_split(
    train_file_paths, train_labels, test_size=0.2, random_state=42)

# Create tf.data.Dataset objects for training, validation, and test sets
train_ds = create_dataset(train_file_paths, train_labels, batch_size=batch_size, shuffle=True)
val_ds   = create_dataset(val_file_paths, val_labels, batch_size=batch_size, shuffle=False)
test_ds  = create_dataset(test_file_paths, test_labels, batch_size=batch_size, shuffle=False)

# ---------------------------
# Model Setup and Training
# ---------------------------

# Load the pre-trained Swin Transformer model for image classification.
# The 'from_pt=True' parameter tells the function to load from a PyTorch checkpoint.
model = TFSwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    num_labels=2,
    from_pt=True
)

# Compile the model with an Adam optimizer and a sparse categorical crossentropy loss.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train (fine-tune) the model
epochs = 5  # Adjust as needed
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# ---------------------------
# Evaluation and Metrics
# ---------------------------

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

# Collect predictions on the test dataset for additional metrics
all_labels = []
all_preds = []
all_pred_probs = []
for batch_images, batch_labels in test_ds:
    logits = model.predict(batch_images)
    probs = tf.nn.softmax(logits, axis=1).numpy()
    preds = np.argmax(probs, axis=1)
    all_labels.extend(batch_labels.numpy())
    all_preds.extend(preds)
    all_pred_probs.extend(probs[:, 1])  # Probability for class "tumor" (label 1)

# Compute performance metrics using scikit-learn
auc = roc_auc_score(all_labels, all_pred_probs)
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
specificity = tn / (tn + fp)
sensitivity = recall
gmean = math.sqrt(sensitivity * specificity)

metrics_df = pd.DataFrame({
    "Metric": ["AUC", "Accuracy", "Precision", "Recall", "F1 Score", "Specificity", "GMean"],
    "Value": [auc, acc, prec, recall, f1, specificity, gmean]
})
print("\nPerformance Metrics:")
print(metrics_df)

# ---------------------------
# Visualization
# ---------------------------

# (Optional) Plot a comparison of true vs. predicted labels for the test set
plt.figure(figsize=(10, 6))
plt.plot(all_labels, 'bo', label="True Labels")
plt.plot(all_preds, 'rx', label="Predicted Labels")
plt.xlabel("Sample Index")
plt.ylabel("Label")
plt.legend()
plt.title("True vs. Predicted Labels (Test Set)")
plt.show()

# Plot a random test image with its true and predicted labels
random_index = random.randint(0, len(test_file_paths) - 1)
# Load and preprocess a single test image using our helper function
img, true_label = load_and_preprocess_image(test_file_paths[random_index], test_labels[random_index])
# Expand dimensions to create a batch for prediction
img_batch = np.expand_dims(img, axis=0)
logits = model.predict(img_batch)
probs = tf.nn.softmax(logits, axis=1).numpy()[0]
predicted_prob = probs[1]  # probability for "tumor"
predicted_label = 1 if predicted_prob > 0.5 else 0

plt.figure(figsize=(6, 6))
# Depending on the processor, the image may be in float range [0,1] – adjust if necessary.
plt.imshow((img * 255).astype(np.uint8))  # scale up if image is normalized
plt.title("True: {} | Predicted: {}".format("Tumor" if true_label == 1 else "No Tumor",
                                              "Tumor" if predicted_label == 1 else "No Tumor"))
plt.axis('off')
plt.show()
