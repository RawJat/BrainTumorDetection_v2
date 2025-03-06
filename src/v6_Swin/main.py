import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import math
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from transformers import TFSwinModel, AutoImageProcessor

# ----- Paths & Parameters -----
train_path = r"D:\ED\braintumor\data\Training"
test_path = r"D:\ED\braintumor\data\Testing"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ----- Data Augmentation & Loading -----
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    color_mode='rgb'  # Ensure images are loaded in RGB format
)
val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    color_mode='rgb'  # Ensure images are loaded in RGB format
)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=1,  # one image at a time for predictions
    class_mode='binary',
    shuffle=False,
    color_mode='rgb'  # Ensure images are loaded in RGB format
)

# ----- Load Pretrained Swin Transformer Base Model -----
base_model = TFSwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224", from_pt=True)
image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

# ----- Define a Normalization Function -----
def normalize_fn(img):
    mean = tf.constant([0.485, 0.456, 0.406], shape=(1, 1, 3), dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], shape=(1, 1, 3), dtype=tf.float32)
    return (img - mean) / std

# ----- Build the Custom Classification Model -----
input_layer = keras.layers.Input(shape=(224, 224, 3), name="image_input")
x = keras.layers.Lambda(normalize_fn, output_shape=(224, 224, 3))(input_layer)
swin_outputs = base_model(x)
x = keras.layers.GlobalAveragePooling1D()(swin_outputs.last_hidden_state)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
output_layer = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.models.Model(inputs=input_layer, outputs=output_layer)

# ----- Compile the Model -----
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ----- Callbacks: Early Stopping & LR Scheduler -----
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# ----- Train the Model -----
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stopping, lr_scheduler]
)

# ----- Evaluate on Test Set -----
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Make predictions on the test set.
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int)
y_true = test_generator.classes

# Compute performance metrics.
auc = roc_auc_score(y_true, y_pred_prob)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp)
sensitivity = recall
gmean = math.sqrt(sensitivity * specificity)

metrics_df = pd.DataFrame({
    "Metric": ["AUC", "Accuracy", "Precision", "Recall", "F1 Score", "Specificity", "GMean"],
    "Value": [auc, acc, prec, recall, f1, specificity, gmean]
})
print("\nPerformance Metrics:")
print(metrics_df)

# Save the trained model.
model.save("swin_tumor_model.keras")

# ----- Plot a Random Test Image with True & Predicted Labels -----
random_idx = random.randint(0, len(test_generator.filenames) - 1)
random_img, true_label = test_generator[random_idx]
random_img = random_img[0]
true_label = int(true_label[0])
predicted_prob = model.predict(random_img[np.newaxis, ...])[0][0]
predicted_label = 1 if predicted_prob > 0.5 else 0

plt.figure(figsize=(6, 6))
plt.imshow(random_img)
plt.title("True: {} | Predicted: {}".format(
    "Tumor" if true_label == 1 else "No Tumor",
    "Tumor" if predicted_label == 1 else "No Tumor"
))
plt.axis('off')
plt.show()