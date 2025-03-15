import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_score, recall_score, f1_score, accuracy_score)


# -------------------------------
# Define a Transformer Block
# -------------------------------
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# -------------------------------
# Build the GraphCNNTransformer Model
# -------------------------------
def create_model(input_shape=(512, 512, 3)):
    inputs = keras.Input(shape=input_shape)

    # CNN part: several convolution and pooling layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    # Reshape to create a sequence (flatten spatial dimensions)
    seq_len = x.shape[1] * x.shape[2]  # number of patches/tokens
    embed_dim = x.shape[3]
    x = layers.Reshape((seq_len, embed_dim))(x)

    # Transformer block
    transformer_block = TransformerBlock(embed_dim=embed_dim, num_heads=4, ff_dim=128)
    x = transformer_block(x)

    # Global pooling and dense classifier
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary classification

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
# Update these paths with your actual directory
train_dir = "input_root/training"
test_dir = "input_root/testing"

batch_size = 32
img_size = (512, 512)

# For training, images are already 512x512
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

# For testing, even though the original images vary in size,
# we resize them to 512x512 to feed into the model.
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

# -------------------------------
# Model Compilation and Training
# -------------------------------
model = create_model(input_shape=(img_size[0], img_size[1], 3))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', keras.metrics.AUC(name='auc')])
model.summary()

# Train for 10 epochs
history = model.fit(train_ds, validation_data=test_ds, epochs=10)

# -------------------------------
# Evaluation and Metric Calculation
# -------------------------------
# Get predictions (predicted probabilities)
y_pred_probs = model.predict(test_ds)
# Convert probabilities to binary predictions (threshold = 0.5)
y_pred = (y_pred_probs > 0.5).astype(int)

# Retrieve true labels from test dataset
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# Calculate metrics using sklearn
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)
# Confusion Matrix to compute Specificity
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
# GMean: geometric mean of sensitivity (recall) and specificity
gmean = np.sqrt(recall * specificity)

# Print all the metrics
print(f"Accuracy:   {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1 Score:   {f1:.4f}")
print(f"AUC:        {roc_auc:.4f}")
print(f"Specificity:{specificity:.4f}")
print(f"GMean:      {gmean:.4f}")

# -------------------------------
# Save Predictions to CSV
# -------------------------------
# Save y_true, predicted probabilities, and binary predictions
df = pd.DataFrame({
    'y_true': y_true.flatten(),
    'y_pred_prob': y_pred_probs.flatten(),
    'y_pred': y_pred.flatten()
})
csv_filename = "predictions.csv"
df.to_csv(csv_filename, index=False)
print(f"Saved predictions to {csv_filename}")

# -------------------------------
# Plot Confusion Matrix
# -------------------------------
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Tumor", "Tumor"],
            yticklabels=["No Tumor", "Tumor"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# Plot ROC Curve
# -------------------------------
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.show()

# -------------------------------
# Display a Random Test Image with True and Predicted Labels
# -------------------------------
# Take one batch from test dataset and pick a random image from that batch.
for images, labels in test_ds.take(1):
    idx = random.randint(0, images.shape[0] - 1)
    img = images[idx]
    true_label = labels[idx].numpy()
    # Get prediction for the single image
    pred_prob = model.predict(tf.expand_dims(img, 0))[0][0]
    pred_label = 1 if pred_prob > 0.5 else 0

    plt.figure()
    plt.imshow(img.numpy().astype("uint8"))
    plt.title(f"True: {int(true_label)}, Predicted: {pred_label} (Prob: {pred_prob:.2f})")
    plt.axis("off")
    plt.show()
