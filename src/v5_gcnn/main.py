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
import seaborn as sns


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

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# -------------------------------
# Build the GraphCNNTransformer Model with Extra Pooling
# -------------------------------
def create_model(input_shape=(512, 512, 3)):
    inputs = keras.Input(shape=input_shape)

    # CNN layers: extract spatial features
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)  # Output: (256,256,32)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)  # Output: (128,128,64)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)  # Output: (64,64,128)

    # Additional pooling to reduce sequence length and memory usage
    x = layers.MaxPooling2D(pool_size=(4, 4))(x)  # Now output: (16,16,128)

    # Reshape to create a sequence of tokens for the Transformer block
    seq_len = x.shape[1] * x.shape[2]  # 16*16 = 256 tokens
    embed_dim = x.shape[3]
    x = layers.Reshape((seq_len, embed_dim))(x)

    # Transformer block
    transformer_block = TransformerBlock(embed_dim=embed_dim, num_heads=4, ff_dim=128)
    x = transformer_block(x)

    # Global pooling and dense classifier for binary classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
# Update these paths with your actual input root directory
train_dir = r"D:\ED\braintumor\data\Training"
test_dir = r"D:\ED\braintumor\data\Testing"

batch_size = 32
img_size = (512, 512)

# Load training dataset (images are already 512x512)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

# Load testing dataset (images resized to 512x512)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

# -------------------------------
# Model Loading / Training & Saving
# -------------------------------
model_path = "brain_tumor_model.keras"

if os.path.exists(model_path):
    print(f"Loading saved model from {model_path}")
    model = keras.models.load_model(model_path, custom_objects={'TransformerBlock': TransformerBlock})
else:
    model = create_model(input_shape=(img_size[0], img_size[1], 3))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    model.summary()
    history = model.fit(train_ds, validation_data=test_ds, epochs=10)
    model.save(model_path)
    print(f"Model saved to {model_path}")

# -------------------------------
# Evaluation and Metric Calculation
# -------------------------------
# Get predictions (as probabilities)
y_pred_probs = model.predict(test_ds)
# Convert probabilities to binary predictions using threshold=0.5
y_pred = (y_pred_probs > 0.5).astype(int)

# Get true labels from the test dataset
y_true = np.concatenate([labels for _, labels in test_ds], axis=0)

# Calculate metrics
accuracy_val = accuracy_score(y_true, y_pred)
precision_val = precision_score(y_true, y_pred)
recall_val = recall_score(y_true, y_pred)
f1_val = f1_score(y_true, y_pred)
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc_val = auc(fpr, tpr)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity_val = tn / (tn + fp)
gmean_val = np.sqrt(recall_val * specificity_val)

print(f"Accuracy:    {accuracy_val:.4f}")
print(f"Precision:   {precision_val:.4f}")
print(f"Recall:      {recall_val:.4f}")
print(f"F1 Score:    {f1_val:.4f}")
print(f"AUC:         {roc_auc_val:.4f}")
print(f"Specificity: {specificity_val:.4f}")
print(f"GMean:       {gmean_val:.4f}")

# -------------------------------
# Save Predictions to CSV
# -------------------------------
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
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc_val:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.show()

# -------------------------------
# Plot Scatter: True vs Predicted Labels
# -------------------------------
plt.figure(figsize=(8, 4))
plt.scatter(range(len(y_true)), y_true, label="True Labels", color='blue', marker='o', alpha=0.7)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted Labels", color='red', marker='x', alpha=0.7)
plt.xlabel("Sample Index")
plt.ylabel("Label")
plt.title("True vs Predicted Labels Scatter Plot")
plt.legend()
plt.show()

# -------------------------------
# Display a Random Test Image with True and Predicted Labels
# -------------------------------
for images, labels in test_ds.take(1):
    idx = random.randint(0, images.shape[0] - 1)
    img = images[idx]
    true_label = labels[idx].numpy()
    pred_prob = model.predict(tf.expand_dims(img, 0))[0][0]
    pred_label = 1 if pred_prob > 0.5 else 0

    plt.figure()
    plt.imshow(img.numpy().astype("uint8"))
    plt.title(f"True: {int(true_label)}, Predicted: {pred_label} (Prob: {pred_prob:.2f})")
    plt.axis("off")
    plt.show()
