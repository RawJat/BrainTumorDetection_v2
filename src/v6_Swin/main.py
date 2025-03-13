import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_score, recall_score, f1_score, accuracy_score)

# ============================
# 1. Data Loading (No extra preprocessing)
# ============================

# Set your input root directory (using a raw string for Windows paths)
input_root = r"D:\ED\braintumor\data"
train_dir = os.path.join(input_root, 'training')
test_dir = os.path.join(input_root, 'testing')

# image_dataset_from_directory will automatically resize images to 512x512.
batch_size = 32
img_size = (512, 512)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=False
)

# Normalize images to [0,1]
normalization_layer = layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))


# ============================
# 2. SWIN Transformer Implementation (Simplified)
# ============================

class SwinTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1, **kwargs):
        super(SwinTransformerBlock, self).__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = models.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate)
        ])

    def call(self, x, training=False):
        # Self-attention block with skip connection.
        shortcut = x
        x = self.norm1(x)
        attn_out = self.attn(x, x)
        x = shortcut + self.dropout1(attn_out, training=training)

        # MLP block with skip connection.
        shortcut = x
        x = self.norm2(x)
        mlp_out = self.mlp(x, training=training)
        return shortcut + mlp_out


def build_swin_transformer(input_shape=(512, 512, 3),
                           num_classes=1,
                           patch_size=16,
                           embed_dim=96,
                           num_heads=4,
                           mlp_dim=192,
                           num_layers=2):
    inputs = layers.Input(shape=input_shape)
    # Patch embedding: splitting image into patches using a Conv2D layer.
    x = layers.Conv2D(filters=embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    patch_grid_size = (x.shape[1], x.shape[2])
    num_patches = patch_grid_size[0] * patch_grid_size[1]
    x = layers.Reshape((num_patches, embed_dim))(x)

    # Transformer blocks (simplified version without shifted windows)
    for _ in range(num_layers):
        x = SwinTransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# Create and compile the model.
model = build_swin_transformer()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', keras.metrics.AUC(name='auc')])
model.summary()

# ============================
# 3. Training with Model Checkpoint
# ============================

epochs = 10  # You can increase the number of epochs later.
# Callback to save the best model based on validation loss.
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(train_ds, epochs=epochs, validation_data=test_ds, callbacks=[checkpoint])

# Save the final model so you don't have to retrain next time.
model.save('final_model.keras')
print("Final model saved as final_model.keras")

# ============================
# 4. Evaluation and Metrics
# ============================

# Get predictions on the test set.
y_true = []
y_pred_probs = []
for batch_imgs, batch_labels in test_ds:
    preds = model.predict(batch_imgs)
    y_true.extend(batch_labels.numpy())
    y_pred_probs.extend(preds.flatten())

# Convert probabilities to binary labels (threshold 0.5)
y_pred_labels = [1 if prob >= 0.5 else 0 for prob in y_pred_probs]

# Compute metrics
cm = confusion_matrix(y_true, y_pred_labels)
accuracy = accuracy_score(y_true, y_pred_labels)
precision = precision_score(y_true, y_pred_labels)
recall = recall_score(y_true, y_pred_labels)
f1 = f1_score(y_true, y_pred_labels)
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

# Specificity = TN / (TN + FP)
TN, FP, FN, TP = cm.ravel()
specificity = TN / (TN + FP)
gmean = np.sqrt(recall * specificity)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"GMean: {gmean:.4f}")

# ============================
# 5. Save Predictions to CSV
# ============================

results_df = pd.DataFrame({
    'True_Label': y_true,
    'Predicted_Probability': y_pred_probs,
    'Predicted_Label': y_pred_labels
})
results_df.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")

# ============================
# 6. Visualization
# ============================

# (a) Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()
print("Confusion matrix saved as confusion_matrix.png")

# (b) ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()
print("ROC curve saved as roc_curve.png")

# (c) Display a Random Test Image with True and Predicted Label
test_labels = ['tumor', 'notumor']
selected_label = random.choice(test_labels)
selected_folder = os.path.join(test_dir, selected_label)
img_files = [f for f in os.listdir(selected_folder) if f.lower().endswith('.jpg')]
if img_files:
    selected_img_file = random.choice(img_files)
    img_path = os.path.join(selected_folder, selected_img_file)

    from PIL import Image

    img = Image.open(img_path).convert('RGB').resize(img_size)
    img_array = np.array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    pred_prob = model.predict(img_input)[0][0]
    pred_class = 'tumor' if pred_prob >= 0.5 else 'notumor'

    plt.figure()
    plt.imshow(img)
    plt.title(f"True: {selected_label} | Predicted: {pred_class} ({pred_prob:.2f})")
    plt.axis('off')
    plt.savefig("predicted_image.png")
    plt.close()
    print("Predicted image saved as predicted_image.png")
else:
    print("No .jpg images found in the selected test folder.")
