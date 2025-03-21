import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from skimage.segmentation import slic
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, filters, measure
import random
import warnings

warnings.filterwarnings('ignore')


# ---------------------------
# Updated SimpleGCN class for dense adjacency matrices
class SimpleGCN(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(SimpleGCN, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # Input shapes: [node_features, adjacency]
        node_feature_shape = input_shape[0]
        self.kernel = self.add_weight(
            shape=(node_feature_shape[-1], self.units),
            initializer='glorot_uniform',
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bias'
        )
        self.built = True

    def call(self, inputs):
        # Unpack inputs: node features and dense adjacency matrix
        x, a = inputs
        a_dense = a  # already dense

        # Add self-loops: a_dense shape (batch, nodes, nodes)
        a_with_self = a_dense + tf.eye(tf.shape(a_dense)[1])

        # Degree matrix and normalization
        d = tf.reduce_sum(a_with_self, axis=2)  # shape (batch, nodes)
        d_inv_sqrt = tf.pow(d, -0.5)
        d_inv_sqrt = tf.where(tf.math.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt)

        batch_size = tf.shape(a_dense)[0]
        max_nodes = tf.shape(a_dense)[1]
        # Create batched diagonal matrices using scatter_nd
        batch_indices = tf.repeat(tf.range(batch_size)[:, tf.newaxis], max_nodes, axis=1)
        diag_indices = tf.reshape(tf.tile(tf.range(max_nodes)[tf.newaxis, :], [batch_size, 1]), [-1])
        indices = tf.stack([
            tf.reshape(batch_indices, [-1]),
            diag_indices,
            diag_indices
        ], axis=1)
        values = tf.reshape(d_inv_sqrt, [-1])
        dense_shape = [batch_size, max_nodes, max_nodes]
        d_inv_sqrt_matrix = tf.scatter_nd(indices, values, dense_shape)

        a_norm = tf.matmul(tf.matmul(d_inv_sqrt_matrix, a_with_self), d_inv_sqrt_matrix)

        # Graph convolution: A' * X * W
        x = tf.matmul(a_norm, x)
        output = tf.matmul(x, self.kernel) + self.bias

        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        node_features_shape = input_shape[0]
        return (node_features_shape[0], node_features_shape[1], self.units)


# ---------------------------
# Improved image_to_graph function with error handling
def image_to_graph(image_path, n_segments=100, compactness=10):
    try:
        img = io.imread(image_path)

        # Validate image dimensions
        if len(img.shape) < 2:
            print(f"Error: {image_path} has invalid dimensions {img.shape}")
            return None, None, None

        # Handle different image formats
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif len(img.shape) == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            elif img.shape[2] == 1:
                img = np.concatenate([img, img, img], axis=2)
        else:
            print(f"Error: {image_path} has unexpected shape {img.shape}")
            return None, None, None

        # Check for valid image
        if img.size == 0 or np.max(img) == np.min(img):
            print(f"Error: {image_path} is empty or has no contrast")
            return None, None, None

        # Resize image
        img = tf.image.resize(img, (512, 512)).numpy().astype('uint8')
        gray_img = color.rgb2gray(img)
        mask = gray_img > 0.05  # exclude dark background

        # Superpixel segmentation
        segments = slic(img, n_segments=n_segments, compactness=compactness,
                        start_label=0, mask=mask)
        n_nodes = segments.max() + 1
        if n_nodes < 2:
            x_grid, y_grid = np.meshgrid(np.linspace(0, img.shape[1] - 1, 10, dtype=int),
                                         np.linspace(0, img.shape[0] - 1, 10, dtype=int))
            segments = np.zeros_like(gray_img, dtype=int)
            segment_idx = 0
            for i in range(9):
                for j in range(9):
                    x_min, x_max = x_grid[i, j], x_grid[i, j + 1]
                    y_min, y_max = y_grid[i, j], y_grid[i + 1, j]
                    segments[y_min:y_max, x_min:x_max] = segment_idx
                    segment_idx += 1
            n_nodes = segment_idx

        # Extract node features (12 features per node)
        node_features = np.zeros((n_nodes, 12))
        edges = filters.sobel(gray_img)
        region_props = measure.regionprops(segments + 1, intensity_image=gray_img)
        region_dict = {prop.label - 1: prop for prop in region_props}

        for i in range(n_nodes):
            mask_seg = segments == i
            if mask_seg.sum() > 0:
                region = gray_img[mask_seg]
                node_features[i, 0] = np.mean(region)
                node_features[i, 1] = np.std(region) if len(region) > 1 else 0
                node_features[i, 2] = np.max(region)
                node_features[i, 3] = np.mean(edges[mask_seg])
                if region.size > 25:
                    flat_region = (region * 255).astype('uint8')
                    try:
                        sample_size = min(8, int(np.sqrt(region.size)))
                        reshaped_region = flat_region.flatten()[:sample_size * sample_size].reshape(sample_size,
                                                                                                    sample_size)
                        if reshaped_region.size >= 4:
                            glcm = graycomatrix(reshaped_region, [1], [0],
                                                levels=min(256, reshaped_region.max() + 1),
                                                symmetric=True, normed=True)
                            node_features[i, 4] = graycoprops(glcm, 'contrast').mean()
                            node_features[i, 5] = graycoprops(glcm, 'homogeneity').mean()
                            node_features[i, 6] = graycoprops(glcm, 'energy').mean()
                            node_features[i, 7] = graycoprops(glcm, 'correlation').mean()
                            node_features[i, 8] = graycoprops(glcm, 'dissimilarity').mean()
                    except Exception:
                        node_features[i, 4:9] = 0
                if i in region_dict:
                    prop = region_dict[i]
                    node_features[i, 9] = prop.area / mask_seg.size
                    node_features[i, 10] = prop.perimeter / (np.sqrt(mask_seg.size) + 1e-8)
                    node_features[i, 11] = prop.eccentricity

        # Normalize features
        for i in range(node_features.shape[1]):
            col_max = np.max(node_features[:, i])
            col_min = np.min(node_features[:, i])
            if col_max > col_min:
                node_features[:, i] = (node_features[:, i] - col_min) / (col_max - col_min + 1e-8)

        # Create adjacency matrix based on superpixel neighbors
        adjacency = np.zeros((n_nodes, n_nodes))
        from scipy import ndimage
        for i in range(n_nodes):
            mask_seg = segments == i
            if mask_seg.sum() > 0:
                dilated = ndimage.binary_dilation(mask_seg, structure=np.ones((3, 3)))
                overlapping = np.unique(segments[dilated & ~mask_seg])
                for j in overlapping:
                    if j != i and 0 <= j < n_nodes:
                        adjacency[i, j] = adjacency[j, i] = 1

        return node_features, adjacency, segments
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None


# ---------------------------
# Build Graph CNN Model (using dense adjacency matrices)
def build_gcnn_model(n_features):
    node_features_input = keras.Input(shape=(None, n_features), name='node_features')
    adjacency_input = keras.Input(shape=(None, None), name='adjacency', sparse=False)

    x = SimpleGCN(64, activation='relu')([node_features_input, adjacency_input])
    x = layers.BatchNormalization()(x)
    x = SimpleGCN(64, activation='relu')([x, adjacency_input])
    x = layers.BatchNormalization()(x)
    x = SimpleGCN(32, activation='relu')([x, adjacency_input])
    x = layers.BatchNormalization()(x)

    x = tf.reduce_mean(x, axis=1)  # Global average pooling
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[node_features_input, adjacency_input], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    )
    return model


# ---------------------------
# Create Graph Dataset from directory
def create_graph_dataset(directory):
    graph_features = []
    graph_adjacency = []
    labels = []
    file_paths = []

    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            label = 1 if class_name.lower() == "tumor" else 0
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Processing {len(image_files)} images from {class_name}...")
            successful = 0
            failed = 0
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                try:
                    node_features, adjacency, segments = image_to_graph(img_path)
                    if node_features is not None and adjacency is not None and segments is not None:
                        graph_features.append(node_features)
                        graph_adjacency.append(adjacency)
                        labels.append(label)
                        file_paths.append(img_path)
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    failed += 1
            print(f"Class {class_name}: Successfully processed {successful} images, Failed on {failed} images")
    print(f"Total dataset: {len(graph_features)} graphs")
    return graph_features, graph_adjacency, np.array(labels), file_paths


# ---------------------------
# Visualization: random test image and its graph representation
def visualize_random_prediction_with_graph(test_file_paths, test_labels, y_pred, y_pred_probs):
    idx = np.random.randint(0, len(test_labels))
    img_path = test_file_paths[idx]
    true_label = test_labels[idx]
    pred_label = y_pred[idx]
    pred_prob = y_pred_probs[idx]

    try:
        img = io.imread(img_path)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title(f"Original Image\nTrue: {'Tumor' if true_label == 1 else 'No Tumor'}\n"
                  f"Pred: {'Tumor' if pred_label == 1 else 'No Tumor'} (Prob: {pred_prob:.2f})")
        plt.axis('off')

        node_features, adjacency, segments = image_to_graph(img_path)
        if segments is None:
            plt.subplot(1, 3, 2)
            plt.imshow(img)
            plt.title("Segmentation failed")
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(img)
            plt.title("Graph representation unavailable")
            plt.axis('off')
        else:
            from skimage.segmentation import mark_boundaries
            plt.subplot(1, 3, 2)
            img_resized = tf.image.resize(img, (512, 512)).numpy().astype('uint8')
            plt.imshow(mark_boundaries(img_resized, segments))
            plt.title(f'Superpixel Segmentation\n({segments.max() + 1} segments)')
            plt.axis('off')

            n_nodes = segments.max() + 1
            centroids = np.zeros((n_nodes, 2))
            for i in range(n_nodes):
                mask_seg = segments == i
                if mask_seg.sum() > 0:
                    coords = np.column_stack(np.where(mask_seg))
                    centroids[i] = coords.mean(axis=0)
            plt.subplot(1, 3, 3)
            plt.imshow(img_resized, alpha=0.7)
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if adjacency[i, j] > 0:
                        plt.plot([centroids[i, 1], centroids[j, 1]],
                                 [centroids[i, 0], centroids[j, 0]], 'y-', alpha=0.3, linewidth=0.5)
            for i in range(n_nodes):
                plt.plot(centroids[i, 1], centroids[i, 0], 'ro', markersize=3, alpha=0.5)
            plt.title(f'Graph Representation\n({int(np.sum(adjacency) / 2)} edges)')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error visualizing {img_path}: {e}")


# ---------------------------
# Main execution block
print("Starting Brain Tumor Detection using Graph CNN...")

# Define paths
model_save_path = "brain_tumor_gcnn_model.keras"
train_dir = r"D:\ED\braintumor\data\Training"
test_dir = r"D:\ED\braintumor\data\Testing"

# Load saved model if available; else train and save it.
if os.path.exists(model_save_path):
    print(f"Loading saved model from {model_save_path}")
    model = load_model(model_save_path, custom_objects={'SimpleGCN': SimpleGCN})
    retrain = False
else:
    retrain = True

if retrain:
    print("Creating graph dataset from training images...")
    train_features, train_adjacency, train_labels, _ = create_graph_dataset(train_dir)
    n_features = train_features[0].shape[1]
    print("Building and training Graph CNN model...")
    model = build_gcnn_model(n_features)
    history = model.fit(
        [np.array(train_features), np.array(train_adjacency)],
        train_labels,
        epochs=10,
        batch_size=8,
        verbose=1
    )
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
else:
    print("Using loaded model; skipping training.")

print("Processing test dataset...")
test_features, test_adjacency, test_labels, test_file_paths = create_graph_dataset(test_dir)

print("Making predictions on test dataset...")
y_pred_probs = []
for i in range(len(test_features)):
    x = test_features[i]
    a = test_adjacency[i]
    x_batch = np.expand_dims(x, axis=0)
    a_batch = np.expand_dims(a, axis=0)
    pred = model.predict([x_batch, a_batch])[0][0]
    y_pred_probs.append(pred)
y_pred_probs = np.array(y_pred_probs)
y_pred = (y_pred_probs > 0.5).astype(int)
test_labels = np.array(test_labels)

# Save y_test and y_predict in a CSV file with columns "y_test" and "y_predict"
results_df = pd.DataFrame({
    "y_test": test_labels,
    "y_predict": y_pred
})
results_df.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")

# Compute evaluation metrics
accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred)
recall = recall_score(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred)
fpr, tpr, _ = roc_curve(test_labels, y_pred_probs)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(test_labels, y_pred)
if cm.size == 4:
    TN, FP, FN, TP = cm.ravel()
else:
    TN, FP, FN, TP = (0, 0, 0, 0)
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
gmean = np.sqrt(recall * specificity)

print("\nEvaluation Metrics:")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1 Score:    {f1:.4f}")
print(f"AUC:         {roc_auc:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"G-Mean:      {gmean:.4f}")

# ---------------------------
# Display interactive plots

# Confusion Matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
classes = ['No Tumor', 'Tumor']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.0
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Scatter Plot: True vs Predicted Labels
plt.figure(figsize=(8, 5))
plt.scatter(range(len(test_labels)), test_labels, label="True Labels", color='blue', marker='o', alpha=0.7)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted Labels", color='red', marker='x', alpha=0.7)
plt.xlabel("Sample Index")
plt.ylabel("Label (0: No Tumor, 1: Tumor)")
plt.title("True vs Predicted Labels")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Visualize a random test image with its prediction and graph representation
visualize_random_prediction_with_graph(test_file_paths, test_labels, y_pred, y_pred_probs)

print("Analysis complete!")
