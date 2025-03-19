import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import spektral
from spektral.layers import GCNConv, GlobalSumPool
from spektral.data import Dataset, Graph
from skimage.segmentation import slic
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, filters, measure
import random
import warnings

warnings.filterwarnings('ignore')

# Set your paths
train_dir = r"D:\ED\braintumor\data\Training"
test_dir = r"D:\ED\braintumor\data\Testing"
model_save_path = "brain_tumor_gcnn_model.keras"


# Improved function to create graph from image using superpixel segmentation
def image_to_graph(image_path, n_segments=100, compactness=10):
    # Load image
    img = io.imread(image_path)

    # Resize to standard dimensions
    img = tf.image.resize(img, (512, 512)).numpy().astype('uint8')

    # Handle grayscale vs RGB
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):  # Grayscale
        # Convert to 3-channel for consistent processing
        if len(img.shape) == 2:
            img_3channel = np.stack([img, img, img], axis=-1)
            gray_img = img / 255.0  # Normalize to [0,1]
        else:  # Already 3D but single channel
            img_3channel = np.concatenate([img, img, img], axis=2)
            gray_img = img[:, :, 0] / 255.0
    else:  # RGB
        img_3channel = img
        gray_img = color.rgb2gray(img)

    # Create a mask to exclude the background (assuming background is very dark)
    mask = gray_img > 0.05  # Threshold to exclude pure black background

    # Apply superpixel segmentation on the masked area
    segments = slic(img_3channel, n_segments=n_segments, compactness=compactness,
                    start_label=0, mask=mask)

    # Extract features for each superpixel
    n_nodes = segments.max() + 1

    # Node features: expanded for medical imaging (12 features)
    # - 3 intensity features (mean, std, max)
    # - 6 texture features (GLCM)
    # - 3 shape features (area, perimeter, eccentricity)
    node_features = np.zeros((n_nodes, 12))

    # Calculate additional image enhancement for feature extraction
    # Edge detection to highlight boundaries
    edges = filters.sobel(gray_img)

    # Region properties for shape analysis
    region_props = measure.regionprops(segments + 1, intensity_image=gray_img)
    region_dict = {prop.label - 1: prop for prop in region_props}

    for i in range(n_nodes):
        mask = segments == i
        if mask.sum() > 0:
            # Region intensity statistics
            region = gray_img[mask]
            node_features[i, 0] = np.mean(region)  # Mean intensity
            node_features[i, 1] = np.std(region)  # Std deviation of intensity
            node_features[i, 2] = np.max(region)  # Max intensity

            # Edge content (tumor boundaries often have strong edges)
            node_features[i, 3] = np.mean(edges[mask])

            # Texture features (GLCM properties)
            if region.size > 1:
                flat_region = (region * 255).astype('uint8')
                try:
                    # Reshape to 2D for GLCM if possible
                    if region.size >= 64:  # Need reasonable size for texture
                        reshaped_region = flat_region.flatten()[:64].reshape(8, 8)
                        glcm = graycomatrix(reshaped_region, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                            levels=64, symmetric=True, normed=True)
                        node_features[i, 4] = graycoprops(glcm, 'contrast').mean()
                        node_features[i, 5] = graycoprops(glcm, 'homogeneity').mean()
                        node_features[i, 6] = graycoprops(glcm, 'energy').mean()
                        node_features[i, 7] = graycoprops(glcm, 'correlation').mean()
                        node_features[i, 8] = graycoprops(glcm, 'dissimilarity').mean()
                except:
                    # If GLCM fails, use zeros for texture features
                    node_features[i, 4:9] = 0

            # Shape features if available from regionprops
            if i in region_dict:
                prop = region_dict[i]
                node_features[i, 9] = prop.area / mask.size  # Normalized area
                node_features[i, 10] = prop.perimeter / np.sqrt(mask.size)  # Normalized perimeter
                node_features[i, 11] = prop.eccentricity  # Eccentricity

    # Min-max normalize features column-wise
    for i in range(node_features.shape[1]):
        if np.max(node_features[:, i]) > np.min(node_features[:, i]):
            node_features[:, i] = (node_features[:, i] - np.min(node_features[:, i])) / (
                        np.max(node_features[:, i]) - np.min(node_features[:, i]))

    # Create adjacency matrix based on superpixel neighbors
    adjacency = np.zeros((n_nodes, n_nodes))

    # Find neighboring superpixels - more comprehensive approach
    dilated_segments = np.copy(segments)

    # Enumerate all segments
    for i in range(n_nodes):
        # Create a mask for current segment
        mask = segments == i
        if mask.sum() > 0:
            # Dilate the mask slightly to find neighbors
            from scipy import ndimage
            dilated = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
            # Find which other segments this dilated mask overlaps with
            overlapping = np.unique(segments[dilated & ~mask])
            # Add edges to adjacency matrix
            for j in overlapping:
                if j != i and j >= 0 and j < n_nodes:  # Valid node index
                    adjacency[i, j] = adjacency[j, i] = 1

    # Create Spektral Graph object
    return Graph(x=node_features, a=adjacency)


# Function to process directories and create graph datasets
def create_graph_dataset(directory):
    graph_list = []
    labels = []
    file_paths = []

    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            label = 1 if class_name.lower() == "tumor" else 0

            # Get all images for this class
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Processing {len(image_files)} images from {class_name}...")

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                try:
                    graph = image_to_graph(img_path)
                    graph_list.append(graph)
                    labels.append(label)
                    file_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    return graph_list, np.array(labels), file_paths


# Create a custom dataset class
class BrainTumorDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels
        super().__init__()

    def read(self):
        return self.graphs

    def __getitem__(self, index):
        return self.graphs[index]

    def __len__(self):
        return len(self.graphs)


# Build improved Graph CNN model with additional techniques for medical imaging
def build_gcnn_model(n_features):
    # Define model inputs
    node_features = keras.Input(shape=(n_features,), name='node_features')
    adjacency = keras.Input(shape=(None,), name='adjacency', sparse=True)

    # Graph convolution layers with skip connections for better gradient flow
    graph_conv_1 = GCNConv(64, activation='relu')([node_features, adjacency])
    graph_conv_1_bn = layers.BatchNormalization()(graph_conv_1)

    graph_conv_2 = GCNConv(64, activation='relu')([graph_conv_1_bn, adjacency])
    graph_conv_2_bn = layers.BatchNormalization()(graph_conv_2)

    # Residual connection
    graph_conv_3 = GCNConv(32, activation='relu')([graph_conv_2_bn, adjacency])
    graph_conv_3_bn = layers.BatchNormalization()(graph_conv_3)

    # Global pooling - using both sum and max for better feature capture
    pooled_sum = GlobalSumPool()(graph_conv_3_bn)

    # Dense layers with dropout for regularization
    dense_1 = layers.Dense(32, activation='relu')(pooled_sum)
    dropout_1 = layers.Dropout(0.3)(dense_1)
    dense_2 = layers.Dense(16, activation='relu')(dropout_1)
    dropout_2 = layers.Dropout(0.2)(dense_2)

    # Output layer
    output = layers.Dense(1, activation='sigmoid')(dropout_2)

    # Create model
    model = Model(inputs=[node_features, adjacency], outputs=output)

    # Compile with appropriate metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(),
                 keras.metrics.Recall(),
                 keras.metrics.AUC()]
    )

    return model


# Main execution block
print("Starting Brain Tumor Detection using Graph CNN...")

# Check if saved model exists
if os.path.exists(model_save_path):
    print(f"Loading saved model from {model_save_path}")
    model = load_model(model_save_path, custom_objects={'GCNConv': GCNConv, 'GlobalSumPool': GlobalSumPool})
    retrain = False
else:
    retrain = True

# Process datasets
if retrain:
    print("Creating graph datasets from images...")
    train_graphs, train_labels, _ = create_graph_dataset(train_dir)

    # Create and train the model
    print("Building and training Graph CNN model...")
    n_features = train_graphs[0].x.shape[1]
    model = build_gcnn_model(n_features)

    # Convert to Spektral dataset
    train_dataset = BrainTumorDataset(train_graphs, train_labels)

    # Prepare data loader for training
    from spektral.data import BatchLoader

    loader_train = BatchLoader(train_dataset, batch_size=8, shuffle=True)

    # Learning rate scheduler for better convergence
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1
    )

    # Early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )

    # Train the model
    history = model.fit(
        loader_train.load(),
        steps_per_epoch=loader_train.steps_per_epoch,
        epochs=10,
        verbose=1,
        callbacks=[lr_scheduler]
    )

    # Save the model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()

# Process test dataset
print("Processing test dataset...")
test_graphs, test_labels, test_file_paths = create_graph_dataset(test_dir)
test_dataset = BrainTumorDataset(test_graphs, test_labels)
loader_test = spektral.data.BatchLoader(test_dataset, batch_size=1)

# Predict on test set
print("Making predictions on test dataset...")
y_pred_probs = []

for batch in loader_test:
    inputs, _ = batch
    batch_preds = model.predict(inputs)
    y_pred_probs.extend(batch_preds.flatten())

y_pred_probs = np.array(y_pred_probs)
y_pred = (y_pred_probs > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred)
recall = recall_score(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred)
specificity = (np.sum((y_pred == 0) & (test_labels == 0)) /
               np.sum(test_labels == 0))
g_mean = np.sqrt(recall * specificity)

# ROC curve and AUC
fpr, tpr, _ = roc_curve(test_labels, y_pred_probs)
roc_auc = auc(fpr, tpr)

# Save predictions to CSV
results_df = pd.DataFrame({
    'file_path': test_file_paths,
    'y_true': test_labels,
    'y_pred': y_pred,
    'y_pred_prob': y_pred_probs
})
results_df.to_csv('brain_tumor_gcnn_predictions.csv', index=False)
print("Predictions saved to brain_tumor_gcnn_predictions.csv")

# Display metrics
print("\nBrain Tumor Detection - Graph CNN Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"G-Mean: {g_mean:.4f}")
print(f"AUC: {roc_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(test_labels, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualizations
# ROC Curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# True vs Predicted Labels
plt.figure(figsize=(12, 6))
plt.scatter(range(len(test_labels)), test_labels, label="True Labels", color='blue', marker='o', alpha=0.7)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted Labels", color='red', marker='x', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Label (0: No Tumor, 1: Tumor)')
plt.title('True vs Predicted Labels')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
classes = ['No Tumor', 'Tumor']
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# Function to visualize the graph construction process for a sample image
def visualize_graph_construction(image_path):
    # Load and process the image
    img = io.imread(image_path)
    img = tf.image.resize(img, (512, 512)).numpy().astype('uint8')

    # Handle grayscale vs RGB
    if len(img.shape) == 2:  # Grayscale
        img_display = np.stack([img, img, img], axis=-1)
        gray_img = img / 255.0
    else:  # RGB
        img_display = img
        gray_img = color.rgb2gray(img)

    # Create mask to exclude background
    mask = gray_img > 0.05

    # Apply superpixel segmentation
    segments = slic(img_display, n_segments=100, compactness=10, start_label=0, mask=mask)

    # Create visualization of the segmentation
    from skimage.segmentation import mark_boundaries
    img_with_boundaries = mark_boundaries(img_display, segments)

    # Create a graph representation for visualization (simplified)
    n_nodes = segments.max() + 1

    # Find centroids of segments for node positions
    centroids = np.zeros((n_nodes, 2))
    for i in range(n_nodes):
        mask = segments == i
        if mask.sum() > 0:
            coords = np.column_stack(np.where(mask))
            centroids[i] = coords.mean(axis=0)

    # Create adjacency using the same approach as in image_to_graph
    adjacency = np.zeros((n_nodes, n_nodes))
    dilated_segments = np.copy(segments)

    for i in range(n_nodes):
        mask = segments == i
        if mask.sum() > 0:
            from scipy import ndimage
            dilated = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
            overlapping = np.unique(segments[dilated & ~mask])
            for j in overlapping:
                if j != i and j >= 0 and j < n_nodes:
                    adjacency[i, j] = adjacency[j, i] = 1

    # Visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_display if len(img_display.shape) == 3 else img_display[:, :, 0], cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_with_boundaries)
    plt.title(f'Superpixel Segmentation\n({n_nodes} segments)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_display if len(img_display.shape) == 3 else img_display[:, :, 0], cmap='gray', alpha=0.7)

    # Draw edges between connected segments
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adjacency[i, j] > 0:
                plt.plot([centroids[i, 1], centroids[j, 1]],
                         [centroids[i, 0], centroids[j, 0]], 'y-', alpha=0.3, linewidth=0.5)

    # Draw nodes
    for i in range(n_nodes):
        plt.plot(centroids[i, 1], centroids[i, 0], 'ro', markersize=3, alpha=0.5)

    plt.title(f'Graph Representation\n({np.sum(adjacency) / 2:.0f} edges)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Display a random test image with its prediction and graph visualization
def show_random_prediction_with_graph():
    idx = np.random.randint(0, len(test_labels))
    img_path = test_file_paths[idx]
    true_label = test_labels[idx]
    pred_label = y_pred[idx]
    pred_prob = y_pred_probs[idx]

    # Display the original image with prediction
    img = io.imread(img_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    plt.title(f"True: {'Tumor' if true_label == 1 else 'No Tumor'} | "
              f"Predicted: {'Tumor' if pred_label == 1 else 'No Tumor'} "
              f"(Prob: {pred_prob:.4f})")
    plt.axis('off')
    plt.show()

    # Visualize the graph construction process
    visualize_graph_construction(img_path)


# Show random prediction with graph visualization
show_random_prediction_with_graph()

print("Analysis complete!")