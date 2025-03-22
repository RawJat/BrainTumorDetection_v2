import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import spektral
from spektral.layers import GCNConv, GlobalMaxPool
from spektral.utils import gcn_filter
import random
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from scipy.spatial.distance import pdist, squareform
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Paths
train_dir = r"D:\ED\braintumor\data\Training"
test_dir = r"D:\ED\braintumor\data\Testing"
model_save_path = "brain_tumor_graph_cnn_model.h5"
csv_save_path = "prediction_results.csv"

# Parameters
IMG_SIZE = (128, 128)  # Reduced size for faster processing
BATCH_SIZE = 16
EPOCHS = 10
N_CLASSES = 2
N_SUPERPIXELS = 100  # Number of superpixels for graph construction


# Function to load and preprocess images
def load_data(directory, img_size):
    images = []
    labels = []

    # Load tumor images (label 1)
    tumor_dir = os.path.join(directory, 'tumor')
    for img_name in os.listdir(tumor_dir):
        img_path = os.path.join(tumor_dir, img_name)
        try:
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(1)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    # Load notumor images (label 0)
    notumor_dir = os.path.join(directory, 'notumor')
    for img_name in os.listdir(notumor_dir):
        img_path = os.path.join(notumor_dir, img_name)
        try:
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(0)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    return np.array(images), np.array(labels)


# Convert image to graph
def image_to_graph(image, n_superpixels=100, compactness=10):
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[2] == 1:
        image = np.concatenate([image, image, image], axis=-1)

    # Ensure image values are in [0, 1]
    if image.max() > 1.0:
        image = image / 255.0

    # Generate superpixels
    segments = slic(image, n_segments=n_superpixels, compactness=compactness)

    # Extract node features (mean color and position for each superpixel)
    n_nodes = segments.max() + 1
    node_features = np.zeros((n_nodes, 5))  # 3 for RGB, 2 for position

    for i in range(n_nodes):
        mask = segments == i
        if mask.sum() > 0:
            # Mean color
            node_features[i, :3] = np.mean(image[mask], axis=0)

            # Mean position (normalized)
            rows, cols = np.where(mask)
            node_features[i, 3] = np.mean(rows) / image.shape[0]
            node_features[i, 4] = np.mean(cols) / image.shape[1]

    # Create adjacency matrix based on spatial proximity
    centroids = node_features[:, 3:5]
    distances = squareform(pdist(centroids, metric='euclidean'))
    sigma = 0.1  # bandwidth parameter
    adjacency = np.exp(-distances ** 2 / (2 * sigma ** 2))

    # Set diagonal to 0
    np.fill_diagonal(adjacency, 0)

    # Normalize adjacency matrix using GCN normalization
    adjacency_normalized = gcn_filter(adjacency)

    return node_features, adjacency_normalized, segments


# Process a batch of images
def process_batch(images, labels=None):
    batch_node_features = []
    batch_adjacency = []
    batch_labels = []

    for i, image in enumerate(images):
        node_features, adjacency, _ = image_to_graph(image, N_SUPERPIXELS)
        batch_node_features.append(node_features)
        batch_adjacency.append(adjacency)
        if labels is not None:
            batch_labels.append(labels[i])

    if labels is not None:
        return batch_node_features, batch_adjacency, np.array(batch_labels)
    else:
        return batch_node_features, batch_adjacency


# Custom data generator for graph data
class GraphDataGenerator(keras.utils.Sequence):
    def __init__(self, images, labels, batch_size, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = [self.images[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]

        # Process batch to create graph data
        X_node, X_adj, y = process_batch(np.array(batch_images), np.array(batch_labels))

        # For binary classification
        y_one_hot = to_categorical(y, num_classes=N_CLASSES)

        return [X_node, X_adj], y_one_hot

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# Build Graph CNN model
def build_graph_cnn_model(n_node_features, n_classes):
    # Define inputs for node features and adjacency matrix
    node_features_input = keras.Input(shape=(None, n_node_features), name="node_features")
    adjacency_input = keras.Input(shape=(None, None), name="adjacency_matrix")

    # Graph convolutional layers
    x = GCNConv(32, activation="relu")([node_features_input, adjacency_input])
    x = layers.Dropout(0.2)(x)
    x = GCNConv(64, activation="relu")([x, adjacency_input])
    x = layers.Dropout(0.2)(x)
    x = GCNConv(128, activation="relu")([x, adjacency_input])

    # Pooling layer to get a fixed-size representation
    x = GlobalMaxPool()(x)

    # Dense layers for classification
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    # Create model
    model = keras.Model(inputs=[node_features_input, adjacency_input], outputs=outputs)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Metrics calculation
def calculate_metrics(y_true, y_pred, y_pred_prob=None):
    # Convert to binary predictions if probabilities are provided
    if y_pred_prob is not None:
        y_pred = (y_pred_prob[:, 1] > 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate specificity (true negative rate)
    specificity = tn / (tn + fp)

    # Calculate geometric mean of sensitivity and specificity
    gmean = np.sqrt(recall * specificity)

    # Calculate AUC if probabilities are provided
    auc_score = None
    if y_pred_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
        auc_score = auc(fpr, tpr)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Specificity': specificity,
        'G-Mean': gmean,
        'AUC': auc_score
    }

    return metrics, cm, (fpr, tpr) if y_pred_prob is not None else None


# Visualization functions
def plot_confusion_matrix(cm, class_names=['No Tumor', 'Tumor']):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(fpr, tpr, auc_score):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_predictions_vs_true(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_true)), y_true, label="True Labels", color='blue', marker='o', alpha=0.7)
    plt.scatter(range(len(y_pred)), y_pred, label="Predicted Labels", color='red', marker='x', alpha=0.7)
    plt.title('True vs Predicted Labels')
    plt.xlabel('Sample Index')
    plt.ylabel('Label')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def visualize_random_prediction(X_test, y_true, y_pred, segments_list=None):
    idx = np.random.randint(0, len(X_test))
    image = X_test[idx]
    true_label = y_true[idx]
    pred_label = y_pred[idx]

    plt.figure(figsize=(10, 5))

    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Display image with graph structure (if segments are available)
    plt.subplot(1, 2, 2)
    if segments_list is not None and idx < len(segments_list):
        plt.imshow(mark_boundaries(image, segments_list[idx]))
        plt.title('Graph Structure (Superpixels)')
    else:
        plt.imshow(image)
        plt.title('Processed Image')

    plt.suptitle(f'True: {"Tumor" if true_label == 1 else "No Tumor"}, ' +
                 f'Predicted: {"Tumor" if pred_label == 1 else "No Tumor"} ' +
                 f'{"(Correct)" if true_label == pred_label else "(Incorrect)"}',
                 fontsize=14)
    plt.axis('off')
    plt.show()


# Main execution
def main():
    print("Loading training data...")
    X_train, y_train = load_data(train_dir, IMG_SIZE)
    print(f"Loaded {len(X_train)} training images")

    print("Loading testing data...")
    X_test, y_test = load_data(test_dir, IMG_SIZE)
    print(f"Loaded {len(X_test)} testing images")

    # Creating training data generator
    train_generator = GraphDataGenerator(X_train, y_train, BATCH_SIZE)

    # Get node feature dimension from a sample
    sample_node_features, _, _ = image_to_graph(X_train[0], N_SUPERPIXELS)
    n_node_features = sample_node_features.shape[1]

    # Build and train the model
    print("Building Graph CNN model...")
    model = build_graph_cnn_model(n_node_features, N_CLASSES)
    print(model.summary())

    # Check if saved model exists
    if os.path.exists(model_save_path):
        print(f"Loading saved model from {model_save_path}")
        model = keras.models.load_model(model_save_path,
                                        custom_objects={"GCNConv": GCNConv,
                                                        "GlobalMaxPool": GlobalMaxPool})
    else:
        print("Training the model...")
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            verbose=1
        )

        # Save the model
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

    # Process test data
    print("Processing test data for evaluation...")
    test_nodes = []
    test_adjs = []
    test_segments = []

    for img in X_test:
        node_features, adjacency, segments = image_to_graph(img, N_SUPERPIXELS)
        test_nodes.append(node_features)
        test_adjs.append(adjacency)
        test_segments.append(segments)

    # Get predictions
    print("Making predictions...")
    y_pred_prob = []
    pred_batch_size = 8  # Smaller batch size for prediction to manage memory

    for i in range(0, len(X_test), pred_batch_size):
        batch_nodes = test_nodes[i:i + pred_batch_size]
        batch_adjs = test_adjs[i:i + pred_batch_size]
        batch_pred = model.predict([batch_nodes, batch_adjs])
        y_pred_prob.extend(batch_pred)

    y_pred_prob = np.array(y_pred_prob)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Calculate metrics
    print("Calculating metrics...")
    metrics, cm, roc_data = calculate_metrics(y_test, y_pred, y_pred_prob)

    # Print metrics
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")

    # Save predictions to CSV
    results_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'prob_tumor': y_pred_prob[:, 1] if y_pred_prob.shape[1] > 1 else y_pred_prob.flatten()
    })
    results_df.to_csv(csv_save_path, index=False)
    print(f"Predictions saved to {csv_save_path}")

    # Create visualizations
    print("Creating visualizations...")
    # Confusion matrix
    plot_confusion_matrix(cm)

    # ROC curve
    if roc_data is not None:
        fpr, tpr = roc_data
        plot_roc_curve(fpr, tpr, metrics['AUC'])

    # True vs predicted labels
    plot_predictions_vs_true(y_test, y_pred)

    # Visualize a random prediction
    visualize_random_prediction(X_test, y_test, y_pred, test_segments)

    print("Done!")


if __name__ == "__main__":
    main()