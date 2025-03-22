import os
import numpy as np
import pandas as pd
from skimage.segmentation import slic
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, filters, measure
from scipy import ndimage
import tensorflow as tf
import multiprocessing
from tqdm import tqdm
import pickle
import warnings
import h5py

warnings.filterwarnings('ignore')


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

        # Resize image - use more efficient method with OpenCV if possible
        img = tf.image.resize(img, (256, 256)).numpy().astype('uint8')  # Reduced size for faster processing
        gray_img = color.rgb2gray(img)
        mask = gray_img > 0.05  # exclude dark background

        # Superpixel segmentation with fewer segments for speed
        segments = slic(img, n_segments=n_segments, compactness=compactness,
                        start_label=0, mask=mask, sigma=1)  # Added sigma for smoother segments
        n_nodes = segments.max() + 1
        if n_nodes < 2:
            # Fallback segmentation with grid
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

        # Extract node features (reduced to 8 essential features for efficiency)
        node_features = np.zeros((n_nodes, 8))
        edges = filters.sobel(gray_img)

        # Pre-calculate region props once
        region_props = measure.regionprops(segments + 1, intensity_image=gray_img)
        region_dict = {prop.label - 1: prop for prop in region_props}

        for i in range(n_nodes):
            mask_seg = segments == i
            if mask_seg.sum() > 0:
                region = gray_img[mask_seg]
                # Basic intensity features
                node_features[i, 0] = np.mean(region)
                node_features[i, 1] = np.std(region) if len(region) > 1 else 0
                node_features[i, 2] = np.mean(edges[mask_seg])

                # Texture features - simplified
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
                            node_features[i, 3] = graycoprops(glcm, 'contrast').mean()
                            node_features[i, 4] = graycoprops(glcm, 'homogeneity').mean()
                    except Exception:
                        node_features[i, 3:5] = 0

                # Shape features
                if i in region_dict:
                    prop = region_dict[i]
                    node_features[i, 5] = prop.area / mask_seg.size
                    node_features[i, 6] = prop.perimeter / (np.sqrt(mask_seg.size) + 1e-8)
                    node_features[i, 7] = prop.eccentricity

        # Normalize features
        for i in range(node_features.shape[1]):
            col_max = np.max(node_features[:, i])
            col_min = np.min(node_features[:, i])
            if col_max > col_min:
                node_features[:, i] = (node_features[:, i] - col_min) / (col_max - col_min + 1e-8)

        # Create adjacency matrix more efficiently
        adjacency = np.zeros((n_nodes, n_nodes))

        # Use a more efficient algorithm to find adjacency
        # Dilate each segment mask and check for overlaps
        for i in range(n_nodes):
            mask_seg = segments == i
            if mask_seg.sum() > 0:
                dilated = ndimage.binary_dilation(mask_seg)
                neighbors = np.unique(segments[dilated & ~mask_seg])
                for j in neighbors:
                    if j != i and 0 <= j < n_nodes:
                        adjacency[i, j] = adjacency[j, i] = 1

        return node_features, adjacency, segments
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None


def process_image(args):
    img_path, label = args
    node_features, adjacency, segments = image_to_graph(img_path, n_segments=75, compactness=15)
    if node_features is not None and adjacency is not None:
        return {
            'path': img_path,
            'features': node_features,
            'adjacency': adjacency,
            'label': label
        }
    return None


def process_directory(directory, output_file):
    image_data = []
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            label = 1 if class_name.lower() == "tumor" else 0
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Processing {len(image_files)} images from {class_name}...")

            # Create processing arguments
            args_list = [(os.path.join(class_dir, img_file), label) for img_file in image_files]

            # Use multiprocessing to speed up processing
            n_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free for system
            print(f"Using {n_cores} CPU cores for parallel processing")

            with multiprocessing.Pool(processes=n_cores) as pool:
                results = list(tqdm(pool.imap(process_image, args_list), total=len(args_list)))

            # Filter out None results
            valid_results = [r for r in results if r is not None]
            image_data.extend(valid_results)
            print(f"Class {class_name}: Successfully processed {len(valid_results)} of {len(image_files)} images")

    # Save processed data in efficient HDF5 format
    with h5py.File(output_file, 'w') as f:
        for i, data in enumerate(image_data):
            group = f.create_group(f'sample_{i}')
            group.create_dataset('features', data=data['features'])
            group.create_dataset('adjacency', data=data['adjacency'])
            group.attrs['label'] = data['label']
            group.attrs['path'] = data['path']

    print(f"Saved {len(image_data)} processed graphs to {output_file}")
    return image_data


if __name__ == "__main__":
    train_dir = r"D:\ED\braintumor\data\Training"
    test_dir = r"D:\ED\braintumor\data\Testing"

    train_output = "brain_tumor_train_graphs.h5"
    test_output = "brain_tumor_test_graphs.h5"

    print("Processing training data...")
    train_data = process_directory(train_dir, train_output)

    print("Processing test data...")
    test_data = process_directory(test_dir, test_output)

    print("Preprocessing complete! Graph data saved to disk.")