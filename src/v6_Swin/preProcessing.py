import os
from PIL import Image

# Hard-coded file paths
input_root = r"D:\\ED\\braintumor\\data"  # Input directory containing training/testing subfolders
output_root = r"D:\\ED\\BrainTumor_v2\\data"  # Output directory for resized images
quality = 100  # JPEG quality (1-95)
target_size = (128, 128)  # Desired image size


def convert_and_compress(input_root, output_root, target_size=(128, 128), quality=100):
    """
    Recursively converts images from the input_root folder (which should contain
    subfolders 'training' and 'testing', each with 'notumor' and 'tumor' subfolders)
    to a target size with optimized compression and saves them in the output_root folder,
    preserving the folder structure.
    """
    for root, dirs, files in os.walk(input_root):
        for file in files:
            # Process only common image formats
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                input_path = os.path.join(root, file)
                # Create a relative path from input_root
                rel_path = os.path.relpath(root, input_root)
                # Construct the corresponding output folder path
                output_folder = os.path.join(output_root, rel_path)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output_path = os.path.join(output_folder, file)

                try:
                    with Image.open(input_path) as im:
                        im = im.convert("RGB")  # Ensure image is in RGB mode
                        im_resized = im.resize(target_size, Image.LANCZOS)
                        # Save as JPEG with optimization enabled
                        im_resized.save(output_path, format="JPEG", optimize=True, quality=quality)
                        print(f"Converted and saved: {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")


def main():
    print("Starting conversion...")
    convert_and_compress(input_root, output_root, target_size, quality)
    print("Conversion complete.")


if __name__ == "__main__":
    main()
