import os

# Replace with your actual dataset path
dataset_path = "UTKFace/part1"
# Check if the dataset path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The dataset path '{dataset_path}' does not exist.")

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# Count images
image_count = sum(
    1 for file in os.listdir(dataset_path) 
    if file.lower().endswith(image_extensions)
)

print(f"Total number of images: {image_count}")