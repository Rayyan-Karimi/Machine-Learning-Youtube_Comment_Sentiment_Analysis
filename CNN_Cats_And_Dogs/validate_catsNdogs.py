from PIL import Image
import os

def validate_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify the integrity of the image
            except Exception as e:
                print(f"Removing invalid image: {file_path} - Error: {e}")
                os.remove(file_path)

# Validate train and validation datasets
validate_images('dataset/train')
validate_images('dataset/test')
