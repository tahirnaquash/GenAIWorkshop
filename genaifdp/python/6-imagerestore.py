# code for "Smart Image Restorer: A GAN-based System for Enhancing Historical and Low-Resolution Images"
from email import generator
import os
import sys          
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms as T
from PIL import Image
from model import Generator
generator = Generator()  # Assuming you have a Generator class defined in model.py
import torch.optim as optim 
from dataset import ImageFolderDataset
def restore_image(input_image_path, output_image_path, model_path):
    # Load the pre-trained GAN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()    

    # Load and preprocess the input image
    input_image = Image.open(input_image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # Generate the restored image
    with torch.no_grad():
        restored_tensor = generator(input_tensor)

    # Post-process and save the restored image
    restored_image = restored_tensor.squeeze(0).cpu().clamp(0, 1)
    save_image(restored_image, output_image_path)   
    print(f"Restored image saved to {output_image_path}")
# Example usage
from PIL import Image
import torchvision.transforms as T
import torch
input_image_path = "C:\\Users\\Lenovo49\\genaifdp\\python\\input_image.jpg"
# Path to the low-resolution or historical image
output_image_path = "C:\\Users\\Lenovo49\\genaifdp\\python\\restored_image.jpg"
# Load image (if you have a file path)
input_image = Image.open(input_image_path).convert('L')  # 'L' for grayscale, 'RGB' for color

# Define your transforms
transform = T.Compose([
    T.ToTensor(),  # Converts to [C, H, W] and scales to [0, 1]
    # Add more transforms if needed (e.g., normalization)
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = transform(input_image).unsqueeze(0).to(device)  # Add batch dimension and move to device
# Ensure you have the necessary libraries installed:
# pip install torch torchvision pillow opencv-python         
if __name__ == "__main__":
    input_image_path = "input_image.jpg"  # Path to the low-resolution or historical image
    output_image_path = "restored_image.jpg"  # Path to save the restored image
    model_path = "pretrained_gan_model.pth"  # Path to the pre-trained GAN model
    if not os.path.exists(input_image_path):
        print(f"Input image {input_image_path} does not exist.")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist.")
        sys.exit(1)
    # Ensure the output directory exists
    dir_name = os.path.dirname(output_image_path)
    if dir_name:  # Only create directory if dir_name is not empty
        os.makedirs(dir_name, exist_ok=True)
    # Call the restore_image function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    with torch.no_grad():
        input_image = Image.open(input_image_path).convert('L')  # or 'RGB'
        transform = T.ToTensor()
        input_tensor = transform(input_image).unsqueeze(0).to(device)  # [1, C, H, W]  
        print("Input tensor shape:", input_tensor.shape)
        latent_dim = 100  # Must match your Generator's latent_dim
        z = torch.randn(1, latent_dim).to(device)
        restored_tensor = generator(z)
        restored_image = restored_tensor.squeeze(0).cpu().clamp(0, 1)
        save_image(restored_image, output_image_path)

# Ensure you have the necessary libraries installed:
# pip install torch torchvision pillow opencv-python
# Make sure to replace the model and dataset paths with your actual paths.


# This code is a simple implementation of a GAN-based image restoration system.
# It includes a function to restore images using a pre-trained GAN model.

# The code assumes you have a pre-trained GAN model and a dataset class defined in `model.py` and `dataset.py`.
# The `restore_image` function loads the model, processes the input image, generates a restored image, and saves it.
# Note: This code is a simplified version and may require adjustments based on your specific model and dataset.

# Ensure you have the necessary libraries installed:
# pip install torch torchvision pillow opencv-python
# Make sure to replace the model and dataset paths with your actual paths.
# This code is a simple implementation of a GAN-based image restoration system.
# It includes a function to restore images using a pre-trained GAN model.
# The code assumes you have a pre-trained GAN model and a dataset class defined in `model.py` and `dataset.py`.
# The `restore_image` function loads the model, processes the input image, generates a restored image, and saves it.
