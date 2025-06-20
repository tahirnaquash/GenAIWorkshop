import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform or transforms.ToTensor()
        self.image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')  # 'L' for grayscale, change as needed
        if self.transform:
            image = self.transform(image)
        return image