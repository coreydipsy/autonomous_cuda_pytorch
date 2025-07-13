import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

# Custom Dataset class to load images and corresponding binary lane masks
class TuSimpleLaneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        # Paths to the directory containing input images and corresponding masks
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Collect all image filenames ending with .jpg
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

        # If a transform is passed, use it. Otherwise, use default: convert image to tensor
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),  # Converts (H, W, C) uint8 image to (C, H, W) float in [0.0, 1.0]
        ])

    def __len__(self):
        # Total number of samples in the dataset
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Get the full file paths for the image and its corresponding binary mask
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])  # assumes same filename

        # Load the image and mask using OpenCV
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # load mask as 1-channel grayscale

        # Raise an error if either file is missing or corrupted
        if image is None or mask is None:
            raise RuntimeError(f"Could not load {img_path} or {mask_path}")

        # Resize both image and mask to the target input size of the model (64x64)
        image = cv2.resize(image, (128, 128))
        mask = cv2.resize(mask, (128, 128))

        # Apply transformation to image (e.g., convert to tensor)
        image = self.transform(image)

        # Convert mask from NumPy array to PyTorch tensor
        # - float(): convert from uint8 to float32
        # - unsqueeze(0): add channel dimension to make shape (1, 64, 64)
        # - / 255.0: normalize binary values to 0.0 or 1.0
        mask = mask.astype(np.float32) / 255.0  # Normalize to [0, 1]
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension

        # Return a tuple of (input image tensor, label mask tensor)
        return image, mask
