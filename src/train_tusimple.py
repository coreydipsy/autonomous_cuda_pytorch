# train_tusimple.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tusimple_dataset import TuSimpleLaneDataset  # Our custom dataset class
from model import TinyCNN  # Our binary segmentation model
from model import BiggerCNN
from model import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=5, p=0.4),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])


def dice_loss(pred, target, smooth=1e-6):
    """
    Compute Dice Loss: a measure of overlap between prediction and target masks.
    Higher overlap → lower loss.
    """
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)  # sum over H and W

    dice = (2. * intersection + smooth) / (
        pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth
    )

    return 1 - dice.mean()

bce = nn.BCELoss()

def combined_loss(pred, target):
    return 0.5 * bce(pred, target) + 0.5 * dice_loss(pred, target)

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths to the preprocessed images and masks
IMAGE_DIR = os.path.join("..", "TuSimple", "processed", "images")
MASK_DIR = os.path.join("..", "TuSimple", "processed", "masks")




# Create training and validation datasets using transforms
full_dataset = TuSimpleLaneDataset(IMAGE_DIR, MASK_DIR)

# Split into 80% train and 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size 

# Get consistent filenames
all_indices = list(range(len(full_dataset)))
train_indices = all_indices[:train_size]
val_indices = all_indices[train_size:]

# Split filenames for consistency
train_image_filenames = [full_dataset.image_filenames[i] for i in train_indices]
val_image_filenames = [full_dataset.image_filenames[i] for i in val_indices]

# Now rebuild datasets using image_filenames manually
train_dataset = TuSimpleLaneDataset(IMAGE_DIR, MASK_DIR, transform=train_transform)
train_dataset.image_filenames = train_image_filenames  # overwrite with selected subset

val_transform = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
val_dataset = TuSimpleLaneDataset(IMAGE_DIR, MASK_DIR, transform=val_transform)
val_dataset.image_filenames = val_image_filenames

# Now create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize the model and move it to GPU (or CPU fallback)
# model = TinyCNN().to(device)
# model = BiggerCNN().to(device)
model = UNet().to(device)

# Define loss function: Binary Cross Entropy for pixel-wise binary classification
criterion = combined_loss

# Use Adam optimizer with learning rate of 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Number of epochs to train
EPOCHS = 100

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


# Save the trained model weights
torch.save(model.state_dict(), "lane_model.pth")
print("Training complete and model saved!")
