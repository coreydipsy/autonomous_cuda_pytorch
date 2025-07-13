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
import os


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

# Create dataset and dataloader
dataset = TuSimpleLaneDataset(IMAGE_DIR, MASK_DIR)
print("Total samples in dataset:", len(dataset))

# Split into 80% train and 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Dataloaders
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
