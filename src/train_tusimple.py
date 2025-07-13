# train_tusimple.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tusimple_dataset import TuSimpleLaneDataset  # Our custom dataset class
from model import TinyCNN  # Our binary segmentation model
from model import BiggerCNN
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
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize the model and move it to GPU (or CPU fallback)
# model = TinyCNN().to(device)
model = BiggerCNN().to(device)

# Define loss function: Binary Cross Entropy for pixel-wise binary classification
criterion = combined_loss

# Use Adam optimizer with learning rate of 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Number of epochs to train
EPOCHS = 1000

# Training loop
for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    total_loss = 0

    for images, masks in dataloader:
        images = images.to(device)  # Move input images to GPU
        masks = masks.to(device)    # Move ground truth masks to GPU

        optimizer.zero_grad()       # Clear previous gradients
        outputs = model(images)     # Forward pass
        loss = criterion(outputs, masks)  # Compare predictions to ground truth
        loss.backward()             # Backpropagation: compute gradients
        optimizer.step()            # Update model weights

        total_loss += loss.item()

    # Print loss for this epoch
    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {total_loss / len(dataloader):.4f}")

# Save the trained model weights
torch.save(model.state_dict(), "lane_model.pth")
print("Training complete and model saved!")
