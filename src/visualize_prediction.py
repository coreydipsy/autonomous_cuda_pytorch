# visualize_prediction.py
import torch
import matplotlib.pyplot as plt
from tusimple_dataset import TuSimpleLaneDataset
from model import TinyCNN
from model import BiggerCNN
from model import UNet

# Load trained model
model = UNet()
model.load_state_dict(torch.load("lane_model.pth", map_location=torch.device('cpu')))
model.eval()

# Load a sample from dataset
dataset = TuSimpleLaneDataset(
    image_dir="../TuSimple/processed/images",
    mask_dir="../TuSimple/processed/masks"
)

image, mask = dataset[0]
with torch.no_grad():
    pred = model(image.unsqueeze(0))  # Add batch dimension
    pred = pred.squeeze().numpy()     # Remove batch + channel dims

# Plot
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(image.permute(1, 2, 0))  # CHW → HWC
axs[0].set_title("Input Image")
axs[1].imshow(mask.squeeze(), cmap='gray')
axs[1].set_title("Ground Truth Mask")
axs[2].imshow(pred, cmap='gray')
axs[2].set_title("Model Prediction")
for ax in axs: ax.axis('off')
plt.tight_layout()
plt.show()
