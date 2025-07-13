import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TinyCNN
from dataset import DummyLaneDataset
import matplotlib.pyplot as plt

# Basic training loop
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyCNN().to(device)                    # Load model on GPU or CPU
    dataset = DummyLaneDataset()                    # Dummy dataset
    loader = DataLoader(dataset, batch_size=8, shuffle=True)  # Batch loader

    criterion = nn.BCELoss()                        # Loss for binary output
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):                          # Train for 5 epochs
        total_loss = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)                 # Forward pass
            loss = criterion(outputs, targets)      # Compute loss
            loss.backward()                         # Backpropagation
            optimizer.step()                        # Update weights

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Visualization function to test model on a sample input
def visualize_sample(model, dataset, device):
    model.eval()
    with torch.no_grad():
        sample_input, sample_target = dataset[0]
        input_batch = sample_input.unsqueeze(0).to(device)  # Add batch dimension

        # Run model to get prediction
        output = model(input_batch).cpu().squeeze(0).squeeze(0).numpy()

        # Show input image, target mask, and predicted mask
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(sample_input.permute(1, 2, 0).numpy())  # CHW → HWC
        axs[0].set_title("Input Image")
        axs[1].imshow(sample_target.squeeze(0).numpy(), cmap="gray")
        axs[1].set_title("Ground Truth")
        axs[2].imshow(output, cmap="gray")
        axs[2].set_title("Model Prediction")
        plt.tight_layout()
        plt.show()

# Run training and visualize output
if __name__ == "__main__":
    train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCNN().to(device)
    dataset = DummyLaneDataset()
    visualize_sample(model, dataset, device)
