import torch.nn as nn

# Define a custom neural network by subclassing nn.Module
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()

        # The encoder compresses spatial information and extracts features
        self.encoder = nn.Sequential(
            # First convolution layer:
            # - Input: 3-channel RGB image
            # - Output: 16 feature maps
            # - 3x3 kernel with padding=1 keeps input size
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Normalize activations to stabilize training
            nn.ReLU(),

            # Second convolution:
            # - Input: 16 channels → Output: 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Helps convergence and gradient flow
            nn.ReLU(),

            # Downsampling:
            # - Halve spatial resolution
            nn.MaxPool2d(kernel_size=2)  # 128x128 → 64x64
        )

        # The decoder reconstructs the segmentation mask
        self.decoder = nn.Sequential(
            # Upsampling: reverse the downsampling
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 64x64 → 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Final layer: reduce to 1-channel mask
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Output pixel probabilities between 0 and 1
        )

    def forward(self, x):
        # Forward pass through encoder
        x = self.encoder(x)

        # Forward pass through decoder
        x = self.decoder(x)

        # Output is a 1-channel probability map of the same spatial size as input
        return x
    

class BiggerCNN(nn.Module):
    def __init__(self):
        super(BiggerCNN, self).__init__()

        # Encoder: Downsample and extract features
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # [B, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B, 64, 64, 64]

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [B, 128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# [B, 128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B, 128, 32, 32]
        )

        # Decoder: Upsample back to original size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # [B, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=1),  # Final binary mask
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

