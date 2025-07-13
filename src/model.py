import torch.nn as nn
import torch

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
    
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),  # third conv
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)  # binary mask output

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out(d1))
