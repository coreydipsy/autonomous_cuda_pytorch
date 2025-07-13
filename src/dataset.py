import torch
from torch.utils.data import Dataset

# Create a dummy dataset to simulate image-mask pairs
class DummyLaneDataset(Dataset):
    def __init__(self, size=100, img_size=(3, 64, 64)):
        # Generate random input images: size x channels x height x width
        self.data = torch.rand(size, *img_size)

        # Generate random masks (ground truth): size x 1 x height x width
        self.labels = torch.randint(0, 2, (size, 1, img_size[1], img_size[2])).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return one (image, mask) pair
        return self.data[idx], self.labels[idx]