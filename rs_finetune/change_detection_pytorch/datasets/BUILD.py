import os
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np


class BuildingDataset(Dataset):
    def __init__(self, file_list_path, bands=None, mean=None, std=None, transform=None):
        """
        Args:
            file_list_path (str): Path to the .txt file containing folder paths.
            bands (list): List of band indices (e.g., [1, 2, 3]).
            mean (list or np.array): Per-channel mean for normalization.
            std (list or np.array): Per-channel std for normalization.
            transform (callable, optional): Transform to apply to the data.
        """
        with open(file_list_path, 'r') as f:
            self.folders = [line.strip() for line in f.readlines()]
        self.bands = bands
        self.mean = np.array(mean) if mean is not None else None
        self.std = np.array(std) if std is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]

        # Load mask
        mask_path = os.path.join(folder, "buildings10m.tif")
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)  # Read the first band as the mask

        # Load image bands
        images = []
        for band in self.bands:
            band_path = os.path.join(folder, f"B{band}.tif")
            with rasterio.open(band_path) as src:
                images.append(src.read(1))  # Read the first band

        # Stack bands into a single array
        image = np.stack(images, axis=0)  # Shape: (num_bands, H, W)

        # Normalize using mean and std
        if self.mean is not None and self.std is not None:
            image = (image - self.mean[:, None, None]) / self.std[:, None, None]

        # Apply additional transformations
        if self.transform:
            image, mask = self.transform(image, mask)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# Example usage
if __name__ == "__main__":
    # File path to the training set
    train_file = "train.txt"

    # Specify bands to include and their normalization statistics
    bands = [1, 2, 3, 4]  # Example: Bands B1, B2, B3, B4
    mean = [0.45, 0.43, 0.44, 0.42]  # Example: Per-channel mean
    std = [0.21, 0.22, 0.23, 0.24]   # Example: Per-channel std

    # Initialize dataset
    train_dataset = BuildingDataset(
        file_list_path=train_file,
        bands=bands,
        mean=mean,
        std=std
    )

    # Initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Iterate through the DataLoader
    for images, masks in train_loader:
        print("Image Shape:", images.shape)  # (batch_size, num_bands, H, W)
        print("Mask Shape:", masks.shape)    # (batch_size, H, W)
        break
