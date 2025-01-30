import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np


class BuildingDataset(Dataset):
    def __init__(self, split_list, bands=None, mean=None, std=None, transform=None):
        """
        Args:
            split_list (str): Path to the .txt file containing folder paths.
            bands (list): List of band indices (e.g., [1, 2, 3]).
            mean (list or np.array): Per-channel mean for normalization.
            std (list or np.array): Per-channel std for normalization.
            transform (callable, optional): Transform to apply to the data.
        """
        with open(split_list, 'r') as f:
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
            folder_path = os.path.join(folder, "B")
            band_path = next((f for f in os.listdir(folder_path) if f.endswith(f"{band}.tif")), None)
            band_path = os.path.join(folder_path, band_path)
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

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        return image_tensor, mask_tensor