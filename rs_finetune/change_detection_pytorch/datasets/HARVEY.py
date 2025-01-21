import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np


class FloodDataset(Dataset):
    def __init__(self, split_list, bands=None, transform=None):
        """
        Args:
            file_list_path (str): Path to the .txt file containing folder paths.
            bands (list): List of band names (e.g., ['B1', 'B2', 'B3']).
            transform (callable, optional): Transform to apply to the data.
        """
        with open(split_list, 'r') as f:
            self.folders = [line.strip() for line in f.readlines()]
        self.bands = bands
        self.transform = transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]

        # Load mask
        mask_path = os.path.join(folder, "flooded10m.tif")
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)  # Read the first band as the mask

        # Load "before" and "after" images
        before_images, after_images = [], []
        for band in self.bands:
            before_path = os.path.join(folder, "B" f"{band}.tif")
            after_path = os.path.join(folder, "A" f"{band}.tif")

            with rasterio.open(before_path) as src:
                before_images.append(src.read(1))  
            
            with rasterio.open(after_path) as src:
                after_images.append(src.read(1))  

        # Stack bands into a single array
        before_image = np.stack(before_images, axis=0)  # Shape: (num_bands, H, W)
        after_image = np.stack(after_images, axis=0)    # Shape: (num_bands, H, W)

        if self.transform:
            before_image, after_image, mask = self.transform(before_image, after_image, mask)

        return torch.tensor(before_image, dtype=torch.float32), \
               torch.tensor(after_image, dtype=torch.float32), \
               torch.tensor(mask, dtype=torch.float32)

