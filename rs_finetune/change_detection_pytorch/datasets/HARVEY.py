import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np


STATS = {
    'mean': {
        'B1': 1743.6986580141129,
        'B2': 1322.2082124589308,
        'B3': 1426.125276342686,
        'B4': 1519.4304951183542,
        'B5': 1458.3511756085722,
        'B6': 2189.1243839605736,
        'B7': 2612.642424348492,
        'B8': 2432.5066766189702,
        'B8A': 2845.825088205645,
        'B9': 420.1201108870968,
        'B10': 10.09842279905914,
        'B11': 2023.2420465576463,
        'B12': 1295.3533746826463,
        'vv': -8.462828636169434,
        'vh': -8.510013580322266,
        },
    'std': {
        'B1': 771.5183806369524,
        'B2': 1056.4326625501772,
        'B3': 900.3624734298407,
        'B4': 873.4760016751915,
        'B5': 988.208460744263,
        'B6': 967.7348032162768,
        'B7': 1033.7542057178632,
        'B8': 1017.1386786261996,
        'B8A': 1101.856993173505,
        'B9': 301.371729572531,
        'B10': 2.74859003425678,
        'B11': 1066.2916584258733,
        'B12': 910.9452607474732,
        'vv': 4.903507709503174,
        'vh': 3.7153587341308594,
        }
}


WAVES = {
    "B2": 0.493,
    "B3": 0.56,
    "B4": 0.665,
    "B5": 0.704,
    "B6": 0.74,
    "B7": 0.783,
    "B8": 0.842,
    "B8A": 0.865,
    "B11": 1.61,
    "B12": 2.19,
    'VV': 3.5,
    'VH': 4.0
}

def normalize_channel(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img

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
            b_folder = os.path.join(folder, "B")
            a_folder = os.path.join(folder, "A")
            b_matching_file = next((f for f in os.listdir(b_folder) if f.endswith(f"{band}.tif")), None)
            a_matching_file = next((f for f in os.listdir(a_folder) if f.endswith(f"{band}.tif")), None)

            if b_matching_file is None or a_matching_file is None:
                raise FileNotFoundError(f"Missing band {band} files in folder {folder}")


            before_path = os.path.join(b_folder, b_matching_file)
            after_path = os.path.join(a_folder, a_matching_file)


            # before_path = os.path.join(folder, "B", f"{band}.tif")
            # after_path = os.path.join(folder, "A" f"{band}.tif")

            with rasterio.open(before_path) as src:
                ch = src.read(1)
                ch = normalize_channel(ch, mean=STATS['mean'][band], std=STATS['std'][band])
                before_images.append(ch)  
            
            with rasterio.open(after_path) as src:
                ch = src.read(1)
                ch = normalize_channel(ch, mean=STATS['mean'][band], std=STATS['std'][band])
                after_images.append(ch)  

        # Stack bands into a single array
        before_image = np.stack(before_images, axis=0)  # Shape: (num_bands, H, W)
        after_image = np.stack(after_images, axis=0)    # Shape: (num_bands, H, W)

        if self.transform:
            before_image, after_image, mask = self.transform(before_image, after_image, mask)

        return torch.tensor(before_image, dtype=torch.float32), \
               torch.tensor(after_image, dtype=torch.float32), \
               torch.tensor(mask, dtype=torch.float32)

