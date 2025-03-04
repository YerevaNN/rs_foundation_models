import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import glob

BAND_STATS = {
    'mean': {
        'B01': 1353.72696296,
        'B02': 1117.20222222,
        'B03': 1041.8842963,
        'B04': 946.554,
        'B05': 1199.18896296,
        'B06': 2003.00696296,
        'B07': 2374.00874074,
        'B08': 2301.22014815,
        'B8A': 2599.78311111,
        'B09': 732.18207407,
        'B10': 12.09952894,
        'B11': 1820.69659259,
        'B12': 1118.20259259
    },
    'std': {
        'B01': 897.27143653,
        'B02': 736.01759721,
        'B03': 684.77615743,
        'B04': 620.02902871,
        'B05': 791.86263829,
        'B06': 1341.28018273,
        'B07': 1595.39989386,
        'B08': 1545.52915718,
        'B8A': 1750.12066835,
        'B09': 475.11595216,
        'B10': 98.26600935,
        'B11': 1216.48651476,
        'B12': 736.6981037
    }
}
BAND_STATS_S1 = {
    'mean': {
        'VV': -12.59,
        'VH': -20.26
    },
    'std': {
        'VV': 5.26,
        'VH': 5.91
    }
}

MS_CHANNELS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
MS_CHANNEL_INDEX = {band: idx for idx, band in enumerate(MS_CHANNELS)}
SAR_CHANNELS = ["VV", "VH"]
SAR_CHANNEL_INDEX = {band: idx for idx, band in enumerate(SAR_CHANNELS)}

class EuroSATCombinedDataset(Dataset):
    def __init__(self, ms_dir, sar_dir, bands, split_path, split='train', transform=None):
        """
        ms_dir: Base directory for multispectral TIFFs.
        sar_dir: Base directory for SAR TIFFs.
        bands: List of bands to extract (e.g., ["B02", "B03", "B05", "VV", "VH"]).
        split_file: Path to the split file (e.g., "train.txt", "val.txt", or "test.txt").
        transform: Optional transform to apply.
        """
        self.ms_dir = ms_dir
        self.sar_dir = sar_dir
        self.bands = [b.strip().upper() for b in bands]  # Normalize band names
        self.transform = transform

        split_file = os.path.join(split_path, f'eurosat-{split}.txt')
        # Load valid filenames from split file
        self.valid_filenames = set()
        with open(split_file, "r") as f:
            for line in f:
                self.valid_filenames.add(line.strip().replace('.jpg', '.tif'))

        # Find all subfolders (class labels)
        self.subfolders = sorted([f for f in os.listdir(ms_dir) if os.path.isdir(os.path.join(ms_dir, f))])
        self.label_map = {subfolder: idx for idx, subfolder in enumerate(self.subfolders)}

        # Create list of file pairs (matching MS and SAR TIFFs based on split)
        self.file_pairs = []
        for subfolder in self.subfolders:
            ms_files = sorted(glob.glob(os.path.join(ms_dir, subfolder, "*.tif")))
            sar_files = sorted(glob.glob(os.path.join(sar_dir, subfolder, "*.tif")))

            # Filter valid filenames
            ms_files = [f for f in ms_files if os.path.basename(f) in self.valid_filenames]
            sar_files = [f for f in sar_files if os.path.basename(f) in self.valid_filenames]

            # Ensure matching MS and SAR files
            ms_filenames = {os.path.basename(f) for f in ms_files}
            sar_filenames = {os.path.basename(f) for f in sar_files}
            common_files = ms_filenames.intersection(sar_filenames)

            for filename in common_files:
                ms_path = os.path.join(ms_dir, subfolder, filename)
                sar_path = os.path.join(sar_dir, subfolder, filename)
                self.file_pairs.append((ms_path, sar_path, self.label_map[subfolder]))  # Use label index

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        ms_path, sar_path, label = self.file_pairs[idx]  # Extract label (subfolder name)

        with rasterio.open(ms_path) as src:
            ms_data = src.read()  # shape: (num_ms_channels, H, W)
        ms_img = np.transpose(ms_data, (1, 2, 0))  # (H, W, num_ms_channels)

        with rasterio.open(sar_path) as src:
            sar_data = src.read()  # shape: (num_sar_channels, H, W)
        sar_img = np.transpose(sar_data, (1, 2, 0))  # (H, W, num_sar_channels)

        H, W, _ = ms_img.shape

        # Select and normalize channels
        norm_channels = []
        for band in self.bands:
            if band.startswith("B"):  # Multispectral band
                if band not in MS_CHANNEL_INDEX:
                    print(f"Multispectral band {band} not recognized. Skipping.")
                    continue
                channel_idx = MS_CHANNEL_INDEX[band]
                channel = ms_img[:, :, channel_idx]
                mean = BAND_STATS['mean'][band]
                std = BAND_STATS['std'][band]
                norm = (channel - mean) / std
                norm_channels.append(norm)
            elif band in ["VV", "VH"]:  # SAR band
                channel_idx = SAR_CHANNEL_INDEX[band]
                channel = sar_img[:, :, channel_idx]
                mean = BAND_STATS_S1['mean'][band]
                std = BAND_STATS_S1['std'][band]
                norm = (channel - mean) / std
                norm_channels.append(norm)
            else:
                print(f"Band {band} is not recognized. Skipping.")

        if not norm_channels:
            raise ValueError("No valid bands were processed.")

        # Stack channels to form a tensor of shape [H, W, C]
        combined = np.stack(norm_channels, axis=-1)
        tensor = torch.from_numpy(combined).float().permute(2, 0, 1)  # (C, H, W)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label  # Return (image, label)