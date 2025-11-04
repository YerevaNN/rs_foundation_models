import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.neighbors import KNeighborsClassifier
from classifier_utils import load_encoder
from utils import get_band_indices

BAND_STATS_MS = {
    'mean': {
        'B02': 1465.2816,
        'B03': 1230.4503,
        'B04': 1141.8767,
        'B05': 1144.5559,
        'B06': 1356.3506,
        'B07': 1941.1139,
        'B08': 2220.791,
        'B8A': 2163.9197,
        'B11': 2418.9978,
        'B12': 792.9662
    },
    'std': {
        'B02': 752.7074,
        'B03': 747.9919,
        'B04': 746.67334,
        'B05': 967.35925,
        'B06': 953.7203,
        'B07': 990.367,
        'B08': 1086.71,
        'B8A': 1061.9794,
        'B11': 1140.0596,
        'B12': 584.30853,
    }
}

BAND_STATS_SAR = {
    'mean': {
        'VV': -11.781458,
        'VH': -18.734514
    },
    'std': {
        'VV': 4.6192575,
        'VH': 5.1911554
    }
}

# Define full multispectral and SAR channel orders
MS_CHANNELS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
MS_CHANNEL_INDEX = {band: idx for idx, band in enumerate(MS_CHANNELS)}
SAR_CHANNELS = ["VV", "VH"]
SAR_CHANNEL_INDEX = {band: idx for idx, band in enumerate(SAR_CHANNELS)}

class CombinedSeasonsDataset(Dataset):
    def __init__(self, root_dir, split_list_file, label_file, bands, transform=None):
        """
        Parameters:
          root_dir: Root directory containing season folders (e.g. ROIs2017_winter, ROIs1970_fall, etc.).
          split_list_file: Path to train_list.txt or test_list.txt (filenames without extension).
          label_file: Path to labels_train.txt or labels_test.txt; each line: "<filename> <label>".
          bands: Comma-separated list of bands (e.g., "B02,B03,B05,VV,VH") to extract.
          transform: Optional transformation to apply.
        """
        self.root_dir = root_dir
        self.transform = transform
        
       
        self.bands = bands
        # Load valid filenames from split list (filenames without extension)
        self.valid_filenames = set()
        with open(split_list_file, "r") as f:
            for line in f:
                self.valid_filenames.add(line.strip())
        
        # Load labels mapping: filename -> label (integer)
        self.label_map = {}
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) >= 2:
                    fname, lbl = parts[0], int(parts[1])
                    self.label_map[fname] = lbl
        # Now, iterate over all season folders under root_dir.
        # In each season folder, we assume there are subfolders with names starting with "s1_" for SAR and "s2_" for MS.
        self.samples = []  # List of tuples: (ms_file, sar_file, label)
        season_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and "ROI"in d]
        
        for season_folder in season_folders:
            # In each season folder, find subfolders for s1 and s2.
            s1_subfolders = sorted(glob.glob(os.path.join(season_folder, "s1*")))
            # Create a mapping from common identifier (e.g., numeric suffix) to s1 and s2 folders.
            # We assume subfolder names are like "s1_100" and "s2_100".
            for s1_folder in s1_subfolders:
                s2_folder = s1_folder.replace('s1_', 's2_')
                s1_files = glob.glob(os.path.join(s1_folder, "*.tif"))
                for s1_file in s1_files:
                    s2_file = os.path.join(s2_folder, os.path.basename(s1_file.replace('_s1_', '_s2_')))
                    if not os.path.exists(s2_file):
                        continue

                    base = s2_file.split('/')[-1]
                    # Only consider if filename is in valid list and label exists.
                    if base not in self.valid_filenames or base not in self.label_map:
                        continue
                    # Build corresponding s2 file path (assume same filename exists)
                    # Append sample tuple. Label is taken from label_map.

                    self.samples.append((s2_file, s1_file, self.label_map[base]))
                    # Note: here we assume s2_file (MS) holds multispectral data and s1_file (SAR) holds SAR data.
                    # Adjust if your naming convention is reversed.

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ms_file, sar_file, label = self.samples[idx]
        # Load multispectral image (one TIFF with multiple MS bands)
        with rasterio.open(ms_file) as src:
            ms_data = src.read()  # shape: (num_ms_channels, H, W)
        ms_img = np.transpose(ms_data, (1, 2, 0))  # shape: (H, W, num_ms_channels)
        
        # Load SAR image (one TIFF with SAR channels)
        with rasterio.open(sar_file) as src:
            sar_data = src.read()  # shape: (num_sar_channels, H, W)
        sar_img = np.transpose(sar_data, (1, 2, 0))  # shape: (H, W, num_sar_channels)
        
        # Extract and normalize each requested band.
        norm_channels = []
        for band in self.bands:
            if band.startswith("B"):
                # Multispectral: assume ms_img channels are in order defined by MS_CHANNELS.
                if band not in MS_CHANNEL_INDEX:
                    print(f"MS band {band} not recognized. Skipping.")
                    continue
                idx_band = MS_CHANNEL_INDEX[band]
                channel = ms_img[:, :, idx_band]
                mean = BAND_STATS_MS['mean'][band]
                std = BAND_STATS_MS['std'][band]
                norm = (channel - mean) / std
                norm_channels.append(norm)
            elif band in ["VV", "VH"]:
                # SAR: assume sar_img channels are in order defined by SAR_CHANNELS.
                idx_band = SAR_CHANNEL_INDEX[band]
                channel = sar_img[:, :, idx_band]
                mean = BAND_STATS_SAR['mean'][band]
                std = BAND_STATS_SAR['std'][band]
                norm = (channel - mean) / std
                norm_channels.append(norm)
            else:
                print(f"Band {band} not recognized. Skipping.")
        
        if not norm_channels:
            raise ValueError("No valid bands processed for sample.")
        
        # Stack channels along a new axis: shape becomes (C, H, W)
        combined = np.stack(norm_channels, axis=0)
        tensor = torch.from_numpy(combined).float()
        if self.transform:
            tensor = self.transform(tensor)
            
        return tensor, label

def main(args):
    # Command-line arguments or hard-coded paths.
    root_dir = args.root_dir  # Contains folders like ROIs2017_winter, etc.
    train_list = args.train_list   # Filenames (without extension) for training
    test_list = args.test_list
    feature_extractor = load_encoder(args.extractor_name, args.extractor_weights)
    feature_extractor.eval()
    feature_extractor.to(device=args.device)
     # Parse requested bands.
    if isinstance(args.train_bands, str):
        train_bands = [b.strip().upper() for b in args.train_bands.split(",")]
        test_bands = [b.strip().upper() for b in args.test_bands.split(",")]

    else:
        train_bands = [b.upper() for b in args.train_bands]
        test_bands = [b.upper() for b in args.test_bands]

    # For demonstration, here we'll create a dataset for training and one for testing.
    # Since our dataset class already scans all seasons, we can pass the same root_dir.
    train_dataset = CombinedSeasonsDataset(
        root_dir=root_dir,
        split_list_file=train_list,
        label_file=args.labels,
        bands=train_bands,
        transform=None
    )
    test_dataset = CombinedSeasonsDataset(
        root_dir=root_dir,
        split_list_file=test_list,
        label_file=args.labels,
        bands=test_bands,
        transform=None
    )
    print(len(train_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    features_train = []
    labels_train = []
    print("Extracting training features...")
    for imgs, labels in tqdm(train_loader):
        # imgs: [batch_size, C, H, W]
        feats = []
        lbls = []
        for i in range(len(imgs)):  # process one image at a time
            img = imgs[i].unsqueeze(0).to(args.device)
            lbls.append(labels[i].item())
            with torch.no_grad():
                feat = feature_extractor(img, channel_idxs=get_band_indices(train_bands))
            feats.append(feat.squeeze(0).cpu())
        feats = torch.stack(feats, dim=0)
        features_train.append(feats)
        labels_train.append(labels)
    features_train = torch.cat(features_train, dim=0).numpy()
    labels_train = torch.cat(labels_train, dim=0).numpy()
    print(features_train.shape, labels_train.shape)
    features_test = []
    labels_test = []
    print("Extracting test features...")
    for imgs, labels in tqdm(test_loader):
        feats = []
        lbls = []
        for i in range(len(imgs)):
            lbls.append(labels[i].item())
            img = imgs[i].unsqueeze(0).to(args.device)
            with torch.no_grad():
                feat = feature_extractor(img, channel_idxs=get_band_indices(test_bands))
            feats.append(feat.squeeze(0).cpu())
        feats = torch.stack(feats, dim=0)
        features_test.append(feats)
        labels_test.append(labels)
    features_test = torch.cat(features_test, dim=0).numpy()
    labels_test = torch.cat(labels_test, dim=0).numpy()

    # labels_train = np.random.randint(1, 11, size=(162555,), dtype=np.int32)
    # features_train = np.random.rand(162555, 768).astype(np.float32)
    # labels_test = np.random.randint(1, 11, size=(18106,), dtype=np.int32)
    # features_test = np.random.rand(18106, 768).astype(np.float32)

    # Train 1-NN classifier
    print("Training 1-NN classifier...")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(features_train, labels_train)

    # Evaluate
    acc = knn.score(features_test, labels_test)
    acc_save_path = f"{args.extractor_weights}_knn_accuracy_{args.type}.npy"
    np.save(acc_save_path, np.array(acc))
    print(f"Test accuracy (1-NN): {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature extraction and 1-NN classification on combined MS and SAR data.")
    parser.add_argument("--root_dir", type=str, required=True, help="Base directory for multispectral data (contains season folders).")
    parser.add_argument("--train_list", type=str, required=True, help="Path to train_list.txt (filenames without extension).")
    parser.add_argument("--test_list", type=str, required=True, help="Path to test_list.txt (filenames without extension).")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels_train.txt (format: <filename> <label>).")
    parser.add_argument("--train_bands", type=str, default="B02,B03,B04",
                        help="Comma-separated list of bands to use (e.g., 'B02,B03,B05,VV,VH').")
    parser.add_argument("--test_bands", type=str, default="B02,B03,B04",
                        help="Comma-separated list of bands to use (e.g., 'B02,B03,B05,VV,VH').")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of DataLoader workers.")
    parser.add_argument("--extractor_name", type=str, default='cvit-pretrained', help="")
    parser.add_argument("--extractor_weights", type=str, default='', help="")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--type", type=str, default='rgb_rgb')


    args = parser.parse_args()

    main(args)