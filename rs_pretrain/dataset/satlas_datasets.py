import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import math
import random
from collections import defaultdict
from tqdm import tqdm


class SatlasDataset(Dataset):
    def __init__(self, data_path, transform, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(SatlasDataset, self).__init__()
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch
        
        self.data_path = data_path
        self.transform = transform
        
    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def get_masks(self, images):
        masks = []
        for img in images:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False
            masks.append(mask)
        return masks
        
    def load_stats(self, stats_dir, band):
        """
        Load the stats for a given band from the stats directory.
        For multispectral bands (b05, b06, etc.), the file is assumed to be '{band}_band_stats.npy'
        For the tci folder, the file is assumed to be 'tci_band_stats.npy' and returns a dict with keys "mean" and "std"
        """
        if band =='tci':
            stats_path = os.path.join(stats_dir, f"final_rgb_stats.npy")
        else:
            stats_path = os.path.join(stats_dir, f"{band}_band_stats.npy")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
        return np.load(stats_path, allow_pickle=True).item()

    def save_error_path(self, error_path):
        save_path = os.path.join(os.path.dirname(self.data_path), f"{self.data_path.split('/')[-1]}_error_paths.txt")
        with open(save_path, 'a') as f:
            f.write(f"{error_path}\n")

class Sen1Dataset(SatlasDataset):
    def __init__(self, data_path, stats_dir, transform=None, **kwargs):
        super(Sen1Dataset, self).__init__(data_path=data_path, transform=transform, **kwargs)
        
        self.band_names = ['VV', 'VH']
        # Load precomputed statistics for VV and VH bands (assumed file names: vv_band_stats.npy and vh_band_stats.npy)
        vv_stats = np.load(os.path.join(stats_dir, "vv_band_stats.npy"), allow_pickle=True).item()
        vh_stats = np.load(os.path.join(stats_dir, "vh_band_stats.npy"), allow_pickle=True).item()

        self.vv_mean, self.vv_std = vv_stats['mean'], vv_stats['std']
        self.vh_mean, self.vh_std = vh_stats['mean'], vh_stats['std']

        subfolders = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        self.band_paths = {'VV': [], 'VH': []}
        for subfolder in tqdm(subfolders, desc="SEN1 subfolders", unit="folder",  mininterval=10):
            vv_folder = os.path.join(subfolder, "vv")
            vh_folder = os.path.join(subfolder, "vh")
            if not (os.path.isdir(vv_folder) and os.path.isdir(vh_folder)):
                continue

            for png_file in os.listdir(vv_folder):
                if not png_file.lower().endswith(".png"):
                    continue

                vv_path = os.path.join(vv_folder, png_file)
                vh_path = os.path.join(vh_folder, png_file)
                if not os.path.exists(vh_path):
                    # print(f"Missing VH file for {png_file}, skipping.")
                    continue
                
                self.band_paths['VV'].append(vv_path)
                self.band_paths['VH'].append(vh_path)

        self.ok_indices = list(range(len(self.band_paths['VV'])))
    
    def __len__(self):
        return len(self.band_paths['VV'])
    
    def __getitem__(self, index):
        if index not in self.ok_indices:
            index = np.random.choice(self.ok_indices)
            
        vv_path = self.band_paths['VV'][index]
        vh_path = self.band_paths['VH'][index]
        
        try:
            vv_img = cv2.imread(vv_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            vh_img = cv2.imread(vh_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        except Exception as e:
            self.save_error_path(vv_path)
            self.ok_indices.remove(index)
            new_index = np.random.choice(self.ok_indices)
            return self.__getitem__(new_index)
        if vv_img is None or vh_img is None:
            self.save_error_path(vv_path)
            self.ok_indices.remove(index)
            new_index = np.random.choice(self.ok_indices)
            return self.__getitem__(new_index)
        if vv_img.shape != vh_img.shape:
            self.save_error_path(vv_path)
            self.ok_indices.remove(index)
            new_index = np.random.choice(self.ok_indices)
            return self.__getitem__(new_index)

        if np.isnan(vv_img).any() or np.isnan(vh_img).any():
            self.save_error_path(vv_path)
            self.ok_indices.remove(index)
            new_index = np.random.choice(self.ok_indices)
            return self.__getitem__(new_index)

        # Normalize the images
        vv_img = (vv_img - self.vv_mean) / self.vv_std
        vh_img = (vh_img - self.vh_mean) / self.vh_std

        images = np.stack((vv_img, vh_img), axis=-1)
        images = images.astype(np.float32)
        images = self.transform(images)  
        masks = self.get_masks(images)
        
        return images, masks, self.band_names, self.data_path
    
    
class Sen2Dataset(SatlasDataset):
    def __init__(self, data_path, stats_dir, transform=None, **kwargs):
        super(Sen2Dataset, self).__init__(data_path=data_path, transform=transform, **kwargs)
        
        self.band_names = ['B', 'G', 'R', 'E1', 'E2', 'E3', 'N', 'S1', 'S2']
        
        multispectral_bands = ["b05", "b06", "b07", "b08", "b11", "b12"]
        tci_folder = "tci"
        # Load stats for each multispectral band and for tci
        stats_dict = {}
        for band in multispectral_bands:
            stats_dict[band] = self.load_stats(stats_dir, band)
        # For tci, assume stats file is "tci_band_stats.npy"
        tci_stats = self.load_stats(stats_dir, 'tci')
        stats_dict['R'] = tci_stats['R']
        stats_dict['B'] = tci_stats['B']
        stats_dict['G'] = tci_stats['G']
        
        # Create header datasets for band_names, means, and stds.
        # Define channel order: multispectral bands then tci channels.
        self.means = []
        self.stds = []
        for band in ['B', 'G', 'R']:
            self.means.append(stats_dict[band]["mean"])
            self.stds.append(stats_dict[band]["std"])
        for band in multispectral_bands:
            self.means.append(stats_dict[band]["mean"])
            self.stds.append(stats_dict[band]["std"])
        
        self.band_paths = defaultdict(list)
        sample_band = multispectral_bands[0]
        subfolders = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        for subfolder in tqdm(subfolders, desc="SEN2 subfolders", unit="folder",  mininterval=10):
            # For each subfolder, get common files based on multispectral band (e.g., b05)
            sample_folder = os.path.join(subfolder, sample_band)
            if not os.path.isdir(sample_folder):
                continue
            # Get list of PNG filenames in sample_folder
            filenames = [f for f in os.listdir(sample_folder) if f.lower().endswith(".png")]
            for fname in filenames:
                paths = {}
                valid = True
                # For each multispectral band:
                for band in multispectral_bands:
                    band_path = os.path.join(subfolder, band)
                    fpath = os.path.join(band_path, fname)
                    if not os.path.exists(fpath):
                        valid = False
                        break
                    paths[band] = fpath
                # For the tci folder:
                tci_path = os.path.join(subfolder, tci_folder)
                fpath = os.path.join(tci_path, fname)
                if not os.path.exists(fpath):
                    valid = False
                paths[tci_folder] = fpath
                if valid:
                    for key in paths:
                        self.band_paths[key].append(paths[key])

        self.ok_indices = list(range(len(self.band_paths[sample_band])))

    def __len__(self):
        return len(self.band_paths['tci'])
    
    def __getitem__(self, index):
        if index not in self.ok_indices:
            index = np.random.choice(self.ok_indices)
        
        channels = []
        try:
            tci_img = cv2.imread(self.band_paths["tci"][index])
        except Exception as e:
            self.save_error_path(self.band_paths["tci"][index])
            self.ok_indices.remove(index)
            new_index = np.random.choice(self.ok_indices)
            return self.__getitem__(new_index)
        
        if tci_img is None or tci_img.ndim < 3 or tci_img.shape[2] < 3:
            self.save_error_path(self.band_paths["tci"][index])
            self.ok_indices.remove(index)
            new_index = np.random.choice(self.ok_indices)
            return self.__getitem__(new_index)
            
        channels.append(tci_img[..., 0])
        channels.append(tci_img[..., 1])
        channels.append(tci_img[..., 2])

        for band in ["b05", "b06", "b07", "b08", "b11", "b12"]:
            try:
                img = cv2.imread(self.band_paths[band][index], cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                self.save_error_path(self.band_paths[band][index])
                self.ok_indices.remove(index)
                new_index = np.random.choice(self.ok_indices)
                return self.__getitem__(new_index)
            if img is None:
                self.save_error_path(self.band_paths[band][index])
                self.ok_indices.remove(index)
                new_index = np.random.choice(self.ok_indices)
                return self.__getitem__(new_index)
            channels.append(img)
    
        # Ensure all channels have the same shape
        shape_set = {ch.shape for ch in channels}
        if len(shape_set) != 1:
            self.save_error_path(self.band_paths["tci"][index])
            self.ok_indices.remove(index)
            new_index = np.random.choice(self.ok_indices)
            return self.__getitem__(new_index)
        combined = np.stack(channels, axis=-1)
        if np.isnan(combined).any():
            self.save_error_path(self.band_paths["tci"][index])
            self.ok_indices.remove(index)
            new_index = np.random.choice(self.ok_indices)
            return self.__getitem__(new_index)
        
        # Normalize the images
        combined = (combined - self.means) / self.stds
        images = combined.astype(np.float32)
        images = self.transform(images)  
        masks = self.get_masks(images)
        
        return images, masks, self.band_names, self.data_path


class NaipDataset(SatlasDataset):
    def __init__(self, data_path, stats_dir, transform=None, **kwargs):
        super(NaipDataset, self).__init__(data_path=data_path, transform=transform, **kwargs)
        
        self.band_names = ['B', 'G', 'R', 'N']
        
        tci_folder = "tci"
        # Load stats for each multispectral band and for tci
        stats_dict = {}
        stats_dict['ir'] = self.load_stats(stats_dir, 'ir')
        # For tci, assume stats file is "tci_band_stats.npy"
        tci_stats = self.load_stats(stats_dir, 'tci')
        stats_dict['R'] = tci_stats['R']
        stats_dict['B'] = tci_stats['B']
        stats_dict['G'] = tci_stats['G']
            
        # Create header datasets for band_names, means, and stds.
        # Define channel order: multispectral bands then tci channels.
        self.means = []
        self.stds = []
        for band in ['B', 'G', 'R']:
            self.means.append(stats_dict[band]["mean"])
            self.stds.append(stats_dict[band]["std"])
        self.means.append(stats_dict['ir']["mean"])
        self.stds.append(stats_dict['ir']["std"])
    
        self.band_paths = defaultdict(list)
        sample_band = 'ir'
        subfolders = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        for subfolder in tqdm(subfolders, desc="NAIP subfolders", unit="folder",  mininterval=10):
            # For each subfolder, get common files based on multispectral band (e.g., b05)
            sample_folder = os.path.join(subfolder, sample_band)
            if not os.path.isdir(sample_folder):
                continue
            # Get list of PNG filenames in sample_folder
            filenames = [f for f in os.listdir(sample_folder) if f.lower().endswith(".png")]
            for fname in filenames:
                paths = {}
                valid = True
                # For each multispectral band:
                band_path = os.path.join(subfolder, 'ir')
                fpath = os.path.join(band_path, fname)
                if not os.path.exists(fpath):
                    valid = False
                paths['ir'] = fpath
                # For the tci folder:
                tci_path = os.path.join(subfolder, tci_folder)
                fpath = os.path.join(tci_path, fname)
                if not os.path.exists(fpath):
                    valid = False
                paths[tci_folder] = fpath
                if valid:
                    for key in paths:
                        self.band_paths[key].append(paths[key])

        self.ok_indices = list(range(len(self.band_paths[sample_band])))

    def __len__(self):
        return len(self.band_paths['tci'])
    
    def __getitem__(self, index):
        if index not in self.ok_indices:
            index = np.random.choice(self.ok_indices)
        
        channels = []
        try:
            tci_img = cv2.imread(self.band_paths["tci"][index])
        except Exception as e:
            self.save_error_path(self.band_paths["tci"][index])
            self.ok_indices.remove(index)
            new_index = np.random.choice(self.ok_indices)
            return self.__getitem__(new_index)
        
        if tci_img is None or tci_img.ndim < 3 or tci_img.shape[2] < 3:
            self.save_error_path(self.band_paths["tci"][index])
            self.ok_indices.remove(index)
            new_index = np.random.choice(self.ok_indices)
            return self.__getitem__(new_index)
            
        channels.append(tci_img[..., 0])
        channels.append(tci_img[..., 1])
        channels.append(tci_img[..., 2])

        try:
            img = cv2.imread(self.band_paths['ir'][index], cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            self.save_error_path(self.band_paths['ir'][index])
            self.ok_indices.remove(index)
            new_index = np.random.choice(self.ok_indices)
            return self.__getitem__(new_index)
        if img is None:
            self.save_error_path(self.band_paths['ir'][index])
            self.ok_indices.remove(index)
            new_index = np.random.choice(self.ok_indices)
            return self.__getitem__(new_index)
        channels.append(img)
    
        # Ensure all channels have the same shape
        shape_set = {ch.shape for ch in channels}
        if len(shape_set) != 1:
            self.save_error_path(self.band_paths['ir'][index])
            self.ok_indices.remove(index)
            new_index = np.random.choice(self.ok_indices)
            return self.__getitem__(new_index)
        combined = np.stack(channels, axis=-1)
        if np.isnan(combined).any():
            self.save_error_path(self.band_paths['ir'][index])
            self.ok_indices.remove(index)
            new_index = np.random.choice(self.ok_indices)
            return self.__getitem__(new_index)
        
        # Normalize the images
        combined = (combined - self.means) / self.stds
        images = combined.astype(np.float32)
        images = self.transform(images)  
        masks = self.get_masks(images)
        
        return images, masks, self.band_names, self.data_path