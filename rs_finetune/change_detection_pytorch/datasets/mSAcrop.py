import os
import re
import cv2
import torch
import json
import h5py
import pickle
import ast
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (Compose, Resize, RandomHorizontalFlip, 
                                    RandomApply, RandomChoice, RandomRotation)
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


STATS = {
    'mean': {
        '01 - Coastal aerosol': 12.729473,
        '02 - Blue': 16.545663,
        '03 - Green': 26.690091,
        '04 - Red': 36.960014,
        '05 - Vegetation Red Edge': 46.640303,
        '06 - Vegetation Red Edge': 58.338267,
        '07 - Vegetation Red Edge': 63.557080,
        '08 - NIR': 68.105470,
        '08A - Vegetation Red Edge': 69.052113,
        '09 - Water vapour': 69.795670,
        '11 - SWIR': 84.243284,
        '12 - SWIR': 66.429824,
        '13 - VH': 3.962732,
        '14 - VV': 17.329660,
    },
    'std': {
        '01 - Coastal aerosol': 7.472967,
        '02 - Blue': 9.283443,
        '03 - Green': 12.526701,
        '04 - Red': 19.147744,
        '05 - Vegetation Red Edge': 19.264738,
        '06 - Vegetation Red Edge': 19.724572,
        '07 - Vegetation Red Edge': 21.278095,
        '08 - NIR': 22.744390,
        '08A - Vegetation Red Edge': 21.995149,
        '09 - Water vapour': 21.507289,
        '11 - SWIR': 27.653947,
        '12 - SWIR': 26.886302,
        '13 - VH': 5.527869,
        '14 - VV': 16.507998,
    }
}

WAVES = {
    "01 - Coastal aerosol": 0.443,
    "02 - Blue": 0.493,
    "03 - Green": 0.56,
    "04 - Red": 0.665,
    "05 - Vegetation Red Edge": 0.704,
    "06 - Vegetation Red Edge": 0.74,
    "07 - Vegetation Red Edge": 0.783,
    "08 - NIR": 0.842,
    "08A - Vegetation Red Edge": 0.865,
    "09 - Water vapour": 0.945,
    "11 - SWIR": 1.61,
    "12 - SWIR": 2.19,
    '14 - VV': 3.5,
    '13 - VH': 4.0
}




def normalize_channel(img, mean, std):
    img = (img - mean) / std
    img = np.clip(img, -3, 3).astype(np.float32)

    return img


class mSAcrop(Dataset):
    def __init__(self, 
                split,
                bands,
                img_size=256,
                # transform=None,
                fill_zeros=False,
                h5_dir="/nfs/h100/raid/rs/geobench/sa_crop_type/", 
                ):

        self.h5_dir = h5_dir
        self.img_size = img_size
        # self.transform = transform
        self.fill_zeros = fill_zeros

        self.transforms = Compose([
            # Resize(self.img_size),
            RandomHorizontalFlip(p=0.5),
            RandomApply([
                RandomChoice([
                    RandomRotation((90,  90)),
                    RandomRotation((180, 180)),
                    RandomRotation((270, 270)),
                ])
            ], p=0.5),
        ])

        
        self.split = split

        
        with open (os.path.join(h5_dir, "default_partition.json"), 'r') as f:
            data = json.load(f)
        
        self.files = [f"{os.path.join(h5_dir, f)}.hdf5" for f in data[self.split]]

        sa_srop_bands = {
            "B01": '01 - Coastal aerosol', 
            "B02": '02 - Blue', 
            "B03": '03 - Green', 
            "B04": '04 - Red', 
            "B05": '05 - Vegetation Red Edge', 
            "B06": '06 - Vegetation Red Edge', 
            "B07": '07 - Vegetation Red Edge', 
            "B08": '08 - NIR',
            "B8A": '08A - Vegetation Red Edge',
            "B09": '09 - Water vapour',
            "B11": '11 - SWIR',
            "B12": '12 - SWIR',
            "VV": '14 - VV',
            "VH": '13 - VH',
        }
        self.bands = [sa_srop_bands[b] for b in bands]

        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.ignore_index = None

    def __len__(self):
        return len(self.files)

    @property
    def num_classes(self):
        return 10

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with h5py.File(file_path, 'r') as fp:
            ds_names = list(fp.keys())

            bands = []
            label = None

            for name in ds_names:
                if name.startswith("label"):
                    label = np.array(fp[name])
                    break

            for band_prefix in self.bands:
                matches = [n for n in ds_names if n.startswith(band_prefix)]
                if not matches:
                    continue

                ds_name = matches[0]
                band = np.array(fp[ds_name])
                band = normalize_channel(band,
                                        STATS['mean'][band_prefix],
                                        STATS['std'][band_prefix])
                bands.append(band)

        bands = np.stack(bands, axis=-1)
        bands = torch.from_numpy(bands).permute(2, 0, 1).float()
        if self.fill_zeros and bands.shape[0] < 3:
            zeros = torch.zeros((1, *bands.shape[1:]), dtype=bands.dtype, device=bands.device)
            bands = torch.cat([bands, zeros], dim=0)

        img = F.resize(bands, self.img_size, interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(torch.tensor(label, dtype=torch.float32).unsqueeze(0),
                        self.img_size,
                        interpolation=InterpolationMode.NEAREST).squeeze(0)

        metadata = {
            'time': '14:00:00',
            'latlon': [9.1449165, 45.4728920, 9.2065429, 45.5330483],
            'gsd': 10,
            'waves': [WAVES[b] for b in self.bands if b in WAVES]
        }

        return img, mask, file_path, metadata
