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
        '01 - Coastal aerosol': 520.1185302734375,
        '02 - Blue': 634.7583618164062,
        '03 - Green': 892.461181640625,
        '04 - Red': 880.7075805664062,
        '05 - Vegetation Red Edge': 1380.6409912109375,
        '06 - Vegetation Red Edge': 2233.432373046875,
        '07 - Vegetation Red Edge': 2549.379638671875,
        '08 - NIR': 2326.05517578125,
        '08A - Vegetation Red Edge': 2643.531982421875,
        '09 - Water vapour': 2852.87451171875,
        '11 - SWIR': 2463.933349609375,
        '12 - SWIR': 1600.9207763671875,
        '13 - VH.Real': -9.948413837479514,
        '14 - VV.Real': -16.446916880713598
    },
    'std': {
        '01 - Coastal aerosol': 204.2023468017578,
        '02 - Blue': 227.25344848632812,
        '03 - Green': 222.32545471191406,
        '04 - Red': 350.47235107421875,
        '05 - Vegetation Red Edge': 280.6436767578125,
        '06 - Vegetation Red Edge': 373.7521057128906,
        '07 - Vegetation Red Edge': 449.9236145019531,
        '08 - NIR': 414.6498107910156,
        '08A - Vegetation Red Edge': 415.1019592285156,
        '09 - Water vapour': 413.8980407714844,
        '11 - SWIR': 494.97430419921875,
        '12 - SWIR': 514.4229736328125,
        '13 - VH.Real': 2.284861894947329,
        '14 - VV.Real': 2.7308642231972375
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
    '14 - VV.Real': 3.5,
    '13 - VH.Real': 4.0
}




def normalize_channel(img, mean, std):
    img = (img - mean) / std
    img = np.clip(img, -3, 3).astype(np.float32)

    return img


class mCashewPlantation(Dataset):
    def __init__(self, 
                split,
                bands,
                img_size=256,
                # transform=None,
                fill_zeros=False,
                h5_dir="/nfs/h100/raid/rs/geobench/cashew_benin/", 
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

        caashew_bands = {
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
            "VV": '14 - VV.Real',
            "VH": '13 - VH.Real',
        }
        self.bands = [caashew_bands[b] for b in bands]

        self.classes = ['0', '1', '2', '3', '4', '5', '6']
        self.ignore_index = None

    def __len__(self):
        return len(self.files)

    @property
    def num_classes(self):
        return 7

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with h5py.File(file_path, 'r') as fp:
            attr_dict = pickle.loads(ast.literal_eval(fp.attrs["pickle"]))
            # print(attr_dict)
            # band_names = [b.rsplit('_', 1)[0] for b in list(attr_dict.keys())]
            band_names, dates = zip(*[
                b.rsplit('_', 1) if '_' in b and re.match(r'.+_\d{4}-\d{2}-\d{2}$', b) 
                else (b, None)
                for b in list(attr_dict.keys())
            ])

            bands = []
            label = None
            # print(len(band_names), len(self.bands))
            # print("bands:", band_names)
            # print("self.bands:", self.bands)
            for band_name in band_names:
                if 'Cloud' in band_name or 'bands_order' in band_name:
                    continue
                if band_name.startswith("label"):
                    label = np.array(fp[band_name])
                elif band_name in self.bands:
                    band = np.array(fp[f"{band_name}_{dates[0]}"])
                    # print(band_name, band.shape)
                    band = normalize_channel(band, STATS['mean'][band_name], STATS['std'][band_name])
                    bands.append(band)

        bands = np.stack(bands, axis=-1)
        bands = torch.from_numpy(bands)  # → (H, W, C) tensor
        bands = bands.permute(2, 0, 1).float()
        # print("label shape:", label.shape)
        # print("label:", np.unique(label))
        if self.fill_zeros and len(self.bands) < 3:
            zeros = torch.zeros((1, bands.shape[1], bands.shape[2]), dtype=bands.dtype, device=bands.device)
            # cat along the channel dim
            bands = torch.cat([bands, zeros], dim=0)


        img = F.resize(bands, self.img_size, interpolation=InterpolationMode.BILINEAR)

        mask = F.resize(torch.tensor(label, dtype=torch.float32).unsqueeze(0), self.img_size,
                        interpolation=InterpolationMode.NEAREST).squeeze(0)

        # if self.split == 'train':
        #     img = self.transforms(img)
        #     mask = self.transforms(mask)

        metadata = {'time': '14:00:00', 
                    'latlon': [9.144916534423828, 
                            45.47289204060055, 
                            9.20654296875, 
                            45.53304838316756], 
                    'gsd': 10, 
                    'waves': []}
        
        metadata.update({'waves': [WAVES[b] for b in self.bands if b in self.bands]})

        return img, mask, file_path, metadata

