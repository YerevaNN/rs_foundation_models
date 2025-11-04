import os
import torch
import json
import h5py
import pickle
import ast
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (Compose, Resize, RandomHorizontalFlip, 
                                    RandomApply, RandomChoice, RandomRotation)

STATS = {
    'mean': {
        '01 - Coastal aerosol': 1356.876219,
        '02 - Blue': 1123.692166,
        '03 - Green': 1052.923118,
        '04 - Red': 948.631638,
        '05 - Vegetation Red Edge': 1211.055683,
        '06 - Vegetation Red Edge': 2046.101992,
        '07 - Vegetation Red Edge': 2428.045141,
        '08 - NIR': 2356.445745,
        '08A - Vegetation Red Edge': 753.852449,
        '09 - Water vapour': 12.18468,
        '11 - SWIR': 1111.888325,
        '12 - SWIR': 2660.320844,
        'VH': -18.364276885986328,
        'VV': -11.433741569519043
    },
    'std': {
        '01 - Coastal aerosol': 259.4540018601834,
        '02 - Blue': 346.2998554031296,
        '03 - Green': 401.7416438274206,
        '04 - Red': 590.8571730540613,
        '05 - Vegetation Red Edge': 551.4152868504857,
        '06 - Vegetation Red Edge': 858.33214359689,
        '07 - Vegetation Red Edge': 1086.9850069551512,
        '08 - NIR': 1123.6935402601525,
        '08A - Vegetation Red Edge': 408.4046853914661,
        '09 - Water vapour': 4.681440728835516,
        '11 - SWIR': 727.2080437850603,
        '12 - SWIR': 1233.1146707403684,
        'VH': 6.097087383270264,
        'VV': 5.903086185455322
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
    'VV': 3.5,
    'VH': 4.0
}




def normalize_channel(img, mean, std):
    img = (img - mean) / std
    img = np.clip(img, -3, 3).astype(np.float32)

    return img


class mEurosat(Dataset):
    def __init__(self, 
                split,
                bands,
                img_size=64,
                # transform=None,
                h5_dir="/nfs/ap/mnt/frtn/rs-multiband/ben/classification_v1.0.0/m-eurosat/", 
                ):

        self.h5_dir = h5_dir
        self.img_size = img_size
        # self.transform = transform

        train_transforms = Compose([
            Resize(self.img_size),
            RandomHorizontalFlip(p=0.5),
            RandomApply([
                RandomChoice([
                    RandomRotation((90,  90)),
                    RandomRotation((180, 180)),
                    RandomRotation((270, 270)),
                ])
            ], p=0.5),
        ])

        test_transforms = Compose([
            Resize(self.img_size),
        ])

        if split == 'train':
            self.transform = train_transforms
        else:
            self.transform = test_transforms
        
        self.split = split
        
        with open (os.path.join(h5_dir, "default_partition.json"), 'r') as f:
            data = json.load(f)
        
        self.files = [f"{os.path.join(h5_dir, f)}.hdf5" for f in data[self.split]]

        m_eurosat_bands = {
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
            "VV": 'VV',
            "VH": 'VH',
        }
        self.bands = [m_eurosat_bands[b] for b in bands]

    def __len__(self):
        return len(self.files)

    @property
    def num_classes(self):
        return 10

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with h5py.File(file_path, 'r') as fp:
            attr_dict = pickle.loads(ast.literal_eval(fp.attrs["pickle"]))

            band_names = attr_dict.get("bands_order", fp.keys())
            bands = []
            label = None
            for band_name in band_names:
                band = np.array(fp[band_name])
                if band_name  in ['VV', 'VH']:
                    band = np.squeeze(band)
                if band_name.startswith("label"):
                    label = band
                elif band_name in self.bands:
                    band = normalize_channel(band, STATS['mean'][band_name], STATS['std'][band_name])
                    bands.append(band)
            if label is None:
                label = attr_dict["label"]

        bands = np.stack(bands, axis=-1)
        bands = torch.from_numpy(bands)  # → (H, W, C) tensor
        bands = bands.permute(2, 0, 1).float()


        if self.transform:
            bands = self.transform(bands)

        metadata = {'time': '14:00:00', 
                    'latlon': [9.144916534423828, 
                            45.47289204060055, 
                            9.20654296875, 
                            45.53304838316756], 
                    'gsd': 10, 
                    'waves': []}
        
        metadata.update({'waves': [WAVES[b] for b in self.bands if b in self.bands]})

        return bands, torch.tensor(label, dtype=torch.long), metadata

