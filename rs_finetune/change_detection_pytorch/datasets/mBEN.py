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
        '01 - Coastal aerosol': 380.40826416015625,
        '02 - Blue': 488.4757995605469,
        '03 - Green': 711.8541870117188,
        '04 - Red': 735.3125,
        '05 - Vegetation Red Edge': 1112.1353759765625,
        '06 - Vegetation Red Edge': 1900.5006103515625,
        '07 - Vegetation Red Edge': 2179.173828125,
        '08 - NIR': 2326.05517578125,
        '08A - Vegetation Red Edge': 2385.028564453125,
        '09 - Water vapour': 2361.1572265625,
        '11 - SWIR': 1897.660888671875,
        '12 - SWIR': 1246.3990478515625,
        'VH': -18.901323318481445,
        'VV': -12.309995651245117
    },
    'std': {
        '01 - Coastal aerosol': 439.2579345703125,
        '02 - Blue': 502.00390625,
        '03 - Green': 542.435302734375,
        '04 - Red': 675.141357421875,
        '05 - Vegetation Red Edge': 682.3658447265625,
        '06 - Vegetation Red Edge': 958.0399780273438,
        '07 - Vegetation Red Edge': 1115.5296630859375,
        '08 - NIR': 2326.05517578125,
        '08A - Vegetation Red Edge': 1198.8616943359375,
        '09 - Water vapour': 1145.2288818359375,
        '11 - SWIR': 1107.837646484375,
        '12 - SWIR': 870.6172485351562,
        'VH': 5.091253280639648,
        'VV': 4.67769718170166
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


class mBigearthnet(Dataset):
    def __init__(self, 
                split,
                bands,
                img_size=120,
                # transform=None,
                h5_dir="/nfs/ap/mnt/frtn/rs-multiband/ben/classification_v1.0.0/m-bigearthnet/", 
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

        m_ben_bands = {
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
        self.bands = [m_ben_bands[b] for b in bands]

    def __len__(self):
        return len(self.files)

    @property
    def num_classes(self):
        return 43

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

