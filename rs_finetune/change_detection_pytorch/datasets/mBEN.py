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
        '01 - Coastal aerosol': 386.651611328125,
        '02 - Blue': 488.9927062988281,
        '03 - Green': 714.6070556640625,
        '04 - Red': 738.2603149414062,
        '05 - Vegetation Red Edge': 1114.421875,
        '06 - Vegetation Red Edge': 1910.189697265625,
        '07 - Vegetation Red Edge': 2191.482421875,
        '08 - NIR': 2334.2919921875,
        '08A - Vegetation Red Edge': 2392.91357421875,
        '09 - Water vapour': 2367.287353515625,
        '11 - SWIR': 1902.6917724609375,
        '12 - SWIR': 1261.06982421875,
        '01 - VH.Real': -19.29836,
        '04 - VV.Imaginary': -12.623948
    },
    'std': {
        '01 - Coastal aerosol': 467.3123474121094,
        '02 - Blue': 510.78656005859375,
        '03 - Green': 551.7964477539062,
        '04 - Red': 691.781494140625,
        '05 - Vegetation Red Edge': 700.4515991210938,
        '06 - Vegetation Red Edge': 976.7468872070312,
        '07 - Vegetation Red Edge': 1134.886474609375,
        '08 - NIR': 1238.0712890625,
        '08A - Vegetation Red Edge': 1215.9765625,
        '09 - Water vapour': 1153.8582763671875,
        '11 - SWIR': 1117.0093994140625,
        '12 - SWIR': 894.739013671875,
        '01 - VH.Real': 5.4643545,
        '04 - VV.Imaginary': 5.1194134
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
    '04 - VV.Imaginary': 3.5,
    '01 - VH.Real': 4.0
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
                h5_dir="/nfs/ap/mnt/frtn/rs-multiband/m_ben/", 
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
        
        with open (os.path.join(h5_dir, "original_partition.json"), 'r') as f:
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
            "VV": '04 - VV.Imaginary',
            "VH": '01 - VH.Real',
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

