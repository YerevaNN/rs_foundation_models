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
        '01 - Coastal aerosol': 574.7587880700896,
        '02 - Blue': 674.3473615470523,
        '03 - Green': 886.3656479311578,
        '04 - Red': 815.0945462528913,
        '05 - Vegetation Red Edge': 1128.8088426870465,
        '06 - Vegetation Red Edge': 1934.450471876027,
        '07 - Vegetation Red Edge': 2045.7652282437202,
        '08 - NIR': 2012.744587807115,
        '08A - Vegetation Red Edge': 1608.6255233989034,
        '09 - Water vapour': 1129.8171906000355,
        '10 - SWIR - Cirrus': 83.27188605598549,
        '11 - SWIR': 90.54924599052214,
        '12 - SWIR': 68.98768652434848,
        '13 - VH': -16.562910079956055,
        '14 - VV': -11.571099281311035
    },
    'std': {
        '01 - Coastal aerosol': 193.60631504991184,
        '02 - Blue': 238.75447480113132,
        '03 - Green': 276.9631260242207,
        '04 - Red': 361.15060137326634,
        '05 - Vegetation Red Edge': 364.5888078793488,
        '06 - Vegetation Red Edge': 724.2707123576525,
        '07 - Vegetation Red Edge': 819.653063972575,
        '08 - NIR': 794.3652427593881,
        '08A - Vegetation Red Edge': 800.8538290702304,
        '09 - Water vapour': 704.0219637458916,
        '10 - SWIR - Cirrus': 36.355745901131705,
        '11 - SWIR': 28.004671947623894,
        '12 - SWIR': 24.268892726362033,
        '13 - VH': 7.251087188720703,
        '14 - VV': 5.895864009857178
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


class BrickKiln(Dataset):
    def __init__(self, 
                split,
                bands,
                img_size=64,
                # transform=None,
                h5_dir="/nfs/h100/raid/rs/geobench/brick-kiln/", 
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

        m_bands = {
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
        self.bands = [m_bands[b] for b in bands]

    def __len__(self):
        return len(self.files)

    @property
    def num_classes(self):
        return 2

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with h5py.File(file_path, 'r') as fp:
            attr_dict = pickle.loads(ast.literal_eval(fp.attrs["pickle"]))
            band_names = list(fp.keys())
            bands = []
            label = None
            for band_name in band_names:
                band = np.array(fp[band_name])
                # if band_name  in ['VV', 'VH']:
                #     band = np.squeeze(band)
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

