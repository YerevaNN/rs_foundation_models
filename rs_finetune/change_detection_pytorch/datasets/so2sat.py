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
        '02 - Blue': 0.12951652705669403,
        '03 - Green': 0.11734361201524734,
        '04 - Red': 0.11374464631080627,
        '05 - Vegetation Red Edge': 0.12693354487419128,
        '06 - Vegetation Red Edge': 0.16917912662029266,
        '07 - Vegetation Red Edge': 0.19080990552902222,
        '08 - NIR': 0.18381330370903015,
        '08A - Vegetation Red Edge': 0.20517952740192413,
        '11 - SWIR': 0.1762811541557312,
        '12 - SWIR': 0.1286638230085373,
        '01 - VH.Real': 0.00030114364926703274, 
        '03 - VV.Real': 0.000289927760604769,
        '02 - VH.Imaginary': -5.6475887504348066e-06, 
        '04 - VV.Imaginary': -0.0005758664919994771 

        },
    'std' :  {
        '02 - Blue': 0.040680479258298874,
        '03 - Green': 0.05125178396701813,
        '04 - Red': 0.07254913449287415,
        '05 - Vegetation Red Edge': 0.06872648745775223,
        '06 - Vegetation Red Edge': 0.07402216643095016,
        '07 - Vegetation Red Edge': 0.08412779122591019,
        '08 - NIR': 0.08534552156925201,
        '08A - Vegetation Red Edge': 0.09248979389667511,
        '11 - SWIR': 0.10270608961582184,
        '12 - SWIR': 0.09284552931785583,
        '01 - VH.Real': 0.20626230537891388, 
        '03 - VV.Real': 0.5187134146690369,
        '02 - VH.Imaginary':  0.19834314286708832, 
        '04 - VV.Imaginary': 0.519291877746582
    },

}
QUANTILES = {
    'min_q': 
    {
        '02 - Blue': 9.999999747378752e-05, 
        '03 - Green': 9.999999747378752e-05, 
        '04 - Red': 9.999999747378752e-05,
        '05 - Vegetation Red Edge': 0.013199999928474426, 
        '06 - Vegetation Red Edge': 0.01360000018030405, 
        '07 - Vegetation Red Edge': 0.011500000022351742, 
        '08 - NIR': 9.999999747378752e-05, 
        '08A - Vegetation Red Edge': 0.00800000037997961, 
        '11 - SWIR': 0.0005000000237487257, 
        '12 - SWIR': 9.999999747378752e-05,
        '01 - VH.Real': -38.8077392578125,  
        '03 - VV.Real': -39.88066482543945,
        '02 - VH.Imaginary': -10.303879737854004, 
        '04 - VV.Imaginary': -39.512115478515625

    },
    'max_q': 
    {
        '02 - Blue': 1.3322999477386475, 
        '03 - Green': 1.4991999864578247, 
        '04 - Red': 1.894700050354004,
        '05 - Vegetation Red Edge': 1.1855000257492065, 
        '06 - Vegetation Red Edge': 1.2854000329971313, 
        '07 - Vegetation Red Edge': 1.089900016784668, 
        '08 - NIR': 1.8743000030517578,
        '08A - Vegetation Red Edge': 0.9067999720573425, 
        '11 - SWIR': 1.0729999542236328, 
        '12 - SWIR': 1.3802000284194946,
        '01 - VH.Real': 43.63044738769531, 
        '03 - VV.Real': 46.09192657470703,
        '02 - VH.Imaginary': 36.00978088378906, 
        '04 - VV.Imaginary': 31.161375045776367,
    }
}

WAVES = {
    '02 - Blue': 0.493,
    '03 - Green': 0.56,
    '04 - Red': 0.665,
    '05 - Vegetation Red Edge': 0.704,
    '06 - Vegetation Red Edge': 0.74,
    '07 - Vegetation Red Edge': 0.783,
    '08 - NIR': 0.842,
    '08A - Vegetation Red Edge': 0.865,
    '11 - SWIR': 1.61,
    '12 - SWIR': 2.19,
    'VV': 3.5,
    'VH': 4.0
}



def normalize_channel(img, mean, std):
    img = (img - mean) / std
    img = np.clip(img, -3, 3).astype(np.float32)

    return img


class So2SatDataset(Dataset):
    def __init__(self, 
                split,
                bands,
                img_size=32,
                # transform=None,
                h5_dir="/nfs/ap/mnt/frtn/rs-multiband/geobench/classification_v1.0/m-so2sat/", 
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

        so2sat_bands = {
            "B02": '02 - Blue', 
            "B03": '03 - Green', 
            "B04": '04 - Red', 
            "B05": '05 - Vegetation Red Edge', 
            "B06": '06 - Vegetation Red Edge', 
            "B07": '07 - Vegetation Red Edge', 
            "B08": '08 - NIR',
            "B8A": '08A - Vegetation Red Edge',
            "B11": '11 - SWIR',
            "B12": '12 - SWIR',
            "VV": 'VV',
            "VH": 'VH',
        }
        self.bands = [so2sat_bands[b] for b in bands]

    def __len__(self):
        return len(self.files)

    @property
    def num_classes(self):
        return 17

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
                    if 'VV' not in band_name and 'VH' not in band_name:
                        band = normalize_channel(band, STATS['mean'][band_name], STATS['std'][band_name])
                        bands.append(band)
            if 'VV' in self.bands:
                vv_i = np.array(fp['04 - VV.Imaginary'])
                vv_i = normalize_channel(vv_i, STATS['mean']['04 - VV.Imaginary'], STATS['std']['04 - VV.Imaginary'])

                vv_r = np.array(fp['03 - VV.Real'])
                vv_r = normalize_channel(vv_r, STATS['mean']['03 - VV.Real'], STATS['std']['03 - VV.Real'])

                vv_int = np.log10(vv_i ** 2 + vv_r ** 2 + 1e-10) * 10
                bands.append(vv_int)
            if 'VH' in self.bands:   
                vh_i = np.array(fp['02 - VH.Imaginary'])
                vh_i = normalize_channel(vh_i, STATS['mean']['02 - VH.Imaginary'], STATS['std']['02 - VH.Imaginary'])

                vh_r = np.array(fp['01 - VH.Real'])
                vh_r = normalize_channel(vh_r, STATS['mean']['01 - VH.Real'], STATS['std']['01 - VH.Real'])
                vh_int = np.log10(vh_i ** 2 + vh_r ** 2 + 1e-10) * 10
                bands.append(vh_int)
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





