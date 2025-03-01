import os
import torch
import random
import rasterio
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

# random.seed(42)  

STATS = {
    'mean': {
        'B01': 1626.91600224,
        'B02': 1396.03470631,
        'B03': 1364.06118417,
        'B04': 1218.22847919,
        'B05': 1466.07290663,
        'B06': 2386.90297537,
        'B07': 2845.61256277,
        'B08': 2622.95796892,
        'B8A': 3077.48221481,
        'B09': 486.87436782,
        'B10': 63.77861008,
        'B11': 2030.64763024,
        'B12': 1179.16607221,
        'VV': -10.184408,
        'VH': -16.895273,
        },
    'std': {
        'B01': 700.17133846,
        'B02': 739.09452682,
        'B03': 735.2482388,
        'B04': 864.936695,
        'B05': 776.8803358,
        'B06': 921.36834309,
        'B07': 1084.37346097,
        'B08': 1022.63418007,
        'B8A': 1196.44255318,
        'B09': 336.61105431,        
        'B10': 143.99923282,       
        'B11': 980.87061347,
        'B12': 764.60836557,
        'VV': 4.255339,
        'VH': 5.290568,
        }
}


class Sen1Floods11(Dataset):
    def __init__(self,
                 bands,
                 img_size=224,
                 metadata_path =None,
                 root_path = '/nfs/ap/mnt/frtn/rs-multiband/sen1floods11/sen1floods11',
                 split_file_path = '/nfs/ap/mnt/frtn/rs-multiband/sen1floods11_splits_with_s2',
                 split = 'train',
                ):
        
        self.classes = ['Not Water', 'Water']
        self.ignore_index = -1
        
        self.split_file_path = split_file_path
        self.split = split
        self.root_path = root_path
        self.img_size = img_size

        self.bands = bands

        self.split_mapping = {"train": "train", 
                              "val": "valid", 
                              "test": "test"}
        
        split_file = os.path.join(
            self.split_file_path,
            f"flood_{self.split_mapping[split]}_data_with_S2.csv",
        )


        data_root = os.path.join(
            root_path, "v1.1", "data/flood_events/HandLabeled/"
        )
        
        with open(split_file) as f:
            file_list = f.readlines()

        file_list = [f.rstrip().split(",") for f in file_list]

        
        self.s1_image_list = [
            os.path.join(data_root, "S1Hand", f[0]) for f in file_list
        ]
        self.s2_image_list = [
            os.path.join(data_root, "S2Hand", f[0].replace("S1Hand", "S2Hand"))
            for f in file_list
        ]
        self.target_list = [
            os.path.join(data_root, "LabelHand", f[2]) for f in file_list
        ]

    def __len__(self):
        return len(self.s1_image_list)



    def __getitem__(self, index):
        with rasterio.open(self.s2_image_list[index]) as src:
            s2_image = src.read()

        with rasterio.open(self.s1_image_list[index]) as src:
            s1_image = src.read()
            s1_image = np.nan_to_num(s1_image)

        with rasterio.open(self.target_list[index]) as src:
            target = src.read(1)
        
        s2_image = torch.from_numpy(s2_image).float()
        s1_image = torch.from_numpy(s1_image).float()

        target = torch.from_numpy(target).long()

        band_index_map = {
            'B01': s2_image[0],
            'B02': s2_image[1],
            'B03': s2_image[2],
            'B04': s2_image[3],
            'B05': s2_image[4],
            'B06': s2_image[5],
            'B07': s2_image[6],
            'B08': s2_image[7],
            'B8A': s2_image[8],
            'B09': s2_image[9],
            'B10': s2_image[10],
            'B11': s2_image[11],
            'B12': s2_image[12],
            'VV': s1_image[0],
            'VH': s1_image[1],
        }

        img = []

        if self.split == 'train':
            i, j, h, w = transforms.RandomCrop.get_params(target, (256, 256))
            for b in self.bands:
                ch = (band_index_map[b] - STATS['mean'][b]) / STATS['std'][b]
                ch = F.crop(ch, i, j, h, w)
                ch = F.resize(ch.unsqueeze(0), [self.img_size, self.img_size],
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )
                img.append(ch.squeeze(0))
            target = F.crop(target, i, j, h, w)
        else:
            for b in self.bands:
                ch = ((band_index_map[b] - STATS['mean'][b]) / STATS['std'][b]).unsqueeze(0)
                ch = F.resize(ch, [self.img_size, self.img_size],
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )
                img.append(ch.squeeze(0))

        target = target.unsqueeze(0)
        target = F.resize(target, [self.img_size, self.img_size],
                            interpolation=transforms.InterpolationMode.NEAREST,
                            )
        target = target.squeeze(0)


        image = torch.stack(img, axis=0) 

        if self.split == 'train':
            if random.random() > 0.5:
                image = F.hflip(image)
                target = F.hflip(target)
            if random.random() > 0.5:
                image = F.vflip(image)
                target = F.vflip(target)
  
    
        return image, target, None, None