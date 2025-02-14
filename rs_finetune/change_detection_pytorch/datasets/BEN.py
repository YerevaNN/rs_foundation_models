import torch
import os

import csv
import json
import rasterio
import numpy as np

from pathlib import Path
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_and_extract_archive, download_url
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class Subset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def random_subset(dataset, frac, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.choice(range(len(dataset)), int(frac * len(dataset)))
    return Subset(dataset, indices)

class _RepeatSampler(object):
    """
    Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class InfiniteDataLoader(DataLoader):
    """
    Dataloader that reuses workers.
    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
RGB_BANDS = ['B02', 'B03', 'B04']

BANDS_ORDER = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'VH', 'VH', 'VV', 'VV']

BAND_STATS = {
    'mean': {
        'B01': 340.76769064,
        'B02': 429.9430203,
        'B03': 614.21682446,
        'B04': 590.23569706,
        'B05': 950.68368468,
        'B06': 1792.46290469,
        'B07': 2075.46795189,
        'B08': 2218.94553375,
        'B8A': 2266.46036911,
        'B09': 2246.0605464,
        'B11': 1594.42694882,
        'B12': 1009.32729131,
        'VH': -19.29836, 
        'VV': -12.623948

    },
    'std': {
        'B01': 554.81258967,
        'B02': 572.41639287,
        'B03': 582.87945694,
        'B04': 675.88746967,
        'B05': 729.89827633,
        'B06': 1096.01480586,
        'B07': 1273.45393088,
        'B08': 1365.45589904,
        'B8A': 1356.13789355,
        'B09': 1302.3292881,
        'B11': 1079.19066363,
        'B12': 818.86747235,
        'VH': 5.4643545,
        'VV':  5.1194134
    }
}
QUANTILES = {
    'min_q': 
    {
        'B01': 1.0, 'B02': 50.0, 'B03': 61.0, 'B04': 27.0,
        'B05': 16.0, 'B06': 1.0, 'B07': 5.0, 'B08': 1.0, 
        'B8A': 1.0, 'B09': 1.0, 'B11': 4.0, 'B12': 6.0
    },
    'max_q': 
    {
        'B01': 2050.0, 'B02': 1862.0, 'B03': 2001.0, 'B04': 2332.0,
        'B05': 2656.0, 'B06': 4039.0, 'B07': 4830.0, 'B08': 5119.0,
        'B8A': 5092.0, 'B09': 4881.0, 'B11': 4035.0, 'B12': 3139.0
    }
    }
LABELS = [
    'Agro-forestry areas', 'Airports',
    'Annual crops associated with permanent crops', 'Bare rock',
    'Beaches, dunes, sands', 'Broad-leaved forest', 'Burnt areas',
    'Coastal lagoons', 'Complex cultivation patterns', 'Coniferous forest',
    'Construction sites', 'Continuous urban fabric',
    'Discontinuous urban fabric', 'Dump sites', 'Estuaries',
    'Fruit trees and berry plantations', 'Green urban areas',
    'Industrial or commercial units', 'Inland marshes', 'Intertidal flats',
    'Land principally occupied by agriculture, with significant areas of '
    'natural vegetation', 'Mineral extraction sites', 'Mixed forest',
    'Moors and heathland', 'Natural grassland', 'Non-irrigated arable land',
    'Olive groves', 'Pastures', 'Peatbogs', 'Permanently irrigated land',
    'Port areas', 'Rice fields', 'Road and rail networks and associated land',
    'Salines', 'Salt marshes', 'Sclerophyllous vegetation', 'Sea and ocean',
    'Sparsely vegetated areas', 'Sport and leisure facilities',
    'Transitional woodland/shrub', 'Vineyards', 'Water bodies', 'Water courses'
]

NEW_LABELS = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]

GROUP_LABELS = {
    'Continuous urban fabric': 'Urban fabric',
    'Discontinuous urban fabric': 'Urban fabric',
    'Non-irrigated arable land': 'Arable land',
    'Permanently irrigated land': 'Arable land',
    'Rice fields': 'Arable land',
    'Vineyards': 'Permanent crops',
    'Fruit trees and berry plantations': 'Permanent crops',
    'Olive groves': 'Permanent crops',
    'Annual crops associated with permanent crops': 'Permanent crops',
    'Natural grassland': 'Natural grassland and sparsely vegetated areas',
    'Sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas',
    'Moors and heathland': 'Moors, heathland and sclerophyllous vegetation',
    'Sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation',
    'Inland marshes': 'Inland wetlands',
    'Peatbogs': 'Inland wetlands',
    'Salt marshes': 'Coastal wetlands',
    'Salines': 'Coastal wetlands',
    'Water bodies': 'Inland waters',
    'Water courses': 'Inland waters',
    'Coastal lagoons': 'Marine waters',
    'Estuaries': 'Marine waters',
    'Sea and ocean': 'Marine waters'
}

WAVES = {
    "B02": 0.493,
    "B03": 0.56,
    "B04": 0.665,
    "B05": 0.704,
    "B06": 0.74,
    "B07": 0.783,
    "B08": 0.842,
    "B8A": 0.865,
    "B11": 1.61,
    "B12": 2.19,
    'VV': 3.5,
    'VH': 4.0
}

def normalize(img, min_q, max_q):
    img = (img - min_q) / (max_q - min_q)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def normalize_stats(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class Bigearthnet(Dataset):
    url = 'http://bigearth.net/downloads/BigEarthNet-v1.0.tar.gz'
    subdir = 'BigEarthNet-v1.0'
    list_file = {
        'train': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-train.txt',
        'val': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-val.txt',
        'test': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-test.txt'
    }
    bad_patches = [
        'http://bigearth.net/static/documents/patches_with_seasonal_snow.csv',
        'http://bigearth.net/static/documents/patches_with_cloud_and_shadow.csv'
    ]

    def __init__(self, 
                root, 
                split, 
                splits_dir, 
                bands=None, 
                transform=None, 
                target_transform=None, 
                download=False, 
                use_new_labels=True, 
                fill_zeros=False, 
                img_size=128, 
                weighted_input=False,
                band_mean_repeat_count=0,
                weight = 11/3,
                replace_rgb_with_others=False):
        self.root = Path(root)
        self.split = split
        self.bands = bands if bands is not None else RGB_BANDS
        self.transform = transform
        self.target_transform = target_transform
        self.use_new_labels = use_new_labels
        self.splits_dir = Path(splits_dir)
        self.weighted_input = weighted_input
        self.weight = weight
        self.band_mean_repeat_count = band_mean_repeat_count 
        self.fill_zeros = fill_zeros

        self.img_size = img_size

        self.replace_rgb_with_others = replace_rgb_with_others

        if download:
            download_and_extract_archive(self.url, self.root)
            download_url(self.list_file[self.split], self.root, f'{self.split}.txt')
            for url in self.bad_patches:
                download_url(url, self.root)

        bad_patches = set()
        for url in self.bad_patches:
            filename = Path(url).name
            with open(self.splits_dir / filename) as f:
                bad_patches.update(f.read().splitlines())

        self.samples = []
        with open(self.splits_dir / f'{self.split}.csv') as train:
            reader = csv.reader(train, delimiter=",", quotechar='"')
            for row in reader:
                patch_id = row[0]
                if patch_id not in bad_patches:
                    self.samples.append(self.root / self.subdir / patch_id)

        self.s1_samples = np.load(f'{self.root}/s2_s1_mapping_train.npy', allow_pickle=True).item()
        self.s1_samples.update(np.load(f'{self.root}/s2_s1_mapping_val.npy', allow_pickle=True).item())
        self.s1_samples.update(np.load(f'{self.root}/s2_s1_mapping_test.npy', allow_pickle=True).item())
        # with open(self.root / f'{self.split}.txt') as f:
        #     for patch_id in f.read().splitlines():
        #         if patch_id not in bad_patches:
        #             self.samples.append(self.root / self.subdir / patch_id)

    def __getitem__(self, index):
        path = self.samples[index]
        patch_id = path.name

        channels = []
        if self.fill_zeros:
            for  b in BANDS_ORDER:
                if b in self.bands:
                    if b == 'VH' or b == 'VV':
                        parts = self.s1_samples[patch_id].split(os.sep)
                        full_part = parts[-2]  
                        folder_part = "_".join(full_part.split('_')[:-3])
                        path_s1 = Path("BigEarthNet_v2/BigEarthNet-S1") / folder_part / full_part
                        fp = next((self.root.parent / path_s1).glob(f'*{b}.tif'))
                        ch = rasterio.open(fp).read(1)
                        ch = normalize_stats(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])

                    else:
                        ch = rasterio.open(path / f'{patch_id}_{b}.tif').read(1)
                        # ch = normalize(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])
                        ch = normalize(ch, min_q=QUANTILES['min_q'][b], max_q=QUANTILES['max_q'][b])
                    channels.append(transforms.functional.resize(torch.from_numpy(ch).unsqueeze(0), self.img_size, 
                                        interpolation=transforms.InterpolationMode.BILINEAR, antialias=True))
                        
                else:
                    if b == 'B08':
                        if 'B8A' in self.bands:
                            ch = rasterio.open(path / f'{patch_id}_B8A.tif').read(1)
                            # ch = normalize(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])
                            ch = normalize(ch, min_q=QUANTILES['min_q'][b], max_q=QUANTILES['max_q'][b])
                            channels.append(transforms.functional.resize(torch.from_numpy(ch).unsqueeze(0), self.img_size, 
                                        interpolation=transforms.InterpolationMode.BILINEAR, antialias=True))
                        else:
                            ch = torch.zeros(self.img_size, self.img_size).unsqueeze(0)
                            channels.append(ch)
                    else:
                        ch = torch.zeros(self.img_size, self.img_size).unsqueeze(0)
                        channels.append(ch)

        else:
            for b in self.bands:
                if b == 'VH' or b == 'VV':
                    parts = self.s1_samples[patch_id].split(os.sep)
                    full_part = parts[-2]  
                    folder_part = "_".join(full_part.split('_')[:-3])
                    path_s1 = Path("BigEarthNet_v2/BigEarthNet-S1") / folder_part / full_part
                    fp = next((self.root.parent / path_s1).glob(f'*{b}.tif'))
                    ch = rasterio.open(fp).read(1)
                    ch = normalize_stats(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])
                else:
                    ch = rasterio.open(path / f'{patch_id}_{b}.tif').read(1)
                    ch = normalize(ch, min_q=QUANTILES['min_q'][b], max_q=QUANTILES['max_q'][b])
                    # ch = normalize(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])

                channels.append(transforms.functional.resize(torch.from_numpy(ch).unsqueeze(0), self.img_size, 
                                    interpolation=transforms.InterpolationMode.BILINEAR, antialias=True))
                
                # channels.append(ch)
        # img = np.dstack(channels)
        # img = Image.fromarray(img)
        img = torch.cat(channels, dim=0)

        if self.weighted_input:
            img = img * self.weight
        
        mean_val = img.float().mean(dim=0).to(torch.uint8)
        mean_channels = mean_val.repeat(self.band_mean_repeat_count, 1, 1)
        img = torch.cat([img, mean_channels], dim=0)


        with open(path / f'{patch_id}_labels_metadata.json', 'r') as f:
            labels = json.load(f)['labels']
        if self.use_new_labels:
            target = self.get_multihot_new(labels)
        else:
            target = self.get_multihot_old(labels)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        with open(f'/nfs/h100/raid/rs/metadata_ben_clay/{patch_id}.json', 'r') as f:
            metadata = json.load(f)
        metadata.update({'waves': [WAVES[b] for b in self.bands if b in self.bands]})

        if self.replace_rgb_with_others:
            metadata.update({'waves': [WAVES[b] for b in RGB_BANDS]})
            
        return (img, target, metadata)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_multihot_old(labels):
        target = np.zeros((len(LABELS),), dtype=np.float32)
        for label in labels:
            target[LABELS.index(label)] = 1
        return target

    @staticmethod
    def get_multihot_new(labels):
        target = np.zeros((len(NEW_LABELS),), dtype=np.float32)
        for label in labels:
            if label in GROUP_LABELS:
                target[NEW_LABELS.index(GROUP_LABELS[label])] = 1
            elif label not in set(NEW_LABELS):
                continue
            else:
                target[NEW_LABELS.index(label)] = 1
        return target

def scale_tensor(sample):
        """Transform a single sample from the Dataset."""
        sample = sample.float().div(255)
        return sample

def custom_collate_fn(batch):
    # Separate images, labels, and metadata
    images, labels, metadata_list = zip(*batch)
    # Combine images and labels as usual (PyTorch does this automatically)
    images = torch.stack(images) 
    labels = torch.tensor(np.array(labels))
    # Keep metadata as a list of dictionaries without combining
    metadata = list(metadata_list)

    return images, labels, metadata


class BigearthnetDataModule(LightningDataModule):

    def __init__(self, 
                data_dir, 
                splits_dir, 
                bands=None, 
                train_frac=None, 
                val_frac=None,
                batch_size=32, 
                num_workers=16, 
                seed=42, 
                fill_zeros=False,
                fill_mean=False, 
                img_size=128,
                weighted_input=False,
                weight= 11/3,
                band_mean_repeat_count=0,
                replace_rgb_with_others=False):
        super().__init__()
        self.data_dir = data_dir
        self.bands = bands
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.splits_dir = splits_dir
        self.weighted_input = weighted_input
        self.band_mean_repeat_count = band_mean_repeat_count
        self.weight = weight


        self.fill_zeros=fill_zeros

        self.train_dataset = None
        self.val_dataset = None

        self.img_size = img_size

        self.replace_rgb_with_others = replace_rgb_with_others

    @property
    def num_classes(self):
        return 19

    def setup(self, stage=None):
        train_transforms = self.train_transform()
        self.train_dataset = Bigearthnet(
            root=self.data_dir,
            split='train',
            bands=self.bands,
            transform=train_transforms,
            splits_dir = self.splits_dir,
            fill_zeros=self.fill_zeros,
            img_size=self.img_size,
            weighted_input=self.weighted_input,
            weight = self.weight,
            band_mean_repeat_count=self.band_mean_repeat_count,
            replace_rgb_with_others=self.replace_rgb_with_others,
        )
        if self.train_frac is not None and self.train_frac < 1:
            self.train_dataset = random_subset(self.train_dataset, self.train_frac, self.seed)

        val_transforms = self.val_transform()
        self.val_dataset = Bigearthnet(
            root=self.data_dir,
            split='val',
            bands=self.bands,
            transform=val_transforms,
            splits_dir = self.splits_dir,
            fill_zeros=self.fill_zeros,
            img_size=self.img_size,
            weighted_input=self.weighted_input,
            weight = self.weight,
            band_mean_repeat_count=self.band_mean_repeat_count,
            replace_rgb_with_others=self.replace_rgb_with_others,
        )
        self.test_dataset = Bigearthnet(
            root=self.data_dir,
            split='test',
            bands=self.bands,
            transform=val_transforms,
            splits_dir = self.splits_dir,
            fill_zeros=self.fill_zeros,
            img_size=self.img_size,
            weighted_input=self.weighted_input,
            weight = self.weight,
            band_mean_repeat_count=self.band_mean_repeat_count,
            replace_rgb_with_others=self.replace_rgb_with_others,
        )

        if self.val_frac is not None and self.val_frac < 1:
            self.val_dataset = random_subset(self.val_dataset, self.val_frac, self.seed)
        
    @staticmethod
    def train_transform():
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            scale_tensor
            # transforms.Resize((128, 128), interpolation=Image.BICUBIC),
            # transforms.ToTensor()
        ])

    @staticmethod
    def val_transform():
        return transforms.Compose([
            scale_tensor,
            # transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            # transforms.Resize((128, 128), interpolation=Image.BICUBIC),
            # transforms.ToTensor()
        ])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        )

