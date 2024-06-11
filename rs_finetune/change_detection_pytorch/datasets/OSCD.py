import random

from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import functional as TF
from pytorch_lightning import LightningDataModule
from pathlib import Path
from itertools import product

from torch.utils.data import Dataset
import rasterio
import numpy as np
from PIL import Image
import random

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

BANDS_ORDER = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12']

ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
RGB_BANDS = ['B04', 'B03', 'B02']

QUANTILES = {
    'min_q': {
        'B02': 885.0,
        'B03': 667.0,
        'B04': 426.0
    },
    'max_q': {
        'B02': 2620.0,
        'B03': 2969.0,
        'B04': 3698.0
    }
}
STATS = {
    'mean': {
        'B02': 1422.4117861742477,
        'B03': 1359.4422181552754,
        'B04': 1414.6326650140888,
        'B05': 1557.91209397433,
        'B06': 1986.5225593959844,
        'B07': 2211.038518780755,
        'B08': 2119.168043369016,
        'B8A': 2345.3866026353567,
        'B11': 2133.990133983443,
        'B12': 1584.1727764661696
        },
    'std' :  {
        'B02': 456.1716680330627,
        'B03': 590.0730894364552,
        'B04': 849.3395398520846,
        'B05': 811.3614662999139,
        'B06': 813.441067258119,
        'B07': 891.792623998175,
        'B08': 901.4549041572363,
        'B8A': 954.7424298485422,
        'B11': 1116.63101989494,
        'B12': 985.2980824905794}

}

def normalize_channel(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def read_image(path, bands, normalize=True):
    patch_id = next(path.iterdir()).name[:-8]
    channels = []
    for b in bands:
        ch = rasterio.open(path / f'{patch_id}_{b}.tif').read(1)
        if normalize:
            ch = rasterio.open(path / f'{patch_id}_{b}.tif').read(1)
            ch = normalize_channel(ch, mean=STATS['mean'][b], std=STATS['std'][b])
            channels.append(ch)

            # min_v = QUANTILES['min_q'][b]
            # max_v = QUANTILES['max_q'][b]
            # ch = (ch - min_v) / (max_v - min_v)
            # ch = np.clip(ch, 0, 1)
            # ch = (ch * 255).astype(np.uint8)
    min_width, min_height = channels[0].shape
    
    for ch in channels:
        width, height = ch.shape
        min_width = width if width < min_width else min_width
        min_height = height if height < min_height else min_height
    resized_channels = []
    for ch in channels:
        res_ch = np.resize(ch, (min_width, min_height))
        resized_channels.append(res_ch)


    img = np.dstack(resized_channels)
    img = Image.fromarray(img)
    return img


class ChangeDetectionDataset(Dataset):

    def __init__(self, root, split='all', bands=None, transform=None, patch_size=96, mode='vanilla', scale=None, fill_zeros=False):
        self.root = Path(root)
        self.split = split
        self.bands = bands if bands is not None else RGB_BANDS
        self.transform = transform
        self.mode = mode
        self.scale = scale
        self.patch_size = patch_size
        self.fill_zeros = fill_zeros

        with open(self.root / f'{split}.txt') as f:
            names = f.read().strip().split(',')

        self.samples = []
        for name in names:
            fp = next((self.root / name / 'imgs_1').glob(f'*{self.bands[0]}*'))
            img = rasterio.open(fp)
            limits = product(range(0, img.width, patch_size), range(0, img.height, patch_size))
            for l in limits:
                self.samples.append((self.root / name, (l[0], l[1], l[0] + patch_size, l[1] + patch_size)))

    def __getitem__(self, index):
        path, limits = self.samples[index]
        
        if self.mode == 'vanilla':
            img_1 = read_image(path / 'imgs_1', self.bands)
            img_2 = read_image(path / 'imgs_2', self.bands)

        elif self.mode == 'wo_train_aug':
            img_1 = read_image(path / 'imgs_1', self.bands)

            if self.split == 'test':
                # im2_paths = ['imgs_2', 'imgs_2_4x', 'imgs_2_8x']
                im2_paths = [f'imgs_2_{self.scale}']

                choosed_res = random.sample(im2_paths, 1)[0]
            else:
                choosed_res = 'imgs_2'

            img_2 = read_image(path / choosed_res, self.bands)

        elif self.mode == 'w_train_aug':
            img_1 = read_image(path / 'imgs_1', self.bands)

            # im2_paths = ['imgs_2', 'imgs_2_4x', 'imgs_2_8x']
            im2_paths = ['imgs_2', 'imgs_2_2x']

            choosed_res = random.sample(im2_paths, 1)[0]
            img_2 = read_image(path / choosed_res, self.bands)

        cm = Image.open(path / 'cm' / 'cm.png').convert('L')

        img_1 = img_1.crop(limits)
        img_2 = img_2.crop(limits)
        cm = cm.crop(limits)

        img_1 = np.array(img_1)
        img_2 = np.array(img_2)
        cm = np.array(cm) / 255

        transformed_data = self.transform(image=img_1, image_2=img_2, mask=cm)
        img_1, img_2, cm = transformed_data['image'], transformed_data['image_2'], transformed_data['mask']

        filename = f'{path}_{limits}'

        if self.fill_zeros:
            new_img_1 = torch.zeros((9, img_1.shape[1], img_1.shape[2]), dtype=img_1.dtype, device=img_1.device)
            new_img_2 = torch.zeros((9, img_2.shape[1], img_2.shape[2]), dtype=img_2.dtype, device=img_2.device)
            for i in range(len(self.bands)):
                if self.bands[i] in BANDS_ORDER:
                    new_img_1[BANDS_ORDER.index(self.bands[i])] = img_1[i]
                    new_img_2[BANDS_ORDER.index(self.bands[i])] = img_2[i]
                else:
                    if self.bands[i] == 'B8A':
                        idx = BANDS_ORDER.index('B08')
                        new_img_1[idx] = img_1[i]
                        new_img_2[idx] = img_2[i]

            img_1 = new_img_1
            img_2 = new_img_2

        return img_1, img_2, cm, filename

    def __len__(self):
        return len(self.samples)

class RandomFlip:

    def __call__(self, *xs):
        if random.random() > 0.5:
            xs = tuple(TF.hflip(x) for x in xs)
        return xs


class RandomRotation:

    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, *xs):
        angle = random.choice(self.angles)
        return tuple(TF.rotate(x, angle) for x in xs)


class RandomSwap:

    def __call__(self, x1, x2, y):
        if random.random() > 0.5:
            return x2, x1, y
        else:
            return x1, x2, y


class ToTensor:

    def __call__(self, *xs):
        return tuple(TF.to_tensor(x) for x in xs)


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *xs):
        for t in self.transforms:
            xs = t(*xs)
        return xs


class ChangeDetectionDataModule(LightningDataModule):

    def __init__(self, data_dir, patch_size=96, mode='vanilla', batch_size=4, scale=None, bands=None, fill_zeros=False):
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.mode = mode
        self.batch_size = batch_size
        self.scale = scale
        self.fill_zeros = fill_zeros
        self.bands=bands
        print(scale)

    def setup(self, stage=None):
        self.train_dataset = ChangeDetectionDataset(
            self.data_dir,
            split='train',
            transform=A.Compose([
                    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.1, rotate_limit=30, p=0.6),
                    A.RandomCrop(self.patch_size, self.patch_size),
                    A.Flip(p=0.5), # either horizontally, vertically or both
                    A.Normalize(),
                    ToTensorV2()
                ], additional_targets={'image_2': 'image'}),
            patch_size=self.patch_size,
            mode = self.mode,
            scale= self.scale,
            fill_zeros=self.fill_zeros,
            bands=self.bands

        )
        self.val_dataset = ChangeDetectionDataset(
            self.data_dir,
            split='test',
            transform=A.Compose([
                    A.RandomCrop(self.patch_size, self.patch_size),
                    A.Normalize(),
                    ToTensorV2()
                ], additional_targets={'image_2': 'image'}),
            patch_size=self.patch_size,
            mode=self.mode,
            scale=self.scale,
            fill_zeros=self.fill_zeros,
            bands = self.bands

        )

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=2,
            drop_last=True,
            pin_memory=True,
            sampler=sampler
        )

    def val_dataloader(self):
        # sampler = DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            drop_last=True,
            pin_memory=True,
            shuffle=False
        )