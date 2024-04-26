import random

from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from pytorch_lightning import LightningDataModule
from pathlib import Path
from itertools import product

from torch.utils.data import Dataset
import rasterio
import numpy as np
from PIL import Image
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2

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


def read_image(path, bands, normalize=True):
    patch_id = next(path.iterdir()).name[:-8]
    channels = []
    for b in bands:
        ch = rasterio.open(path / f'{patch_id}_{b}.tif').read(1)
        if normalize:
            min_v = QUANTILES['min_q'][b]
            max_v = QUANTILES['max_q'][b]
            ch = (ch - min_v) / (max_v - min_v)
            ch = np.clip(ch, 0, 1)
            ch = (ch * 255).astype(np.uint8)
        channels.append(ch)
    img = np.dstack(channels)
    img = Image.fromarray(img)
    return img


class ChangeDetectionDataset(Dataset):

    def __init__(self, root, split='all', bands=None, transform=None, patch_size=96, mode='vanilla', scale=None):
        self.root = Path(root)
        self.split = split
        self.bands = bands if bands is not None else RGB_BANDS
        self.transform = transform
        self.mode = mode
        self.scale = scale
        self.patch_size = patch_size

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

        # img_1 = img_1.crop(limits)
        # img_2 = img_2.crop(limits)
        # cm = cm.crop(limits)
        img_1 = np.array(img_1)
        img_2 = np.array(img_2)
        cm = np.array(cm) / 255
        transformed_data = self.transform(image=img_1, image_2=img_2, mask=cm)
        img_1, img_2, cm = transformed_data['image'], transformed_data['image_2'], transformed_data['mask']
        #img_1, img_2, cm = self.transform(img_1, img_2, cm)

        filename = f'{path}_{limits}'

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

    def __init__(self, data_dir, patch_size=96, mode='vanilla', batch_size=4, scale=None):
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.mode = mode
        self.batch_size = batch_size
        self.scale = scale
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
            scale= self.scale
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
            scale=self.scale
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            pin_memory=True
        )