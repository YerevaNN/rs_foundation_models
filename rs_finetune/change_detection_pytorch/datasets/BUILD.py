import os
import cv2
import json
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np

STATS = {
    'mean': {
        'B1': 1743.6986580141129,
        'B2': 1322.2082124589308,
        'B3': 1426.125276342686,
        'B4': 1519.4304951183542,
        'B5': 1458.3511756085722,
        'B6': 2189.1243839605736,
        'B7': 2612.642424348492,
        'B8': 2432.5066766189702,
        'B8A': 2845.825088205645,
        'B9': 420.1201108870968,
        'B10': 10.09842279905914,
        'B11': 2023.2420465576463,
        'B12': 1295.3533746826463,
        'vv': -8.462828636169434,
        'vh': -8.510013580322266,
        },
    'std': {
        'B1': 771.5183806369524,
        'B2': 1056.4326625501772,
        'B3': 900.3624734298407,
        'B4': 873.4760016751915,
        'B5': 988.208460744263,
        'B6': 967.7348032162768,
        'B7': 1033.7542057178632,
        'B8': 1017.1386786261996,
        'B8A': 1101.856993173505,
        'B9': 301.371729572531,
        'B10': 2.74859003425678,
        'B11': 1066.2916584258733,
        'B12': 910.9452607474732,
        'vv': 4.903507709503174,
        'vh': 3.7153587341308594,
        }
}


WAVES = {
    "B2": 0.49,
    "B3": 0.56,
    "B4": 0.665,
    "B5": 0.705,
    "B6": 0.74,
    "B7": 0.783,
    "B8": 0.842,
    "B8A": 0.865,
    "B9": 0.945,
    "B10": 1.375,
    "B11": 1.61,
    "B12": 2.19,
    'vv': 3.5,
    'vh': 4.0
}

RGB_BANDS = ['B2', 'B3', 'B4']

# def normalize_channel(img, mean, std):
#     img = (img - mean) / std

#     return img.astype(np.float32)
def normalize_channel(img, mean, std):
    min_value = mean - 3 * std
    max_value = mean + 3 * std
    img = (img - min_value) / (max_value - min_value)
    img = np.clip(img, 0, 1).astype(np.float32)
    
    img = img * (max_value - min_value) + min_value
    img = (img - mean) / std

    return img.astype(np.float32)


def random_augment(rate=0.5):
    chance = np.random.rand()
    if chance < rate:
        if chance < rate*(1/3):  # rotation
            angle = np.random.choice([1, 2, 3])
            def augment(img):
                if img.ndim == 3:
                    for idx in range(len(img)):
                        channel = img[idx]
                        channel = np.rot90(channel, angle)
                        img[idx] = channel
                else:
                    img = np.rot90(img, angle)
                return img
        elif chance < rate*(2/3):
            def augment(img):
                if img.ndim == 3:
                    for idx in range(len(img)):
                        channel = img[idx]
                        channel = np.flipud(channel)  # horizontal flip
                        img[idx] = channel
                else:
                    img = np.flipud(img)
                return img
        else:
            def augment(img):
                if img.ndim == 3:
                    for idx in range(len(img)):
                        channel = img[idx]
                        channel = np.fliplr(channel)  # vertical flip
                        img[idx] = channel
                else:
                    img = np.fliplr(img)
                return img
    else:
        def augment(img):
            return img

    return augment


class BuildingDataset(Dataset):
    def __init__(self, split_list, bands=None, mean=None, std=None, transform=None, img_size = 96, 
                 fill_zeros=False,
                 fill_mean=False, 
                 weighted_input=False,
                 weight=1,
                 band_repeat_count=0,
                 replace_rgb_with_others=False,
                 metadata_path='/nfs/h100/raid/rs/metadata_harvey', is_train=False):
        """
        Args:
            split_list (str): Path to the .txt file containing folder paths.
            bands (list): List of band indices (e.g., [1, 2, 3]).
            mean (list or np.array): Per-channel mean for normalization.
            std (list or np.array): Per-channel std for normalization.
            transform (callable, optional): Transform to apply to the data.
        """

        self.classes = ['not a flooded building', 'flooded']
        self.split = os.path.splitext(os.path.basename(split_list))[0]
        self.ignore_index = None

        with open(split_list, 'r') as f:
            self.folders = [line.strip() for line in f.readlines()]
        self.bands = bands
        self.mean = np.array(mean) if mean is not None else None
        self.std = np.array(std) if std is not None else None
        self.transform = transform

        self.img_size = img_size
        self.is_train = is_train
        self.metadata_path = metadata_path

        self.fill_zeros = fill_zeros
        self.fill_mean = fill_mean
        self.weighted_input = weighted_input
        self.weight = weight
        self.band_repeat_count = band_repeat_count
        self.replace_rgb_with_others = replace_rgb_with_others

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]

        # Load mask
        mask_path = os.path.join(folder, "buildings10m.tif")
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)  # Read the first band as the mask

        # Load image bands
        images = []
        for band in self.bands:
            folder_path = os.path.join(folder, "B")
            band_path = next((f for f in os.listdir(folder_path) if f.endswith(f"{band}.tif")), None)
            band_path = os.path.join(folder_path, band_path)
            with rasterio.open(band_path) as src:
                ch = src.read(1) # Read the first band
                # padded_ch = np.zeros((self.img_size, self.img_size), dtype=ch.dtype)
                # padded_ch[:ch.shape[0], :ch.shape[1]] = ch
                # padded_ch = normalize_channel(padded_ch, mean=STATS['mean'][band], std=STATS['std'][band])
                # images.append(padded_ch)
                ch = normalize_channel(ch, mean=STATS['mean'][band], std=STATS['std'][band])
                ch = cv2.resize(ch, (self.img_size, self.img_size), interpolation = cv2.INTER_LINEAR)
                images.append(ch)

        if self.fill_zeros:
            for _ in range(self.band_repeat_count):
                images.append(np.zeros((self.img_size, self.img_size)))

        # import pdb
        # pdb.set_trace()
        # Stack bands into a single array
        image = np.stack(images, axis=0)  # Shape: (num_bands, H, W)

        if self.weighted_input:
            image = image * self.weight

        if self.fill_mean:
            mean_val = np.expand_dims(image.mean(axis=0), axis=0)
            for _ in range(self.band_repeat_count):
                image = np.concatenate([image, mean_val], axis=0)

        # Normalize using mean and std
        # if self.mean is not None and self.std is not None:
        #     image = (image - self.mean[:, None, None]) / self.std[:, None, None]

        # Apply additional transformations
        if self.transform:
            image, mask = self.transform(image, mask)

        # padded_mask = np.zeros((self.img_size, self.img_size), dtype=mask.dtype)
        # padded_mask[:mask.shape[0], :mask.shape[1]] = mask
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation = cv2.INTER_CUBIC)
        mask = (mask >= 0.5).astype(int)

        if self.is_train:
            augment = random_augment()
            image = augment(image)
            mask = augment(mask)

        image_tensor = torch.from_numpy(image.astype(np.float32).copy())
        mask_tensor = torch.from_numpy(mask.astype(np.float32).copy())

        with open(f"{self.metadata_path}/{folder.split('/')[-1]}.json", 'r') as file:
            metadata = json.load(file)
        metadata.update({'waves': [WAVES[b] for b in self.bands if b in self.bands]})

        if self.band_repeat_count != 0:
            wave_values = metadata['waves']
            mean_val = sum(wave_values) / len(wave_values)
            metadata['waves'].extend([mean_val] * self.band_repeat_count)

        if self.replace_rgb_with_others:
            metadata.update({'waves': [WAVES[b] for b in RGB_BANDS]})

        return image_tensor, mask_tensor, folder, metadata