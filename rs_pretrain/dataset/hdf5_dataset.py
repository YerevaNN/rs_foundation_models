import torch
import utils
import h5py

import random
import math
import numpy as np

from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image

from torch.utils.data import Dataset



class HDF5Dataset(Dataset):
    def __init__(self, data_path, transform, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, data_key='BEN', **kwargs):
        super(HDF5Dataset, self).__init__()
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch
        
        self.hdf5 = h5py.File(data_path, 'r')
        self.data = self.hdf5[data_key]
        self.data_key = data_key
        self.data_len = self.data.shape[0]
        self.means = self.hdf5['means'][:]
        self.stds = self.hdf5['stds'][:]
        self.band_names = self.hdf5['band_names'][:]
        self.band_names = [x.decode('utf-8') for x in self.band_names]
        self.transform = transform
        self.data_path = data_path
        
        self.ok_indices = list(range(self.data.shape[0]))
        # self.mask = None
        # self.images = None
        # if data_key == 'SEN12MS':
        #     self.ok_indices.remove(58620)
            
            
    def __len__(self):
        return self.data_len
    
    def __del__(self):
        self.hdf5.close()
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def get_masks(self, images):
        masks = []
        for img in images:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False
            masks.append(mask)
        return masks
    
    def __getitem__(self, index):
        if index not in self.ok_indices:
            index = random.choice(self.ok_indices)
            
        orig_image = self.data[index]
        while np.isnan(orig_image).any():
            self.ok_indices.remove(index)
            index = random.choice(self.ok_indices)
            orig_image = self.data[index]
        
        if orig_image.dtype == np.uint8:
            orig_image = (orig_image - self.means) / self.stds
        
        orig_image = orig_image.astype(np.float32)
        images = self.transform(orig_image) 
        mask = self.get_masks(images)
        
        return images, mask, self.band_names, self.data_path