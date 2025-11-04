import torch
import utils

import random
import math
import cv2
import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image

from dataset.utils import get_files



class MAIDDataset(Dataset):
    def __init__(self, data_path, transform=None, patch_size=224, pred_ratio=0.5, pred_ratio_var=0.1, 
                 pred_aspect_ratio=(0.3, 1/0.3), pred_shape='block', pred_start_epoch=0, **kwargs):
        super(MAIDDataset, self).__init__()
        self.band_names = ['R', 'G', 'B']
        self.data_path = data_path
        self.transform = transform
        
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
        
        self.img_paths = get_files(data_path, ('.png', '.jpg', '.jpeg'))
        self.data_len = len(self.img_paths)
        
    def __len__(self):
        return self.data_len
 
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

    def set_epoch(self, epoch):
        self.epoch = epoch
        
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
        img_path = self.img_paths[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images = self.transform(image)
        masks = self.get_masks(images)
        for i, image in enumerate(images):
            if np.isnan(image).any():
                print(f"NAN in {i} image {img_path}", force=True)
                exit()
        return images, masks, self.band_names, img_path
    

class MAIDDatasetCO(MAIDDataset):
    def __init__(self, *args, **kwargs):
        super(MAIDDatasetCO, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        image = cv2.imread(self.img_paths[index])
        images = self.transform(image)
        masks = self.get_masks(images)
        
        global1, global2 = images[:2]
        global1_i, global1_j = params[0][:2]

        global2_i = params[1][0] - global1_i
        global2_j = params[1][1] - global1_j

        # Cropped shapes
        global1_h, global1_w = params[0][-2:]
        global2_h, global2_w = params[1][-2:]

        crop_overlap_label = torch.zeros((1, global1_h, global1_w))
        overlap_ii, overlap_jj = global2_i + global2_h, global2_j + global2_w
        if overlap_ii > 0 and overlap_jj > 0:
            overlap_i = max(global2_i, 0)
            overlap_j = max(global2_j, 0)
            overlap_ii = min(overlap_ii, global1_h)
            overlap_jj = min(overlap_jj, global1_w)
            crop_overlap_label[0, overlap_i: overlap_ii, overlap_j: overlap_jj] = 1
        
        crop_overlap_label = F.resize(crop_overlap_label, global1.shape[-2:], 
                                      interpolation=transforms.InterpolationMode.NEAREST)

        return images, masks, self.band_names, crop_overlap_label, self.data_path

