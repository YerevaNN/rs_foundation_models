import torch
import utils
import lmdb

import random
import math
import numpy as np

from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image

from torch.utils.data import Dataset



# # Start a read-only transaction
# with env.begin() as txn:
#     # For example, get a specific key
#     key = b'00010000' # this is an index 
#     value = txn.get(key)
#     if value is not None:
#         # Convert the byte data back to a NumPy array.
#         # You need to know the original data type and shape.
#         # For example, assuming float32 and shape (height, width, channels):
#         print(np.frombuffer(value, dtype=np.float32))
#     else:
#         print("Key not found.")
#     # Or, iterate over all keys:
#     with txn.cursor() as cursor:
#         value = txn.get(b'00010000')
#         data = np.frombuffer(value, dtype=np.float32).reshape(320, 320, 4)
#         #reshape to ben-(120, 120, 12), intelinair-(320, 320, 4), so2sat- (32, 32, 12), sen12mms-(256, 256, 12)
#         means = np.frombuffer(txn.get(b'means'), dtype=np.float32)
#         stds = np.frombuffer(txn.get(b'stds'), dtype=np.float32)
#         bands = txn.get(b'bands').decode('utf-8')
#         decoded_bands = []
#         for b in bands.split('/')[:-1]:
#             decoded_bands.append(b.strip('\x00'))
#         print(means, stds, bands, decoded_bands)
#         # for key, value in cursor:
#         #     # Process key/value pairs as needed
#         #     data = np.frombuffer(value, dtype=np.float32)  # adjust dtype/reshape if necessary
#         #     print(f"Key: {key}, Data shape: {data.shape}")
        

class LMDBDataset(Dataset):
    def __init__(self, data_path, transform, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, data_shape=None, normalize=False, **kwargs):
        super(LMDBDataset, self).__init__()
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
        
        self.env = lmdb.open(data_path, readonly=True, lock=False)
        self.txn = self.env.begin(write=False)
        
        stats = self.txn.stat()
        self.data_len = stats['entries'] - 3
        
        if data_shape is None:
            self.data_len = self.data_len // 2
        
        self.transform = transform
        self.data_path = data_path 
        self.data_shape = data_shape
        self.normalize = normalize
        
        self.means = np.frombuffer(self.txn.get(b'means'), dtype=np.float32)
        self.stds = np.frombuffer(self.txn.get(b'stds'), dtype=np.float32)
        bands = self.txn.get(b'bands').decode('utf-8')
        self.band_names = []
        for b in bands.split('/')[:-1]:
            self.band_names.append(b.replace('\x00', ''))
        
        self.ok_indices = list(range(self.data_len))
        
        # Cache for data shapes if needed
        self.shape_cache = {}
        
    def __len__(self):
        return self.data_len
    
    def __del__(self):
        if hasattr(self, 'txn') and self.txn is not None:
            self.txn.abort()
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
    
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
        
        key = f'{index:08d}'.encode()
        orig_image = np.frombuffer(self.txn.get(key), dtype=np.float32)
        # orig_image = np.ones(self.data_shape, dtype=np.float32)
        while np.isnan(orig_image).any():
            self.ok_indices.remove(index)
            index = random.choice(self.ok_indices)
            key = f'{index:08d}'.encode()
            orig_image = np.frombuffer(self.txn.get(key), dtype=np.float32)
        
        data_shape = self.data_shape
        if data_shape is None:
            # Try to get from cache first
            if index not in self.shape_cache:
                shape_key = f'{index:08d}'.encode()
                self.shape_cache[index] = np.frombuffer(self.txn.get(shape_key), dtype=np.int64)
            data_shape = self.shape_cache[index]
            
        orig_image = np.copy(orig_image).reshape(data_shape)
            
        if self.normalize:
            orig_image = (orig_image - self.means) / self.stds
        
        if orig_image.dtype != np.float32:
            orig_image = orig_image.astype(np.float32)
        images = self.transform(orig_image)  
        masks = self.get_masks(images)
        
        return images, masks, self.band_names, self.data_path
    