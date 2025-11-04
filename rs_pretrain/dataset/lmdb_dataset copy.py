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
        self.txn = self.env.begin(write=False)  # Persistent read-only transaction
        
        stats = self.txn.stat()
        self.data_len = stats['entries'] - 3
        
        if data_shape is None:
            self.data_len = self.data_len // 2
        
        self.transform = transform
        self.data_path = data_path 
        self.data_shape = data_shape
        self.normalize = normalize
        
        # Pre-load dataset statistics
        self.means = np.frombuffer(self.txn.get(b'means'), dtype=np.float32)
        self.stds = np.frombuffer(self.txn.get(b'stds'), dtype=np.float32)
        bands = self.txn.get(b'bands').decode('utf-8')
        self.band_names = []
        for b in bands.split('/')[:-1]:
            self.band_names.append(b.replace('\x00', ''))
        
        self.ok_indices = list(range(self.data_len))
        # if data_key == 'SEN12MS':
        #     self.ok_indices.remove(58620)
            
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

    def __getitem__(self, index):
        # Use persistent transaction instead of creating a new one each time
        key = f'{index:08d}'.encode()
        orig_image = np.frombuffer(self.txn.get(key), dtype=np.float32)
        
        data_shape = self.data_shape
        if data_shape is None:
            # Try to get from cache first
            if index not in self.shape_cache:
                shape_key = f'{index:08d}'.encode()
                self.shape_cache[index] = np.frombuffer(self.txn.get(shape_key), dtype=np.int64)
            data_shape = self.shape_cache[index]
        
        orig_image = orig_image.reshape(data_shape)
            
        if self.normalize:
            # Use in-place operations where possible
            orig_image = (orig_image - self.means) / self.stds
        
        # Convert only if necessary
        if orig_image.dtype != np.float32:
            orig_image = orig_image.astype(np.float32)
            
        images = self.transform(orig_image)  
        
        return images, self.band_names, self.data_path
    