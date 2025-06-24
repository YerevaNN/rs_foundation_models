import torch
import numpy as np
import random
import os


# SATLAS_BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12']

def get_band_indices(band_names):
    cvit_bands = ['R', 'G', 'B', 'E1', 'E2', 'E3', 'N', "N'", 'S1', 'S2', 'VV', 'VH']
    band_mapping = {
        'B02': 'B', 'B03': 'G', 'B04': 'R',
        'B05': 'E1', 'B06': 'E2', 'B07': 'E3',
        'B08': 'N', 'B8A': "N'",
        'B11': 'S1', 'B12': 'S2',
        'VV': 'VV', 'VH': 'VH'
    }
    indices = [cvit_bands.index(band_mapping[band]) for band in band_names]
    return indices


def get_band_indices_cvit_so2sat(band_names):
    all_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "VV", "VV", "VH", "VH"]
    indices = [all_bands.index(band) for band in band_names]
    return indices

BAND_ORDER = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', "VH", "VV"]
RGB_BAND_ORDER = ['B02', 'B03', 'B04']

BAND_ORDER_MAPPING = {
    'cvit-pretrained': ['B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', 'VV', 'VH'],  
    'cvit': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', 'VH', 'VH', 'VV', 'VV'],
    'croma': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12',  'VH', 'VV'],
    'satlas': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12'],
    'prithvi': ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']   
}

RGB_ORDER_MAPPING = {
    'cvit-pretrained': ['B04', 'B03', 'B02']
}

def get_band_orders(model_name, rgb=False):
    if rgb:
        return RGB_ORDER_MAPPING.get(model_name, RGB_BAND_ORDER)
    
    return BAND_ORDER_MAPPING.get(model_name, BAND_ORDER)

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_collate_fn(task_type='classification'):
    if task_type == 'classification':
        def collate_fn(batch):
            images, labels, metadata_list = zip(*batch)
            images = torch.stack(images)
            labels = torch.tensor(np.array(labels))
            metadata = list(metadata_list)
            return images, labels, metadata
    elif task_type == 'change_detection':
        def collate_fn(batch):
            images1, images2, labels, filename, metadata_list = zip(*batch)
            images1 = torch.stack(images1)
            images2 = torch.stack(images2)
            labels = torch.tensor(np.array(labels))
            metadata = list(metadata_list)
            return images1, images2, labels, filename, metadata
    elif task_type == 'segmentation':
        def collate_fn(batch):
            images, labels, filename, metadata_list = zip(*batch)
            images = torch.stack(images)
            labels = torch.tensor(np.array(labels))
            metadata = list(metadata_list)
            return images, labels, filename, metadata
    
    return collate_fn