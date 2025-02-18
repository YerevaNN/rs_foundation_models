import torch
import numpy as np
import random
import os


SATLAS_BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12']

def get_band_indices(band_names):
    cvit_bands = ['R', 'G', 'B', 'E1', 'E2', 'E3', 'N', "N'", 'S1', 'S2', 'SW', 'VV', 'VH']
    band_mapping = {
        'B02': 'B', 'B03': 'G', 'B04': 'R',
        'B05': 'E1', 'B06': 'E2', 'B07': 'E3',
        'B08': 'N', 'B8A': "N'",
        'B11': 'S1', 'B12': 'S2',
        'VV': 'VV', 'VH': 'VH'
    }
    indices = [cvit_bands.index(band_mapping[band]) for band in band_names]
    return indices

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True