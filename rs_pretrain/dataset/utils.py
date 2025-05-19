import utils
from PIL import Image
import random
import math
import os
import numpy as np
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torchvision import transforms
import torchvision.transforms.functional as F

from utils import FastRandomResizedCrop


def get_files(path, extensions):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(extensions):
                matches.append(os.path.join(root, filename))
    return matches

class PILRandomRotate90:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            k = random.choice([1, 3])
            angle = 90 * k
            img = img.rotate(angle, expand=True)
        return img
    
class RandomRotate90:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            k = random.choice([1, 3])
            tensor = torch.rot90(tensor, k, dims=(-2, -1))
        return tensor
    
class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur
    """
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=p)
        

class Solarization(transforms.RandomApply):
    """
    Apply Solarization
    """
    def __init__(self, p: float = 0.5, threshold: float = 0.5):
        transform = transforms.RandomSolarize(threshold=threshold)
        super().__init__(transforms=[transform], p=p)
