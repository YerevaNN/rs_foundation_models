import utils
from PIL import Image
import random
import math
import cv2
import numpy as np
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torchvision import transforms
import torchvision.transforms.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset.utils import RandomRotate90, GaussianBlur



class HDF5Augmentation(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        self.flip_and_color_jitter = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1),
            ], p=0.5),
            # A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        ])

        self.global_crops_number = global_crops_number
        
        # transformation for the first global crop
        self.global_transfo1 = A.Compose([
            A.RandomResizedCrop((224, 224), scale=global_crops_scale, interpolation=cv2.INTER_CUBIC),
            self.flip_and_color_jitter,
            A.GaussianBlur(sigma_limit=(0.5, 2.0), p=1.0),
            ToTensorV2(),
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = A.Compose([
            A.RandomResizedCrop((224, 224), scale=global_crops_scale, interpolation=cv2.INTER_CUBIC),
            self.flip_and_color_jitter,
            A.GaussianBlur(sigma_limit=(0.5, 2.0), p=0.1),
            ToTensorV2(),
        ])
        
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = A.Compose([
            A.RandomResizedCrop((96, 96), scale=local_crops_scale, interpolation=cv2.INTER_CUBIC),
            self.flip_and_color_jitter,
            A.GaussianBlur(sigma_limit=(0.5, 2.0), p=0.5),
            ToTensorV2(),
        ])

    def __call__(self, image):
        crops = []
        assert isinstance(image, np.ndarray)
        crops.append(self.global_transfo1(image=image)["image"])
        
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image=image)["image"])
            
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image=image)["image"])
        return crops