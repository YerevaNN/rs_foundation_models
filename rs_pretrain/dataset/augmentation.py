import utils
from PIL import Image
import random
import math
import numpy as np
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torchvision import transforms
import torchvision.transforms.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from utils import FastRandomResizedCrop
from dataset.utils import PILRandomRotate90, RandomRotate90, GaussianBlur, Solarization


        
def pad_to_size(img, desired_size):
    """
    Pad the given PIL Image on all sides to the specified size.

    Args:
    img (PIL.Image): Image to be padded.
    desired_size (tuple): The desired output size as (width, height).

    Returns:
    PIL.Image: Padded image.
    """
    # Calculate padding
    delta_width = max(desired_size[0] - img.width, 0)
    delta_height = max(desired_size[1] - img.height, 0)
    padding = [
        delta_width // 2,
        delta_height // 2,
        delta_width - (delta_width // 2),
        delta_height - (delta_height // 2)
    ]

    # Apply padding
    padded_img = F.pad(img, padding, fill=0, padding_mode='constant')
    return padded_img


def resize_if_needed(image, min_size=224):
    """Resize an image if necessary, maintaining aspect ratio and ensuring minimum dimensions."""

    # Get original dimensions
    original_height, original_width = image.size  # Assumes PIL image

    # Check if resize is needed
    if min(original_height, original_width) < min_size:
        aspect_ratio = original_width / original_height

        # Determine new dimensions based on the minimum size
        if original_height < original_width:  # Resize based on height
            new_height = min_size
            new_width = int(new_height * aspect_ratio)
        else:  # Resize based on width
            new_width = min_size
            new_height = int(new_width / aspect_ratio)

        resize_transform = transforms.Resize((new_height, new_width))
        return resize_transform(image)
    else:
        return image  # No need to resize


class RandomCropWCoords(torch.nn.Module):

    @staticmethod
    def get_params(img, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:

        _, h, w = F.get_dimensions(img)
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = (size, size)

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), (i, j, h, w)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"


class RandomResizedCropWCoords(FastRandomResizedCrop):
     def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation,
                                                  antialias=self.antialias), (i, j, h, w)
    

class RandomResizedCropFixedHW(FastRandomResizedCrop):
    def get_params(self, img: Tensor, h, w) -> Tuple[int, int, int, int]:
        _, height, width = F.get_dimensions(img)

        if 0 < w <= width and 0 < h <= height:
            i = int(self.rand() * (height - h + 1))
            j = int(self.rand() * (width - w + 1))
            return i, j, h, w

        # Fallback to central crop
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img, height, width):
        i, j, h, w = self.get_params(img, height, width)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation,
                                                  antialias=self.antialias), (i, j, h, w)


class MAIDAugmentation(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        # Define normalization values
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        self.flip_and_color_jitter = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1),
            ], p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        ])
        
        self.global_crops_number = global_crops_number
        
        # Global crop transform 1
        self.global_transfo1 = A.Compose([
            A.RandomResizedCrop((224, 224), scale=global_crops_scale, interpolation=cv2.INTER_CUBIC),
            self.flip_and_color_jitter,
            A.GaussianBlur(sigma_limit=(0.5, 2.0), p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ])
        
        # Global crop transform 2
        self.global_transfo2 = A.Compose([
            A.RandomResizedCrop((224, 224), scale=global_crops_scale, interpolation=cv2.INTER_CUBIC),
            self.flip_and_color_jitter,
            A.GaussianBlur(sigma_limit=(0.5, 2.0), p=0.1),
            A.Solarize(p=0.2),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = A.Compose([
            A.RandomResizedCrop((96, 96), scale=local_crops_scale, interpolation=cv2.INTER_CUBIC),
            self.flip_and_color_jitter,
            A.GaussianBlur(sigma_limit=(0.5, 2.0), p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ])

    def __call__(self, image):
        assert isinstance(image, np.ndarray)
        
        crops = []
        crops.append(self.global_transfo1(image=image)["image"])
        
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image=image)["image"])

        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image=image)["image"])
            
        return crops


class MAIDAugmentationCO(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        # Define normalization values
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
        # Keep the flip_and_color_jitter name as requested
        self.flip_and_color_jitter = A.Compose([
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        ])
        
        self.global_crops_number = global_crops_number
        
        # Keep RandomResizedCropWCoords and RandomResizedCropFixedHW as they are
        # since they track coordinates which is not directly available in albumentations
        self.global_crop = RandomResizedCropWCoords(224, scale=global_crops_scale, interpolation=Image.BICUBIC)
        self.global_crop_fixed_hw = RandomResizedCropFixedHW(224, interpolation=Image.BICUBIC)
        
        # Replace transforms for post-crop processing
        self.global_transfo1 = A.Compose([
            self.flip_and_color_jitter,
            A.GaussianBlur(sigma_limit=(0.5, 2.0), p=1.0),
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255),
            ToTensorV2(),
        ])
        
        self.global_transfo2 = A.Compose([
            self.flip_and_color_jitter,
            A.GaussianBlur(sigma_limit=(0.5, 2.0), p=0.1),
            A.Solarize(p=0.2),
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255),
            ToTensorV2(),
        ])
        
        # Replace local crop transforms
        self.local_crops_number = local_crops_number
        self.local_transfo = A.Compose([
            A.RandomResizedCrop((96, 96), scale=local_crops_scale, interpolation=cv2.INTER_CUBIC),
            self.flip_and_color_jitter,
            A.GaussianBlur(sigma_limit=(0.5, 2.0), p=0.5),
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255),
            ToTensorV2(),
        ])
        
        # Flipper to maintain compatibility with original code
        self.flipper = A.HorizontalFlip(p=0.5)

    def __call__(self, image):
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            # Apply flipper to numpy array
            image_np = self.flipper(image=image_np)["image"]
            # Convert back to PIL for crops that still use PIL
            image = Image.fromarray(image_np)
        else:
            # If already numpy, just apply flipper
            image = self.flipper(image=image)["image"]

        global_crop0_img, global_crop0_params = self.global_crop(image)
        h, w = global_crop0_params[-2:]
        
        # Convert PIL crop to numpy for albumentation transforms
        global_crop0_img_np = np.array(global_crop0_img)
        crops = [self.global_transfo1(image=global_crop0_img_np)["image"]]
        
        params = [global_crop0_params]
        
        for _ in range(self.global_crops_number - 1):
            global_crop_img, global_crop_params = self.global_crop_fixed_hw(image, h, w)
            global_crop_img_np = np.array(global_crop_img)
            crops.append(self.global_transfo2(image=global_crop_img_np)["image"])
            params.append(global_crop_params)
            
        for _ in range(self.local_crops_number):
            # For local crops, apply directly to the original image
            crops.append(self.local_transfo(image=np.array(image))["image"])
            
        return crops, params