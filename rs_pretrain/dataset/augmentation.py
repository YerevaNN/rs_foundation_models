import torch
from torch import Tensor
import utils

import random
from PIL import Image
import math
import numpy as np
from typing import List, Optional, Tuple
from torchvision import transforms
import torchvision.transforms.functional as F

from utils import FastRandomResizedCrop



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


class DataAugmentationiBOTCO(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        self.flipper = transforms.RandomHorizontalFlip(p=0.5)
        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_crop = RandomResizedCropWCoords(224, scale=global_crops_scale, interpolation=Image.BICUBIC)
        self.global_crop_fixed_hw = RandomResizedCropFixedHW(224, interpolation=Image.BICUBIC)
        self.global_transfo1 = transforms.Compose([
            color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            # transforms.RandomCrop(96, pad_if_needed=True),
            color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        image = self.flipper(image)
        image = resize_if_needed(image, min_size=224)

        global_crop0_img, global_crop0_params = self.global_crop(image)
        h, w = global_crop0_params[-2:]
        crops = [self.global_transfo1(global_crop0_img)]
        params = [global_crop0_params]
        for _ in range(self.global_crops_number - 1):
            global_crop_img, global_crop_params = self.global_crop_fixed_hw(image, h, w)
            crops.append(self.global_transfo2(global_crop_img))
            params.append(global_crop_params)
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops, params
