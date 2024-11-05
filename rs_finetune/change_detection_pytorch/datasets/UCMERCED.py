# --------------------------------------------------------
# Based from TorchGeo codebase
# https://github.com/microsoft/torchgeo
# --------------------------------------------------------

"""UC Merced dataset."""
import os
import torch
import torchvision.transforms as T

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import _pil_interp
from torch.utils.data import Dataset


class UCMerced(ImageFolder):
    """UC Merced dataset.

    The `UC Merced <http://weegee.vision.ucmerced.edu/datasets/landuse.html>`__
    dataset is a land use classification dataset of 2.1k 256x256 1ft resolution RGB
    images of urban locations around the U.S. extracted from the USGS National Map Urban
    Area Imagery collection with 21 land use classes (100 images per class).

    Dataset features:

    * land use class labels from around the U.S.
    * three spectral bands - RGB
    * 21 classes

    Dataset classes:

    * agricultural
    * airplane
    * baseballdiamond
    * beach
    * buildings
    * chaparral
    * denseresidential
    * forest
    * freeway
    * golfcourse
    * harbor
    * intersection
    * mediumresidential
    * mobilehomepark
    * overpass
    * parkinglot
    * river
    * runway
    * sparseresidential
    * storagetanks
    * tenniscourt

    This dataset uses the train/val/test splits defined in the "In-domain representation
    learning for remote sensing" paper:

    * https://arxiv.org/abs/1911.06721

    If you use this dataset in your research, please cite the following paper:

    * https://dl.acm.org/doi/10.1145/1869790.1869829
    """

    # base_dir = os.path.join("UCMerced_LandUse", "Images")
    classes = [
        "agricultural",
        "airplane",
        "baseballdiamond",
        "beach",
        "buildings",
        "chaparral",
        "denseresidential",
        "forest",
        "freeway",
        "golfcourse",
        "harbor",
        "intersection",
        "mediumresidential",
        "mobilehomepark",
        "overpass",
        "parkinglot",
        "river",
        "runway",
        "sparseresidential",
        "storagetanks",
        "tenniscourt",
    ]

    splits = ["train", "val", "test"]

    def __init__(self, root="data", base_dir= '', split="train", dataset_name='uc_merced', transform=None, image_size=256):
        """Initialize a new UC Merced dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transform: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        assert split in self.splits
        self.root = root
        self.transform = transform
        self.image_size = image_size
        self.base_dir = base_dir

        valid_fns = set()
        with open(os.path.join(f'{self.root}{self.base_dir}', f"{dataset_name}-{split}.txt")) as f:
            for fn in f:
                valid_fns.add(fn.strip())
        is_in_split: Callable[[str], bool] = lambda x: os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.base_dir),
            transform=transform,
            is_valid_file=is_in_split,
        )

    def __getitem__(self, index: int):
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        image, label = self._load_image(index)

        # if self.transform is not None:
        #     image = self.transform(image)

        return image, label

    def __len__(self):
        """Return the number of data points in the dataset.
        Returns:
            length of the dataset
        """
        return len(self.imgs)

    def _load_image(self, index: int):
        """Load a single image and it's class label.
        Args:
            index: index to return
        Returns:
            the image
            the image class label
        """
        img, label = ImageFolder.__getitem__(self, index)
        return img, label
    
def build_transform(split='train', mixup=False, image_size=256):
    if split=='train' and mixup:
        transforms = T.Compose([
                T.Resize((image_size, image_size), interpolation=_pil_interp('bicubic')),
                T.RandomResizedCrop(image_size, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
    elif split=='train':
        transforms = T.Compose([
                T.Resize((image_size, image_size), interpolation=_pil_interp('bicubic')),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
    else:
        transforms = T.Compose([
                T.Resize((image_size, image_size), interpolation=_pil_interp('bicubic')),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
    return transforms