import os
import torch
import random
import rasterio
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

# random.seed(42)  
from tqdm import tqdm
import numpy as np

def calculate_class_distributions(dataset):
    num_classes = dataset.num_classes
    ignore_index = dataset.ignore_index
    class_distributions = []

    for idx in tqdm(range(len(dataset)), desc="Calculating class distributions per sample"):
        target = dataset[idx][1]

        if ignore_index is not None:
            target=target[(target != ignore_index)]

        total_pixels = target.numel()
        if total_pixels == 0:
            class_distributions.append([0] * num_classes)
            continue
        else:
            class_counts = [(target == i).sum().item() for i in range(num_classes)]
            class_ratios = [count / total_pixels for count in class_counts]
            class_distributions.append(class_ratios)

    return np.array(class_distributions)


# Function to bin class distributions using ceil
def bin_class_distributions(class_distributions, num_bins=3, logger=None):
    # logger.info(f"Class distributions are being binned into {num_bins} categories using ceil")
    
    bin_edges = np.linspace(0, 1, num_bins + 1)[1]
    binned_distributions = np.ceil(class_distributions / bin_edges).astype(int) - 1
    return binned_distributions


def balance_seg_indices(
        dataset, 
        strategy, 
        label_fraction=1.0, 
        num_bins=3, 
        logger=None):
    """
    Balances and selects indices from a segmentation dataset based on the specified strategy.

    Args:
    dataset : GeoFMDataset | GeoFMSubset
        The dataset from which to select indices, typically containing geospatial segmentation data.
    
    strategy : str
        The strategy to use for selecting indices. Options include:
        - "stratified": Proportionally selects indices from each class bin based on the class distribution.
        - "oversampled": Prioritizes and selects indices from bins with lower class representation.
    
    label_fraction : float, optional, default=1.0
        The fraction of labels (indices) to select from each class or bin. Values should be between 0 and 1.
    
    num_bins : int, optional, default=3
        The number of bins to divide the class distributions into, used for stratification or oversampling.
    
    logger : object, optional
        A logger object for tracking progress or logging messages (e.g., `logging.Logger`)

    ------
    
    Returns:
    selected_idx : list of int
        The indices of the selected samples based on the strategy and label fraction.

    other_idx : list of int
        The remaining indices that were not selected.

    """
    # Calculate class distributions with progress tracking
    class_distributions = calculate_class_distributions(dataset)

    # Bin the class distributions
    binned_distributions = bin_class_distributions(class_distributions, num_bins=num_bins, logger=logger)
    combined_bins = np.apply_along_axis(lambda row: ''.join(map(str, row)), axis=1, arr=binned_distributions)

    indices_per_bin = {}
    for idx, bin_id in enumerate(combined_bins):
        if bin_id not in indices_per_bin:
            indices_per_bin[bin_id] = []
        indices_per_bin[bin_id].append(idx)

    if strategy == "stratified":
        # Select a proportion of indices from each bin   
        selected_idx = []
        for bin_id, indices in indices_per_bin.items():
            num_to_select = int(max(1, len(indices) * label_fraction))  # Ensure at least one index is selected
            selected_idx.extend(np.random.choice(indices, num_to_select, replace=False))
    elif strategy == "oversampled":
        # Prioritize the bins with the lowest values
        sorted_indices = np.argsort(combined_bins)
        selected_idx = sorted_indices[:int(len(dataset) * label_fraction)]

    # Determine the remaining indices not selected
    other_idx = list(set(range(len(dataset))) - set(selected_idx))

    return selected_idx, other_idx


STATS = {
    'mean': {
        'B01': 1626.91600224,
        'B02': 1396.03470631,
        'B03': 1364.06118417,
        'B04': 1218.22847919,
        'B05': 1466.07290663,
        'B06': 2386.90297537,
        'B07': 2845.61256277,
        'B08': 2622.95796892,
        'B8A': 3077.48221481,
        'B09': 486.87436782,
        'B10': 63.77861008,
        'B11': 2030.64763024,
        'B12': 1179.16607221,
        'VV': -10.184408,
        'VH': -16.895273,
        },
    'std': {
        'B01': 700.17133846,
        'B02': 739.09452682,
        'B03': 735.2482388,
        'B04': 864.936695,
        'B05': 776.8803358,
        'B06': 921.36834309,
        'B07': 1084.37346097,
        'B08': 1022.63418007,
        'B8A': 1196.44255318,
        'B09': 336.61105431,        
        'B10': 143.99923282,       
        'B11': 980.87061347,
        'B12': 764.60836557,
        'VV': 4.255339,
        'VH': 5.290568,
        }
}


WAVES = {
    "B02": 0.493,
    "B03": 0.56,
    "B04": 0.665,
    "B05": 0.704,
    "B06": 0.74,
    "B07": 0.783,
    "B08": 0.842,
    "B8A": 0.865,
    "B11": 1.61,
    "B12": 2.19,
    'VV': 3.5,
    'VH': 4.0
}


class Sen1Floods11(Dataset):
    def __init__(self,
                 bands,
                 img_size=224,
                 metadata_path =None,
                 root_path = '/nfs/ap/mnt/frtn/rs-multiband/sen1floods11/sen1floods11',
                 split_file_path = '/nfs/ap/mnt/frtn/rs-multiband/sen1floods11_splits_with_s2',
                 split = 'train',
                 limited_label=1.0,
                 limited_label_strategy='stratified',
                ):
        
        self.classes = ['Not Water', 'Water']
        self.ignore_index = -1
        
        self.split_file_path = split_file_path
        self.split = split
        self.root_path = root_path
        self.img_size = img_size
        self.metadata_path = metadata_path
        self.bands = bands
        self.num_classes = 2

        self.split_mapping = {"train": "train", 
                              "val": "valid", 
                              "test": "test"}
        
        split_file = os.path.join(
            self.split_file_path,
            f"flood_{self.split_mapping[split]}_data_with_S2.csv",
        )


        data_root = os.path.join(
            root_path, "v1.1", "data/flood_events/HandLabeled/"
        )
        
        with open(split_file) as f:
            file_list = f.readlines()

        file_list = [f.rstrip().split(",") for f in file_list]

        
        self.s1_image_list = [
            os.path.join(data_root, "S1Hand", f[0]) for f in file_list
        ]
        self.s2_image_list = [
            os.path.join(data_root, "S2Hand", f[0].replace("S1Hand", "S2Hand"))
            for f in file_list
        ]
        self.target_list = [
            os.path.join(data_root, "LabelHand", f[2]) for f in file_list
        ]


        self.indices = list(range(len(self.s1_image_list)))

        if self.split == 'train':
            selected_idx, _ = balance_seg_indices(
                self, 
                strategy=limited_label_strategy, 
                label_fraction=limited_label, 
                num_bins=10, 
            )
            self.indices = selected_idx


    def __len__(self):
        # return len(self.s1_image_list)
        return len(self.indices)



    def __getitem__(self, index):
        # print("before: ", index)
        index = self.indices[index]
        # print("after: ", index)


        with rasterio.open(self.s2_image_list[index]) as src:
            s2_image = src.read()

        with rasterio.open(self.s1_image_list[index]) as src:
            s1_image = src.read()
            s1_image = np.nan_to_num(s1_image)

        with rasterio.open(self.target_list[index]) as src:
            target = src.read(1)
        
        s2_image = torch.from_numpy(s2_image).float()
        s1_image = torch.from_numpy(s1_image).float()

        target = torch.from_numpy(target).long()

        band_index_map = {
            'B01': s2_image[0],
            'B02': s2_image[1],
            'B03': s2_image[2],
            'B04': s2_image[3],
            'B05': s2_image[4],
            'B06': s2_image[5],
            'B07': s2_image[6],
            'B08': s2_image[7],
            'B8A': s2_image[8],
            'B09': s2_image[9],
            'B10': s2_image[10],
            'B11': s2_image[11],
            'B12': s2_image[12],
            'VV': s1_image[0],
            'VH': s1_image[1],
        }

        img = []
        # for b in self.bands:
        #     ch = (band_index_map[b] - STATS['mean'][b]) / STATS['std'][b]
        #     ch = F.resize(ch.unsqueeze(0), [self.img_size, self.img_size],
        #             interpolation=transforms.InterpolationMode.BILINEAR,
        #         )
        #     img.append(ch.squeeze(0))


        if self.split == 'train':
            i, j, h, w = transforms.RandomCrop.get_params(target, (256, 256))
            for b in self.bands:
                ch = (band_index_map[b] - STATS['mean'][b]) / STATS['std'][b]
                ch = F.crop(ch, i, j, h, w)
                ch = F.resize(ch.unsqueeze(0), [self.img_size, self.img_size],
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )
                img.append(ch.squeeze(0))
            target = F.crop(target, i, j, h, w)
        else:
            for b in self.bands:
                ch = ((band_index_map[b] - STATS['mean'][b]) / STATS['std'][b]).unsqueeze(0)
                ch = F.resize(ch, [self.img_size, self.img_size],
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )
                img.append(ch.squeeze(0))

        target = target.unsqueeze(0)
        target = F.resize(target, [self.img_size, self.img_size],
                            interpolation=transforms.InterpolationMode.NEAREST,
                            )
        target = target.squeeze(0)


        image = torch.stack(img, axis=0) 

        if self.split == 'train':
            if random.random() > 0.5:
                image = F.hflip(image)
                target = F.hflip(target)
            if random.random() > 0.5:
                image = F.vflip(image)
                target = F.vflip(target)
  
    
        return image, target, None, None