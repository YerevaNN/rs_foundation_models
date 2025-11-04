# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np
import os
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class ImageFolderInstance(ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        return img, target, index

class ImageFolderMask(ImageFolder):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(ImageFolderMask, self).__init__(*args, **kwargs)
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

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(ImageFolderMask, self).__getitem__(index)
                
        masks = []
        for img in output[0]:
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

        return output + (masks,)
    

class MultiDataLoader:
    def __init__(self, dataloaders, weights=None):
        self.dataloaders = dataloaders
        self.indices = []
        for name, dataloader in dataloaders.items():
            w = 1 if weights is None else weights[name]
            self.indices = self.indices + int(w) * len(dataloader) * [name]

    def __iter__(self):
        self.iterators = {name: iter(dataloader) for name, dataloader in self.dataloaders.items()}
        if dist.get_rank() == 0:
            random.shuffle(self.indices)
        torch.distributed.broadcast_object_list(self.indices, src=0)
        
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index >= len(self.indices):
           raise StopIteration

        curr_name = self.indices[self.iter_index]
        self.iter_index += 1

        try:
            batch = next(self.iterators[curr_name])
        except StopIteration:
            self.iterators[curr_name] = iter(self.dataloaders[curr_name])
            batch = next(self.iterators[curr_name])

        return batch
    
    def __len__(self):
        return len(self.indices)
    
    def set_epoch(self, epoch):
        for data_loader in self.dataloaders.values():
            data_loader.sampler.set_epoch(epoch)
            data_loader.dataset.set_epoch(epoch)

    
class ContinuousDistributedSampler(DistributedSampler):
    def __init__(self, 
                 dataset,
                 num_replicas=None, 
                 rank=None, 
                 shuffle=True, 
                 seed=0, 
                 drop_last=False
                 ):
        super(ContinuousDistributedSampler, self).__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_idx = 0

    def set_iteration(self, iteration):
        self.start_idx = iteration % self.num_samples

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        if self.start_idx > 0:
            indices = indices[self.start_idx:]
        return iter(indices)
    

def chunk(indices, size):
    return torch.split(torch.tensor(indices), size)

class MyDistributedBatchSampler(DistributedSampler):
    def __init__(self, datasets_dict, weights,
                 num_replicas=None,
                 rank=None, shuffle=True,
                 seed = 0, drop_last = False, batch_size = 10):
        
        datasets_to_concat = []
        self.datasets = []
        for name, dataset in datasets_dict.items():
            w = 1 if weights is None else weights[name]
            dataset = ConcatDataset([dataset] * int(w))
            self.datasets.append(dataset)

        self.final_dataset = ConcatDataset(self.datasets)
        super().__init__(dataset=self.final_dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
            
        self.batch_size = batch_size
        self.indices = list(range(len(self.final_dataset)))
        self.data_indices = []
        index = 0
        for i, dataset in enumerate(self.datasets):
            self.data_indices.append(self.indices[index:index+len(dataset)])
            index += len(dataset)
            
        num_batches = 0
        for indices in self.data_indices:
            if self.drop_last:
                num_batches += len(indices) // self.batch_size
            else:
                num_batches += (len(indices) + self.batch_size - 1) // self.batch_size

        self.num_batches = math.ceil(num_batches / self.num_replicas)
        self.total_size = self.num_batches * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            for indices in self.data_indices:
                random.shuffle(indices)
                
        all_batches = []
        for indices in self.data_indices:
            batches = chunk(indices, self.batch_size)
            if self.drop_last and len(indices) % self.batch_size != 0:
                batches = batches[:-1]
            all_batches.extend(batches)
            
        all_batches = [batch.tolist() for batch in all_batches]

        # Pad the batch list so that its total length is divisible by num_replicas.
        if len(all_batches) < self.total_size:
            padding_size = self.total_size - len(all_batches)
            all_batches += all_batches[:padding_size]
        assert len(all_batches) == self.total_size

        rank_batches = all_batches[self.rank: self.total_size: self.num_replicas]
        return iter(rank_batches)

    def __len__(self) -> int:
        return self.num_batches


class ContDistBatchSampler(MyDistributedBatchSampler):

    def __init__(self, datasets_dict, weights,
                 num_replicas=None,
                 rank=None, shuffle=True,
                 seed=0, drop_last=False, batch_size=10):
        super().__init__(datasets_dict, weights,
                         num_replicas=num_replicas,
                         rank=rank, shuffle=shuffle,
                         seed=seed, drop_last=drop_last, batch_size=batch_size)
        self.start_idx = 0

    def set_iteration(self, iteration):
        self.start_idx = iteration % self.num_batches

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            for i in range(len(self.data_indices)):
                indices = self.data_indices[i]
                permutation = torch.randperm(len(indices), generator=g).tolist()
                self.data_indices[i] = [indices[j] for j in permutation]

        all_batches = []
        for indices in self.data_indices:
            batches = chunk(indices, self.batch_size)
            if self.drop_last and len(indices) % self.batch_size != 0:
                batches = batches[:-1]
            all_batches.extend(batches)

        all_batches = [batch.tolist() for batch in all_batches]

        if self.shuffle:
            # Deterministically shuffle all batches
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in indices]
        # print(f"all_batches {self.epoch} first 10: {all_batches[:10]}")
        
        if len(all_batches) % self.num_replicas != 0:
            padding_size = self.total_size - len(all_batches)
            all_batches += all_batches[:padding_size]
        assert len(all_batches) == self.total_size
        
        # Partition batches among replicas.
        rank_batches = all_batches[self.rank: self.total_size: self.num_replicas]
        assert len(rank_batches) == self.num_batches

        if self.start_idx > 0:
            rank_batches = rank_batches[self.start_idx:]
        return iter(rank_batches)


class ContDistSyncBatchSampler(MyDistributedBatchSampler):

    def __init__(self, datasets_dict, weights,
                 num_replicas=None,
                 rank=None, shuffle=True,
                 seed=0, drop_last=False, batch_size=10):
        super().__init__(datasets_dict, weights,
                         num_replicas=num_replicas,
                         rank=rank, shuffle=shuffle,
                         seed=seed, drop_last=drop_last, batch_size=batch_size)
        self.start_idx = 0

    def set_iteration(self, iteration):
        self.start_idx = iteration % self.num_batches

    def __iter__(self):
        # Set up a generator for deterministic shuffling (shared across ranks)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        rank_batches = []
        for indices in self.data_indices:
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[j] for j in perm]
            
            # Create batches
            batches = chunk(indices, self.batch_size)  # Helper function to split into batches
            if self.drop_last and len(indices) % self.batch_size != 0:
                batches = batches[:-1]  # Drop incomplete batch if specified
            
            batches = [batch.tolist() for batch in batches]
            # Make length of batches divisible by num_replicas by removing last elements
            remainder = len(batches) % self.num_replicas
            if remainder > 0:
                batches = batches[:-remainder]
            
            selected_batches = [batches[b] for b in range(len(batches)) 
                                if b % self.num_replicas == self.rank]
            
            rank_batches.extend(selected_batches)

        if self.shuffle:
            # Deterministically shuffle rank batches
            perm = torch.randperm(len(rank_batches), generator=g).tolist()
            rank_batches = [rank_batches[i] for i in perm]

        # Step 4: Apply start_idx for continuous sampling (if needed)
        if self.start_idx > 0:
            rank_batches = rank_batches[self.start_idx:]

        return iter(rank_batches)