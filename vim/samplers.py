# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.distributed as dist
import math
from collections import defaultdict
import numpy as np
from PIL import Image

class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, num_repeats: int = 3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if num_repeats < 1:
            raise ValueError("num_repeats should be greater than 0")
        self.dataset = dataset
        # print(self.dataset,"self.dataset")
        self.num_replicas = num_replicas
        # print(self.num_replicas,"self.num_replicas")
        self.rank = rank
        self.num_repeats = num_repeats # each sample is repeated 3 times for augmentation.
        # print(self.num_repeats,"self.num_repeats")
        self.epoch = 0
        # Control how many samples per process (GPU) are yielded in each epoch.
        self.num_samples = int(math.ceil(len(self.dataset) * self.num_repeats / self.num_replicas))
        # print(self.num_samples,"self.num_samples")
        self.total_size = self.num_samples * self.num_replicas # Total number of samples across all ranks after repeat and padding.
        # print(self.total_size,"self.total_size")
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas)) # (nearest multiple of 256 less than dataset size)
        # print(self.num_selected_samples,"self.num_selected_samples")
        # this restricts your per-rank samples to less than half your full dataset. I only want to use as many samples per epoch as I would use from the original dataset (without repeat), and I’ll make it divisible by batch size and number of GPUs.
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))
        # print(indices,"indices1",len(indices))
        # add extra samples to make it evenly divisible
        indices = torch.repeat_interleave(indices, repeats=self.num_repeats, dim=0).tolist()
        # print(indices,"indices2",len(indices))
        padding_size: int = self.total_size - len(indices)
        # print(padding_size,"padding_size")
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas] # each distributed process (rank) gets a different subset of repeated indices.
        # print(indices,"indices")
        # import sys 
        # sys.exit(0)
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self): 
        return self.num_selected_samples # Number of batches per epoch per rank

    def set_epoch(self, epoch):
        self.epoch = epoch


class BalancedRASampler(torch.utils.data.Sampler):
    """Distributed sampler with repeated augmentation and class-balanced sampling."""

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, num_repeats: int = 3, batch_size: int = 64):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if num_repeats < 1:
            raise ValueError("num_repeats must be >= 1")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_repeats = num_repeats
        self.epoch = 0
        self.batch_size=batch_size

        # Step 1: Build index mapping per class
        self.class_to_indices = self._get_class_indices()
        # print(self.class_to_indices,"self.class_to_indices")
        self.class_counts = {cls: len(idxs) for cls, idxs in self.class_to_indices.items()} # {0: 11844, 1: 1608} self.class_counts for streetlight_18k
        # print(self.class_counts,"self.class_counts")
        self.min_class_count = min(self.class_counts.values()) # 1608 for streetlight_18k
        self.max_class_count = max(self.class_counts.values()) # 11844
        # print(self.min_class_count,"self.min_class_count")
        self.repeat_factor = math.floor(self.max_class_count / self.min_class_count)
        # print(self.repeat_factor,"self.repeat_factor")
        self.repeated_min_count = self.min_class_count * self.repeat_factor

        # Step 2: Define how many total samples we use per epoch (balanced)
        # self.total_balanced_samples = self.min_class_count * len(self.class_to_indices) # 3216 for streetlight_18k
        self.total_balanced_samples = self.repeated_min_count + self.max_class_count  # 11256 + 11844 = 23100
        # print(self.total_balanced_samples,"self.total_balanced_samples")
        
        # Step 3: Total repeated and sharded samples per rank
        self.num_samples = int(math.ceil(self.total_balanced_samples * self.num_repeats / self.num_replicas)) # 3216*3/2gpus=4824 for streetlight_18k
        # print(self.num_samples,"self.num_samples")
        self.total_size = self.num_samples * self.num_replicas # 9648 for 2 gpus(replicas)
        # print(self.total_size,"self.total_size")

        # Step 4: Ensure output per rank fits evenly into batches
        alignment = self.batch_size * self.num_replicas
        self.num_selected_samples = int(math.floor(self.total_balanced_samples // alignment * alignment / self.num_replicas)) # Each replica gets 1536 samples, so total across 2 replicas is 3072, which fits in 12 × 256 blocks.
        # print(self.num_selected_samples,"self.num_selected_samples")

    # def _get_class_indices(self):  # this works fine for vim, vit, but mix the order of idx and sample from the dataset when using r50 model
    #     """Group dataset indices by class label."""
    #     class_to_indices = defaultdict(list)
    #     for idx in range(len(self.dataset)):
    #         sample = self.dataset[idx]
    #         if len(sample) == 3:
    #             _, label, _ = sample
    #         else:
    #             _, label = sample
    #         # _, label, _ = self.dataset[idx] 
    #         # _, label = self.dataset[idx]
    #         class_to_indices[label].append(idx)
    #     return class_to_indices

    # def _get_class_indices(self):
    #     class_to_indices = defaultdict(list)
    #     for idx in range(len(self.dataset)):
    #         sample = self.dataset[idx]
    #         # Expect sample is (img, label) OR (idx, img, label)
    #         if isinstance(sample, (list, tuple)) and len(sample) == 2:
    #             _, label = sample
    #         elif isinstance(sample, (list, tuple)) and len(sample) == 3:
    #             # (idx, img, label) or (img, label, idx) - handle both
    #             a, b, c = sample
    #             if isinstance(a, int) and not isinstance(b, (Image.Image, torch.Tensor, np.ndarray)):
    #                 # a is idx -> assume (idx, img, label)
    #                 label = c
    #             elif isinstance(c, int) and not isinstance(b, (Image.Image, torch.Tensor, np.ndarray)):
    #                 # c is idx -> assume (img, label, idx)
    #                 label = b
    #             else:
    #                 # fallback: try to find an int-like element
    #                 for elem in sample:
    #                     if isinstance(elem, (int, np.integer)):
    #                         label = int(elem)
    #                         break
    #                 else:
    #                     raise ValueError(f"Cannot infer label in sample at idx {idx}: {sample}")
    #         else:
    #             raise ValueError(f"Unexpected sample format at idx {idx}: {type(sample)}")
    #         if isinstance(label, torch.Tensor):
    #             label = int(label.item())
    #         class_to_indices[label].append(idx)
    #     return class_to_indices
    def _get_class_indices(self):
        class_to_indices = defaultdict(list)
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]

            # ✅ case 1: IndexedDataset(return_index=True) → (idx, (img, label))
            if isinstance(sample, (list, tuple)) and len(sample) == 2 and isinstance(sample[1], (list, tuple)):
                img_label = sample[1]
                if len(img_label) == 2:
                    _, label = img_label
                else:
                    raise ValueError(f"Unexpected inner format (should be (img,label)) at idx {idx}: {img_label}")

            # ✅ case 2: regular dataset → (img, label)
            elif isinstance(sample, (list, tuple)) and len(sample) == 2:
                _, label = sample

            # ✅ fallback: (idx, img, label)
            elif isinstance(sample, (list, tuple)) and len(sample) == 3:
                a, b, c = sample
                if isinstance(a, int):
                    label = c
                else:
                    label = b
            else:
                raise ValueError(f"Unexpected sample format at idx {idx}: {sample}")

            if isinstance(label, torch.Tensor):
                label = int(label.item())

            class_to_indices[label].append(idx)

        return class_to_indices

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        balanced_indices = []
        for cls, cls_indices in self.class_to_indices.items():
            cls_indices = list(cls_indices) # 11844 for 0, 1608 for 1
            # print(cls_indices,len(cls_indices),"cls_indices") 
            if self.shuffle:
                cls_indices = torch.tensor(cls_indices)[torch.randperm(len(cls_indices), generator=g)].tolist()
                # print(cls_indices,"Shuffles within each class before selecting the first min_class_count samples.")
            else:
                cls_indices = list(cls_indices)

            if len(cls_indices) == self.min_class_count:
                extended_indices = (cls_indices * self.repeat_factor) # 7.4 ~=7 times of repeat 1608 * 7 = 11256
                # print(extended_indices,len(extended_indices),"extended_indices")
                balanced_indices.extend(extended_indices[:self.repeated_min_count])  # repeat to max class size
            else:
                balanced_indices.extend(cls_indices[:self.max_class_count])  # should finally contains 11256+11844=23100(11844*2=23688)
        
        if self.shuffle:
            # Shuffles the final combined list of balanced indices (from all classes).
            balanced_indices = torch.tensor(balanced_indices)[torch.randperm(len(balanced_indices), generator=g)].tolist()
        
        # print(balanced_indices,len(balanced_indices),"balanced_indices") # shuffled, 23100 balanced_indices
        indices = torch.repeat_interleave(torch.tensor(balanced_indices), repeats=self.num_repeats).tolist()   # 23100*3=69300
        # print(indices,len(indices),"indices after repeat.") # 46200 indices after repeat 2 times.
        
        # print(f"len(balanced_indices)={len(balanced_indices)}, num_repeats={self.num_repeats}, "
        #     f"len(indices)={len(indices)}, total_size={self.total_size}, "
        #     f"num_replicas={self.num_replicas}, num_samples={self.num_samples}") # add debug for r50 on 20251111

        assert len(indices) == self.total_size

        # shard across distributed processes
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        # print(indices,len(indices),"indices3") # 23100 indices3
        # import sys
        # sys.exit(0)
        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch








class BalancedGroupRASampler(torch.utils.data.Sampler):
    """
    A class-balanced repeated-augmentation sampler WITH 4-crop grouping.
    
    - Each original image corresponds to 4 consecutive indices in the dataset.
    - This sampler always treats a block of 4 samples as *one group*.
    - RASampler repeat happens at the group level, not individual samples.
    - DDP sharding preserves group integrity.
    """

    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, num_repeats=2, batch_size=64, group_size=4):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Distributed training required.")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Distributed training required.")
            rank = dist.get_rank()

        self.dataset = dataset
        self.group_size = group_size  # usually 4
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.batch_size = batch_size
        self.epoch = 0

        # ======= Check dataset consistency =======
        if len(dataset) % group_size != 0:
            raise ValueError(
                f"Dataset length {len(dataset)} is not divisible by group_size={group_size}. "
                f"Dataset must be in format: img0_c0,img0_c1,img0_c2,img0_c3,img1_c0,..."
            )

        self.num_groups = len(dataset) // group_size

        # ======= Build per-class group index mapping =======
        self.class_to_groups = self._build_class_group_map()

        self.min_class_groups = min(len(v) for v in self.class_to_groups.values())
        self.max_class_groups = max(len(v) for v in self.class_to_groups.values())

        self.repeat_factor = math.floor(self.max_class_groups / self.min_class_groups)
        self.balanced_group_count = self.min_class_groups * self.repeat_factor + self.max_class_groups

        # number of groups after repeated augmentation
        self.num_groups_total = self.balanced_group_count * self.num_repeats

        # Samples = groups * group_size
        self.total_size = self.num_groups_total * group_size

        # Per GPU shard size
        self.num_samples = self.total_size // self.num_replicas

        # Align per-rank samples with batch size
        alignment = self.batch_size * self.num_replicas
        aligned_total = (self.total_size // alignment) * alignment
        self.num_samples = aligned_total // self.num_replicas

    # --------------------------------------------------

    def _build_class_group_map(self):
        """
        Map each class → list of group indices.
        Dataset layout:
            group g corresponds to sample indices: [g*4, g*4+1, g*4+2, g*4+3]
        """
        class_to_groups = defaultdict(list)

        for g in range(self.num_groups):
            sample_idx = g * self.group_size
            _,_, label = self.dataset[sample_idx]
            if isinstance(label, torch.Tensor):
                label = int(label.item())
            class_to_groups[label].append(g)

        return class_to_groups
    # --------------------------------------------------

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # ===== 1. Build class-balanced group list =====
        balanced_groups = []

        for cls, group_list in self.class_to_groups.items():
            group_list = list(group_list)
            if self.shuffle:
                group_list = torch.tensor(group_list)[
                    torch.randperm(len(group_list), generator=g)
                ].tolist()

            if len(group_list) == self.min_class_groups:
                extended = group_list * self.repeat_factor
                balanced_groups.extend(extended[: self.min_class_groups * self.repeat_factor])
            else:
                balanced_groups.extend(group_list[: self.max_class_groups])

        if self.shuffle:
            balanced_groups = torch.tensor(balanced_groups)[
                torch.randperm(len(balanced_groups), generator=g)
            ].tolist()

        # ===== 2. Repeated augmentation at GROUP level =====
        groups = torch.repeat_interleave(
            torch.tensor(balanced_groups), repeats=self.num_repeats
        ).tolist()

        # ===== 3. Flatten groups into crop indices =====
        indices = []
        for g_idx in groups:
            start = g_idx * self.group_size
            indices.extend(list(range(start, start + self.group_size)))

        assert len(indices) == self.total_size

        # ===== 4. DDP shard =====
        indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = indices[: self.num_samples]

        return iter(indices)

    # --------------------------------------------------

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch




class GroupDistributedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, group_size=4, seed=0):
        if num_replicas is None:
            if not torch.distributed.is_available():
                num_replicas = 1
            else:
                num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                rank = 0
            else:
                rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.group_size = group_size

        self.num_groups = len(dataset) // group_size
        self.total_size = self.num_groups * group_size
        self.num_samples = math.ceil(self.num_groups / num_replicas) * group_size

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        # group shuffle
        if self.shuffle:
            indices = torch.randperm(self.num_groups, generator=g).tolist()
        else:
            indices = list(range(self.num_groups))

        # expand group index → inside group 4 indices
        expanded = []
        for gi in indices:
            start = gi * self.group_size
            expanded.extend(list(range(start, start + self.group_size)))

        # partition for DDP
        expanded = expanded[self.rank::self.num_replicas]

        return iter(expanded)

    def __len__(self):
        return self.num_samples