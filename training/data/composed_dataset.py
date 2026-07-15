# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import logging
import random
from abc import ABC

import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset

from .augmentation import get_image_augmentation


class ComposedDataset(Dataset, ABC):
    """
    Composes multiple base datasets and applies common configurations.

    This dataset combines multiple base datasets, applies shared augmentations,
    and converts raw sequence data into tensors used by the training loop.
    """

    def __init__(self, dataset_configs: dict, common_config: dict, **kwargs):
        """
        Initializes the ComposedDataset.

        Args:
            dataset_configs (dict): List of Hydra configurations for base datasets.
            common_config (dict): Shared configurations for augmentations and sampling.
            **kwargs: Additional arguments (unused).
        """
        base_dataset_list = []

        # Instantiate each base dataset with common configuration
        for baseset_dict in dataset_configs:
            baseset = instantiate(baseset_dict, common_conf=common_config)
            base_dataset_list.append(baseset)

        # Use custom concatenation class that supports tuple indexing
        self.base_dataset = TupleConcatDataset(base_dataset_list, common_config)

        # --- Augmentation Settings ---
        # Controls whether to apply identical color jittering across all frames in a sequence
        # if hasattr(common_config, 'augs_teacher'):
        #     augs_to_teacher=True
        self.different = common_config.augs.different
        self.cojitter = common_config.augs.cojitter
        # Probability of using shared jitter vs. frame-specific jitter
        self.cojitter_ratio = common_config.augs.cojitter_ratio
        # Initialize image augmentations (color jitter, grayscale, gaussian blur)
        self.image_aug = get_image_augmentation(
            color_jitter=common_config.augs.color_jitter,
            gray_scale=common_config.augs.gray_scale,
            gau_blur=common_config.augs.gau_blur,
        )

        # --- Optional Fixed Settings (useful for debugging) ---
        # Force each sequence to have exactly this many images (if > 0)
        self.fixed_num_images = common_config.fix_img_num
        # Force a specific aspect ratio for all images
        self.fixed_aspect_ratio = common_config.fix_aspect_ratio

        # --- Mode Settings ---
        # Whether the dataset is being used for training (affects augmentations)
        self.training = common_config.training
        self.common_config = common_config

        self.total_samples = len(self.base_dataset)

    def __len__(self):
        """Returns the total number of sequences in the dataset."""
        return self.total_samples

    def __getitem__(self, idx_tuple):
        """
        Retrieves a data sample (sequence) from the dataset.

        Loads raw data, converts it to PyTorch tensors, and applies augmentations.

        Args:
            idx_tuple (tuple): a tuple of (seq_idx, num_images, aspect_ratio)

        Returns:
            dict: A dictionary containing the sequence data.
        """
        # If fixed settings are provided, override the tuple values
        if self.fixed_num_images > 0:
            seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
            idx_tuple = (seq_idx, self.fixed_num_images, self.fixed_aspect_ratio)

        batch = self.base_dataset[idx_tuple]
        seq_name = batch["seq_name"]
        extrinsics = None
        intrinsics = None
        cam_points = None
        world_points = None

        # --- Data Conversion and Preparation ---
        # Convert numpy arrays to tensors
        images = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).contiguous()

        if 'image_paths' in batch:
            image_paths = batch["image_paths"]  #  concat  list
        else:
            image_paths = None
        # Normalize images from [0, 255] to [0, 1]
        images = images.permute(0, 3, 1, 2).to(torch.get_default_dtype()).div(255)

        if 'teacher_images' in batch and self.different:
            teacher_images = torch.from_numpy(np.stack(batch["teacher_images"]).astype(np.float32)).contiguous()
            teacher_images = teacher_images.permute(0, 3, 1, 2).to(torch.get_default_dtype()).div(255)
        else:
            teacher_images = None

        # Convert other data to tensors with appropriate types
        if 'depths' in batch:
            depths = torch.from_numpy(np.stack(batch["depths"]).astype(np.float32))
            point_masks = torch.from_numpy(
                np.stack(batch["point_masks"]))  # Mask indicating valid depths / world points / cam points per frame
        else:
            depths = None
            point_masks = None

        if 'extrinsics' in batch:
            extrinsics = torch.from_numpy(np.stack(batch["extrinsics"]).astype(np.float32))
            intrinsics = torch.from_numpy(np.stack(batch["intrinsics"]).astype(np.float32))
            cam_points = torch.from_numpy(np.stack(batch["cam_points"]).astype(np.float32))
            world_points = torch.from_numpy(np.stack(batch["world_points"]).astype(np.float32))

        if 'R_matrixs' in batch:
            R_matrixs = torch.from_numpy(np.stack(batch["R_matrixs"]).astype(np.float32))
            thetas = torch.from_numpy(np.stack(batch["thetas"]).astype(np.int16))
            teacher_ids = torch.from_numpy(batch["teacher_ids"])
            mapping = torch.from_numpy(batch["mapping"])
        else:
            R_matrixs = None
            thetas = None
            teacher_ids = None
            mapping = None

        if 'global_thetas' in batch:
            global_R_matrixs = torch.from_numpy(np.stack(batch["global_R_matrixs"]).astype(np.float32))
            global_thetas = torch.from_numpy(np.stack(batch["global_thetas"]).astype(np.float32))
        else:
            global_R_matrixs = None
            global_thetas = None

        ids = torch.from_numpy(batch["ids"])  # Frame indices sampled from the original sequence

        # --- Apply Color Augmentation (training mode only) ---
        if self.training and self.image_aug is not None:
            if self.cojitter and random.random() > self.cojitter_ratio:
                # Apply the same color jittering transformation to all frames
                images = self.image_aug(images)

            else:
                # Apply different color jittering to each frame individually
                for aug_img_idx in range(len(images)):
                    images[aug_img_idx] = self.image_aug(images[aug_img_idx])

        if not self.different:
            teacher_images = images.clone()

        # --- Prepare Final Sample Dictionary ---
        sample = {
            "seq_name": seq_name,
            "ids": ids,
            "images": images,
        }
        if teacher_images is not None:
            sample['teacher_images'] = teacher_images
        if image_paths is not None:
            sample["image_paths"] = image_paths

        if depths is not None:
            sample['depths'] = depths
            sample['point_masks'] = point_masks
        if extrinsics is not None:
            sample['extrinsics'] = extrinsics
            sample['intrinsics'] = intrinsics
            sample['cam_points'] = cam_points
            sample['world_points'] = world_points
        if R_matrixs is not None:
            sample['R_matrixs'] = R_matrixs
            sample['thetas'] = thetas
            sample['teacher_ids'] = teacher_ids
            sample['mapping'] = mapping

        if global_thetas is not None:
            sample['global_R_matrixs'] = global_R_matrixs
            sample['global_thetas'] = global_thetas

        return sample


class TupleConcatDataset(ConcatDataset):
    """
    A custom ConcatDataset that supports indexing with a tuple.

    Standard PyTorch ConcatDataset only accepts an integer index. This class extends
    that functionality to allow passing a tuple like (sample_idx, num_images, aspect_ratio),
    where the first element is used to determine which sample to fetch, and the full
    tuple is passed down to the selected dataset's __getitem__ method.

    It also supports an option to randomly sample across all datasets, ignoring the
    provided index. This is useful during training when shuffling the entire dataset
    might cause memory issues due to duplicating dictionaries. If doing this, you can
    set pytorch's dataloader shuffle to False.
    """

    def __init__(self, datasets, common_config):
        """
        Initialize the TupleConcatDataset.

        Args:
            datasets (iterable): An iterable of PyTorch Dataset objects to concatenate.
            common_config (dict): Common configuration dict, used to check for random sampling.
        """
        super().__init__(datasets)
        # If True, ignores the input index and samples randomly across all datasets
        # This provides an alternative to dataloader shuffling for large datasets
        self.inside_random = common_config.inside_random

        datasets_start = 0
        self.no_label_start = 0
        self.only_depth_start = 0
        self.only_depth_end = 0
        self.no_label_end = 0
        for i, ds in enumerate(self.datasets):
            logging.info("Dataset %d: %s, length=%d", i, type(ds).__name__, len(ds))
            if hasattr(ds, "no_label_dataset"):
                self.no_label_start = datasets_start
                self.no_label_end = datasets_start + len(ds)
            if hasattr(ds, "only_depth_label_dataset"):
                self.only_depth_start = datasets_start
                self.only_depth_end = datasets_start + len(ds)
            datasets_start += len(ds)
        self.normal_start = max(self.no_label_end, self.only_depth_end)
        logging.info(f'\nno label start with {self.no_label_start},\n '
                     f'no label end with {self.no_label_end}, \n'
                     f'only depth start with {self.only_depth_start}, \n'
                     f'only depth end with {self.only_depth_end},\n'
                     f'norm data start with {self.normal_start}, \n'
                     f'norm data end with {self.cumulative_sizes[-1]}\n'
                     )

    def __getitem__(self, idx):
        """
        Retrieves an item using either an integer index or a tuple index.

        Args:
            idx (int or tuple): The index. If tuple, the first element is the sequence
                               index across the concatenated datasets, and the rest are
                               passed down. If int, it's treated as the sequence index.

        Returns:
            The item returned by the underlying dataset's __getitem__ method.

        Raises:
            ValueError: If the index is out of range or the tuple doesn't have exactly 3 elements.
        """
        idx_tuple = None
        class_mode = None
        multi_class_dataset_mode = False
        if isinstance(idx, tuple):
            if len(idx) == 3:
                idx_tuple = idx
                idx = idx_tuple[0]  # Extract the sequence index
            elif len(idx) == 4:
                multi_class_dataset_mode = True
                class_mode = idx[-1]
                idx_tuple = idx[:-1]
                idx = idx_tuple[0]

        # Override index with random value if inside_random is enabled
        if self.inside_random:
            total_len = self.cumulative_sizes[-1]
            if multi_class_dataset_mode:
                if class_mode == 'no_label':
                    idx = random.randint(self.no_label_start, self.no_label_end - 1)
                elif class_mode == 'only_depth':
                    idx = random.randint(self.only_depth_start, self.only_depth_end - 1)
                elif class_mode == 'normal':
                    idx = random.randint(self.normal_start, total_len - 1)
                else:
                    raise ValueError(f"class_mode '{class_mode}' not recognized")
            else:
                idx = random.randint(0, total_len - 1)

        # Handle negative indices
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Create the tuple to pass to the underlying dataset
        if len(idx_tuple) == 3:
            idx_tuple = (sample_idx,) + idx_tuple[1:]
        else:
            raise ValueError("Tuple index must have exactly three elements")

        # Pass the modified tuple to the appropriate dataset
        return self.datasets[dataset_idx][idx_tuple]
