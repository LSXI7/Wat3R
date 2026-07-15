# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import logging
import random
from wat3r.utils.geometry import closed_form_inverse_se3
import numpy as np
from training.data.dataset_util import *
from training.data.base_dataset import BaseDataset
from ..water_synthesis import load_relative_depth_map, sample_water_parameters, synthesize_underwater_image


class MegadepthDataset(BaseDataset):
    def __init__(
            self,
            common_conf,
            split: str = "train",
            MEGADEPTH_DIR: str = 'datasets/megadepth_processed',
            MEGADEPTH_META_PATH: str = 'datasets/megadepth_metadata.npz',
            MEGADEPTH_FAKE_DEPTH_DIR: str = 'datasets/megadepth_fake_depth_da3',
            min_num_images: int = 24,
            len_train: int = 100000,
            len_test: int = 10000,
            expand_ratio: int = 8,
    ):
        """
        Initialize the VKittiDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            VKitti_DIR (str): Directory path to VKitti data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            expand_range (int): Range for expanding nearby image selection.
            get_nearby_thres (int): Threshold for nearby image selection.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.expand_ratio = expand_ratio
        self.Megadepth_DIR = MEGADEPTH_DIR
        self.fake_depth_dir = MEGADEPTH_FAKE_DEPTH_DIR
        self.min_num_images = min_num_images
        npz_path = MEGADEPTH_META_PATH
        with np.load(npz_path) as data:
            self.scenes = data['scenes']
            self.sceneids = data['sceneids']
            self.images = data['images']

        if split == "train":
            self.len_train = len_train
            self.select_scene(('0015', '0022'), opposite=True)

        elif split == "test":
            self.len_train = len_test
            self.select_scene(('0015', '0022'))
        else:
            raise ValueError(f"Invalid split: {split}")

        self.sequence_list = {}
        self.image_to_scene = {}
        for img_idx, scene_idx in enumerate(self.sceneids):
            scene = self.scenes[scene_idx]
            if scene not in self.sequence_list:
                self.sequence_list[scene] = []
            self.sequence_list[scene].append(img_idx)
            self.image_to_scene[img_idx] = scene

        logging.info(f"Megadepth_DIR is {self.Megadepth_DIR}")

        self.sequence_list_len = len(self.sequence_list)
        self.seq_keys = list(self.sequence_list.keys())

        self.depth_max = 80

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Megadepth_DIR size: {self.sequence_list_len}")
        logging.info(f"{status}: Megadepth_DIR length: {len(self)}")

    def select_scene(self, scene, opposite=False):
        scenes = (scene,) if isinstance(scene, str) else tuple(scene)
        scene_id = [s.startswith(scenes) for s in self.scenes]
        assert any(scene_id), 'no scene found'

        valid = np.in1d(self.sceneids, np.nonzero(scene_id)[0])

        if opposite:
            valid = ~valid
        assert valid.any()
        self.sceneids = self.sceneids[valid]
        self.images = self.images[valid]

    def get_data(
            self,
            seq_index: int = None,
            img_per_seq: int = None,
            seq_name: str = None,
            ids: list = None,
            aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            # seq_index = random.randint(0, self.sequence_list_len - 1)
            seq_index = random.choice(self.seq_keys)  # '0000/0', '0000/1', '0001/0', '0002/0', '0003/0',
        if seq_name is None:
            seq_name = seq_index

        full_sequence = self.sequence_list[seq_index]

        num_images = len(full_sequence)

        if ids is None:
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)
        # W,H=target_image_shape

        # print('\n\n\n\nmegadepth\n\n\n\n')

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        image_paths = []

        water_params = sample_water_parameters()
        for image_idx in ids:
            spatial_strength = np.random.uniform(0.05, 0.3)
            image_idx_global = full_sequence[image_idx]
            image_name = self.images[image_idx_global]
            #                 image = imread_cv2(osp.join(seq_path, img + '.jpg'))
            #                 depthmap = imread_cv2(osp.join(seq_path, img + ".exr"))
            #                 camera_params = np.load(osp.join(seq_path, img + ".npz"))

            image_path = osp.join(self.Megadepth_DIR, seq_index, image_name)
            image_paths.append(image_path)
            # print('image_path', image_path)
            image_filepath = osp.join(image_path + '.jpg')
            depth_filepath = osp.join(image_path + ".exr")
            camera_params = np.load(osp.join(image_path + ".npz"))
            depth_fake_filepath = osp.join(self.fake_depth_dir, seq_index, image_name + '.jpg.npy')
            # image_filepath = osp.join(self.VKitti_DIR, seq_name, f"rgb_{image_idx:05d}.jpg")
            # depth_filepath = osp.join(self.VKitti_DIR, seq_name, f"depth_{image_idx:05d}.png").replace("/rgb", "/depth")

            image = read_image_cv2(image_filepath)
            depth_map = read_depth(depth_filepath, 1.0)
            depth_map[depth_map > self.depth_max] = 0.0

            depth_fake = load_relative_depth_map(depth_fake_filepath, image.shape[:2])
            water_image = synthesize_underwater_image(
                image,
                depth_fake,
                water_params,
                spatial_strength=spatial_strength,
            )

            depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=98)

            assert image.shape[
                       :2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            original_size = np.array(image.shape[:2])

            intri_opencv = np.float32(camera_params['intrinsics'])

            cam2world = np.float32(camera_params['cam2world'])
            cam2world = cam2world[None, :]
            extri_opencv = closed_form_inverse_se3(cam2world)
            # assert np.allclose(extri_opencv_temp, extri_opencv, atol=1e-6), \
            #     f"Not equal:\n{extri_opencv_temp}\n{extri_opencv}"
            extri_opencv = extri_opencv[0, :3, :]

            (
                image,
                water_image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_two_image(
                image,
                water_image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_filepath,
            )
            if (water_image.shape[:2] != target_image_shape).any():
                logging.error(f"Wrong shape for {seq_name}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(water_image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "megadepth"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            'image_paths': image_paths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch
