# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os.path as osp
import random
from pathlib import Path

from training.data.base_dataset import BaseDataset
from training.data.dataset_util import *


# import torch.distributed as dist

class TartanairOceanDataset(BaseDataset):
    def __init__(
            self,
            common_conf,
            split: str = "train",
            TARTANAIR_OCEAN_DIR: str = 'datasets/tartanair_ocean',
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
        self.TARTANAIR_OCEAN_DIR = TARTANAIR_OCEAN_DIR
        self.min_num_images = min_num_images

        logging.info(f"TARTANAIR_OCEAN_DIR is {self.TARTANAIR_OCEAN_DIR}")
        self.load_sequence_list()

        fx = 320.0
        fy = 320.0
        cx = 320.0
        cy = 320.0

        self.K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # print(K)

        self.sequence_list_len = len(self.sequence_list)
        self.seq_keys = list(self.sequence_list.keys())

        self.depth_max = 50.0
        if split == "train":
            self.len_train = len_train

        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: TARTANAIR_OCEAN_DIR size: {self.sequence_list_len}")
        logging.info(f"{status}: TARTANAIR_OCEAN_DIR direction: {self.nums_direction}")
        logging.info(f"{status}: TARTANAIR_OCEAN_DIR length: {len(self)}")

    def load_sequence_list(self):
        self.nums_direction=0
        self.sequence_list = {}
        total_images=0
        for difficulty in os.listdir(self.TARTANAIR_OCEAN_DIR):
            # if difficulty=='Data_hard':
            #     continue
            difficulty_dir = os.path.join(self.TARTANAIR_OCEAN_DIR, difficulty, 'Ocean', difficulty)
            difficulty_dir = Path(difficulty_dir)  # ⭐ 

            for tracks_line in os.listdir(difficulty_dir):
                track_dir = difficulty_dir / tracks_line
                if not track_dir.is_dir():
                    continue
                scene_name = os.path.join(difficulty, tracks_line)
                self.sequence_list[scene_name] = {}

                pairs, missing_images = self.collect_depth_image_pairs(track_dir)
                # print(f"\n[Track] {tracks_line}")
                # for (cam, d), v in sorted(pairs.items()):
                #     print(
                #         f"  {cam}_{d}: depth={v['depth'].name} | image={v['image'].name if v['image'] else 'MISSING'}")
                #
                # if missing_images:
                #     print("  Missing image for:", missing_images)
                for (cam, d), v in sorted(pairs.items()):
                    # if cam!='lcam' or d!='right':
                    #     continue
                    direction = os.path.join(cam, d)
                    pose_name = v['depth'].name.replace('depth', 'pose')
                    depth_path = os.path.join(track_dir, v['depth'].name)
                    nums_images = self.count_files(depth_path)
                    self.sequence_list[scene_name][direction] = track_dir, v['depth'].name, v['image'].name, pose_name
                    self.sequence_list[scene_name]['nums_images'] = nums_images
                    self.nums_direction+=1
                    total_images+=nums_images
                if not self.sequence_list[scene_name]:
                    del self.sequence_list[scene_name]

        logging.info(f"HYPERSIM_DIR images: {total_images}")


    def parse_depth_name(self, folder_name: str):
        """
        depth_lcam_front -> ('lcam', 'front')
        depth_rcam_top   -> ('rcam', 'top')
        /6 -> None
        """
        VALID_DIRS = {"front", "back", "left", "right", "top", "bottom"}
        parts = folder_name.split("_")
        if len(parts) != 3:
            return None
        prefix, cam, d = parts
        if prefix != "depth":
            return None
        if cam not in {"lcam", "rcam"}:
            return None
        if d not in VALID_DIRS:
            return None
        return cam, d

    def depth_to_image_name(self, cam: str, d: str) -> str:
        return f"image_{cam}_{d}"

    def collect_depth_image_pairs(self, track_dir):
        """
        :
          pairs: dict[(cam,dir)] = {"depth": Path, "image": Path|None}
          missing_images: list[(cam,dir)]
         depth : pairs
        """
        pairs = {}
        missing_images = []

        names = set(p.name for p in track_dir.iterdir())

        for name in names:
            parsed = self.parse_depth_name(name)
            if not parsed:
                continue
            cam, d = parsed
            depth_path = track_dir / name

            image_folder = self.depth_to_image_name(cam, d)
            image_path = track_dir / image_folder
            if image_folder in names and image_path.is_dir():
                pairs[(cam, d)] = {"depth": depth_path, "image": image_path}
            else:
                pairs[(cam, d)] = {"depth": depth_path, "image": None}
                missing_images.append((cam, d))

        return pairs, missing_images

    def count_files(self, path):
        path = Path(path)
        return sum(1 for p in path.iterdir() if p.is_file())

    def quat_xyzw_to_R(self, qx, qy, qz, qw):
        q = np.array([qx, qy, qz, qw], dtype=np.float64)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.eye(3, dtype=np.float64)
        q /= n
        x, y, z, w = q
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ], dtype=np.float64)
        return R

    def read_pose_line(self,pose_txt, idx):
        with open(pose_txt, "r") as f:
            for i, line in enumerate(f):
                if i == idx:
                    return [float(x) for x in line.strip().split()]
        raise IndexError(idx)

    def get_c2w_3x4(self, pose_txt_path: str, pose_id: int):
        """
         3x4  w2c : [R|t],  X_cam = R * X_world + t

        pose_is_c2w:
          - False:  w2c (world->camera),
          - True :  c2w (camera->world), w2c
        """
        tx, ty, tz, qx, qy, qz, qw = self.read_pose_line(pose_txt_path, pose_id)

        # R = self.quat_xyzw_to_R(qx, qy, qz, qw)
        R = self.quat_xyzw_to_R(qx, qy, qz, qw)
        T = np.eye(4, dtype=np.float64)

        R_opt_to_ned = np.array([[0, 0, 1],
                                 [1, 0, 0],
                                 [0, 1, 0]], dtype=np.float64)
        R = R @ R_opt_to_ned
        T[:3, :3] = R
        T[:3,  3] = [tx, ty, tz]
        return T

    def depth_rgba_float32(self, depth_rgba):
        depth = depth_rgba.view("<f4")
        return np.squeeze(depth, axis=-1)

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

        # scene_name = seq_index

        track_data = self.sequence_list[seq_index].copy()
        num_images = track_data['nums_images']
        # num_images=3
        track_data.pop('nums_images')
        direction_keys = track_data.keys()
        # track_dir,depth_name, image_name, pose_name
        # num_images = self.count_rgb_frames(final_hdf5_path)
        if ids is None:
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)
        # print('ids',ids)
        target_image_shape = self.get_target_shape(aspect_ratio)
        # H, W = target_image_shape

        images = []
        depths = []
        cam_points = []
        world_points = []
        image_paths=[]
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        for image_idx in ids:
            direction = np.random.choice(list(direction_keys))
            track_dir, depth_name, image_name, pose_name = track_data[direction]
            # print('track_dir, depth_name, image_name, pose_name',track_dir, depth_name, image_name, pose_name)
            cam, d = os.path.split(direction)
            image_path = osp.join(track_dir, image_name, f'{image_idx:06d}_{cam}_{d}.png')  # 000000_lcam_back.png
            image_paths.append(image_path)
            depth_path = osp.join(track_dir, depth_name,
                                  f'{image_idx:06d}_{cam}_{d}_depth.png')  # 000000_lcam_back_depth.png
            pose_path = osp.join(track_dir, pose_name + '.txt')
            image = read_image_cv2(str(image_path))

            depth_rgba = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth_map = self.depth_rgba_float32(depth_rgba)
            depth_map[depth_map > self.depth_max] = 0.0
            cam2world = self.get_c2w_3x4(str(pose_path), pose_id=image_idx)
            cam2world = cam2world[None, :]
            extri_opencv = closed_form_inverse_se3(cam2world)
            # assert np.allclose(extri_opencv_temp, extri_opencv, atol=1e-6), \
            #     f"Not equal:\n{extri_opencv_temp}\n{extri_opencv}"
            extri_opencv = extri_opencv[0, :3, :]

            intri_opencv = self.K.copy()

            # print('depth_map',depth_map.shape)
            # print('image',image.shape)
            # print('image',image.dtype)
            # print('depth',depth_map.dtype)
            depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=98)

            assert image.shape[
                       :2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            original_size = np.array(image.shape[:2])

            ################ for fake

            # print('z',z.shape)
            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_path,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "Hypersim"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            'image_paths': image_paths,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch
