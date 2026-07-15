# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging
import os.path as osp
import random
import re

import h5py

from training.data.base_dataset import BaseDataset
from training.data.dataset_util import *
from training.data.water_synthesis import sample_water_parameters, synthesize_underwater_image


# import torch.distributed as dist

class HypersimDataset(BaseDataset):
    def __init__(
            self,
            common_conf,
            split: str = "train",
            HYPERSIM_DIR: str = 'datasets/hypersim',
            HYPERSIM_CAMERA_CSV: str = 'datasets/hypersim_camera_parameters.csv',
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
        self.HYPERSIM_DIR = HYPERSIM_DIR
        self.min_num_images = min_num_images

        logging.info(f"HYPERSIM_DIR is {self.HYPERSIM_DIR}")
        self.load_sequence_list()
        self.intri_opencv_full = {}
        self.camera_cvs_path = HYPERSIM_CAMERA_CSV
        with open(self.camera_cvs_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scene = row["scene_name"]
                self.intri_opencv_full[scene] = row

        self.sequence_list_len = len(self.sequence_list)
        self.seq_keys = list(self.sequence_list.keys())

        self.depth_max = 80
        if split == "train":
            self.len_train = len_train

        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: HYPERSIM_DIR size: {self.sequence_list_len}")
        logging.info(f"{status}: HYPERSIM_DIR length: {len(self)}")

    def load_sequence_list(self):
        self.sequence_list = {}
        total_images=0
        for scene in os.listdir(self.HYPERSIM_DIR):
            scene_dir = os.path.join(self.HYPERSIM_DIR, scene)
            images_dir = os.path.join(scene_dir, 'images')
            detail_path = os.path.join(scene_dir, '_detail')
            camera_ids = self.load_camera_ids(scene_dir)  # ['cam_00']
            # assert len(camera_ids) == 1
            for camera_id in camera_ids:
                scene_name = os.path.join(scene, camera_id)
                # scene_name = scene
                camera_detail_path = os.path.join(detail_path, camera_id)
                final_hdf5_path = osp.join(images_dir, f'scene_{camera_id}_final_hdf5')
                if not os.path.exists(final_hdf5_path):
                    continue
                geometry_hdf5_path = osp.join(images_dir, f'scene_{camera_id}_geometry_hdf5')
                image_ids = self.list_rgb_frame_ids(final_hdf5_path,geometry_hdf5_path)
                self.sequence_list[
                    scene_name] = image_ids, final_hdf5_path, camera_detail_path, geometry_hdf5_path, detail_path
                total_images+=len(image_ids)
        logging.info(f"HYPERSIM_DIR images: {total_images}")

    def load_camera_ids(self, scene_dir: str):
        """
         Hypersim  metadata_cameras.csv, camera_id 
        : ['cam_00', 'cam_01']
        """
        path = f"{scene_dir}/_detail/metadata_cameras.csv"
        camera_ids = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                camera_ids.append(row["camera_name"])

        return camera_ids

    def list_rgb_frame_ids(self, scene_cam_final_dir: str,geometry_hdf5_path) -> list[int]:
        """
         Hypersim scene_cam_XX_final_hdf5 
         RGB (int),
        : [0, 1, 3, 4, 6]
        """
        color_pattern = re.compile(r"^frame\.(\d{4})\.color\.hdf5$")
        frame_ids = []

        for fname in os.listdir(scene_cam_final_dir):
            m = color_pattern.match(fname)
            if not m:
                continue

            frame_idx = int(m.group(1))
            depth_name = f"frame.{frame_idx:04d}.depth_meters.hdf5"
            depth_path = os.path.join(geometry_hdf5_path, depth_name)

            if os.path.isfile(depth_path):
                frame_ids.append(frame_idx)
            else:
                logging.debug('Skip frame without depth: %s', frame_idx)

        frame_ids.sort()
        return frame_ids

    def get_K(self, scene_name: str):
        """
        O(1) 
        """
        scene_name = scene_name.split("/")[0]
        if scene_name not in self.intri_opencv_full:
            raise KeyError(f"scene_name={scene_name} not found in camera parameters CSV")

        row = self.intri_opencv_full[scene_name]

        W = float(row["settings_output_img_width"])
        H = float(row["settings_output_img_height"])
        fov = float(row["settings_camera_fov"])  # rad

        fx = (W / 2.0) / math.tan(fov / 2.0)  # FOV
        fy = fx  # Hypersim:
        cx = W / 2.0
        cy = H / 2.0

        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        return K
        # return K, (int(W), int(H)), (fx, fy, cx, cy)

    def load_camera_poses_c2w(self, carera_detail_path: str):
        R_path = os.path.join(carera_detail_path, "camera_keyframe_orientations.hdf5")
        t_path = os.path.join(carera_detail_path, "camera_keyframe_positions.hdf5")
        R_cw_all = self.read_hdf5_first_dataset(R_path).astype(np.float64)  # (N,3,3)
        t_cw_all = self.read_hdf5_first_dataset(t_path).astype(np.float64)  # (N,3) asset units
        return R_cw_all, t_cw_all

    def _process_image(self, in_rgb_hdf5_file):
        gamma = 1.0 / 2.2  # standard gamma correction exponent
        inv_gamma = 1.0 / gamma
        percentile = 90  # we want this percentile brightness value in the unmodified image...
        brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling
        rgb_color = self.read_hdf5_first_dataset(in_rgb_hdf5_file).astype(np.float32)
        brightness = 0.3 * rgb_color[:, :, 0] + 0.59 * rgb_color[:, :, 1] + 0.11 * rgb_color[
            :, :, 2]  # "CCIR601 YIQ" method for computing brightness
        brightness_valid = brightness

        eps = 0.0001  # if the kth percentile brightness value in the unmodified image is less than this, set the scale to 0.0 to avoid divide-by-zero
        brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:

            # Snavely uses the following expression in the code at https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
            # scale = np.exp(np.log(brightness_nth_percentile_desired)*inv_gamma - np.log(brightness_nth_percentile_current))
            #
            # Our expression below is equivalent, but is more intuitive, because it follows more directly from the expression:
            # (scale*brightness_nth_percentile_current)^gamma = brightness_nth_percentile_desired

            scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current

        rgb_color_tm = np.power(np.maximum(scale * rgb_color, 0), gamma)
        rgb_color_tm = np.clip(rgb_color_tm, 0, 1) * 255
        rgb_color_tm = rgb_color_tm.astype(np.uint8)
        return rgb_color_tm

    def get_single_pose_c2w(self, R_cw_all, t_cw_all, frame_id, meters_per_asset_unit):
        return R_cw_all[frame_id], t_cw_all[frame_id] * meters_per_asset_unit

    def make_cam2world_opencv(self, R_cw_hs: np.ndarray, t_cw_hs: np.ndarray) -> np.ndarray:
        """
        :Hypersim  c2w:  X_world = R_cw_hs X_hs + t_cw_hs
        :OpenCV  c2w:    X_world = R_cw_cv X_cv + t_cw_cv
         X_cv = S X_hs, S=diag(1,-1,-1)
        => X_world = R_cw_hs (S^{-1} X_cv) + t ; S^{-1}=S
        => R_cw_cv = R_cw_hs @ S
        t ()
        """
        S = np.diag([1.0, -1.0, -1.0])  # hs_cam -> cv_cam
        R_cw_cv = R_cw_hs @ S
        t_cw_cv = t_cw_hs

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_cw_cv
        T[:3, 3] = t_cw_cv
        return T

    def get_opencv_camera_params(self, R_cw_all, t_cw_all, frame_id,
                                 meters_per_asset_unit):

        R_cw_hs, t_cw_hs = self.get_single_pose_c2w(R_cw_all, t_cw_all, frame_id, meters_per_asset_unit)
        cam2world = self.make_cam2world_opencv(R_cw_hs, t_cw_hs)  # (4,4) float32 (c2w, OpenCV cam axes)
        return cam2world

    def read_hdf5_first_dataset(self, path: str) -> np.ndarray:
        with h5py.File(path, "r") as f:
            key = next(iter(f.keys()))
            return f[key][()]

    def load_meters_per_asset_unit(self, detail_path: str) -> float:
        path = osp.join(detail_path, 'metadata_scene.csv')
        meters_per_asset_unit = None

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # ['parameter_name', 'parameter_value']

            for row in reader:
                if len(row) >= 2 and row[0] == "meters_per_asset_unit":
                    meters_per_asset_unit = float(row[1])
                    break

        if meters_per_asset_unit is None:
            raise RuntimeError("meters_per_asset_unit not found in metadata_scene.csv")

        return meters_per_asset_unit

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

        scene_name = seq_index

        image_ids, final_hdf5_path, camera_detail_path, geometry_hdf5_path, detail_path = self.sequence_list[seq_index]

        # num_images = self.count_rgb_frames(final_hdf5_path)
        num_images = len(image_ids)

        if ids is None:
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)
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
        intri_opencv_shared = self.get_K(scene_name)
        R_cw_all, t_cw_all = self.load_camera_poses_c2w(camera_detail_path)
        # print('R_cw_all',len(R_cw_all))
        # print('t_cw_all',len(t_cw_all))
        # print('num_images',num_images)
        meters_per_asset_unit = self.load_meters_per_asset_unit(detail_path)
        # print('meters_per_asset_unit',meters_per_asset_unit)

        for image_idx in ids:
            spatial_strength = np.random.uniform(0.05, 0.3)
            image_name_idx = image_ids[image_idx]
            image_name = f"frame.{image_name_idx:04d}.color.hdf5"
            # print('image_name',image_name)
            image_filepath = osp.join(final_hdf5_path, image_name)
            image_paths.append(image_filepath)
            image=self._process_image(image_filepath)
            # image = self.read_hdf5_first_dataset(image_filepath).astype(np.float64)
            # image = np.nan_to_num(image, nan=0.0, posinf=1e6, neginf=0.0)
            # color = np.clip(image, 0.0, None)
            # color = color / (1.0 + color)
            # image = np.clip(color, 0.0, 1.0) * 255
            # image = image.astype(np.uint8)
            # print('image', np.max(image))
            depth_name = f"frame.{image_name_idx:04d}.depth_meters.hdf5"  # frame.0000.depth_meters.hdf5
            depth_filepath = osp.join(geometry_hdf5_path, depth_name)
            distance_map = self.read_hdf5_first_dataset(depth_filepath).astype(np.float64)
            # print('distance_map', np.min(distance_map))
            invalid_mask = ~np.isfinite(distance_map)
            distance_map[invalid_mask] = 0.0
            # has_nan = np.isnan(distance_map).any()
            # print(has_nan)
            # print('distance_map',distance_map.shape)
            intri_opencv = intri_opencv_shared.copy()
            H_temp, W_temp = distance_map.shape
            fx, fy = intri_opencv[0, 0], intri_opencv[1, 1]
            cx, cy = intri_opencv[0, 2], intri_opencv[1, 2]

            u, v = np.meshgrid(np.arange(W_temp), np.arange(H_temp))

            x = (u - cx) / fx
            y = (v - cy) / fy

            den = np.sqrt(x * x + y * y + 1.0)
            depth_z = distance_map / (den + 1e-12)

            depth_map = depth_z.astype(np.float64)
            # print('depth_map max',np.max(depth_map))
            # print('depth_map min',np.min(depth_map))
            # print('depth_map',depth_map.shape)
            # print('image',image.dtype)
            # print('depth',depth_map.dtype)
            depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=98)

            assert image.shape[
                       :2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            original_size = np.array(image.shape[:2])

            cam2world = self.get_opencv_camera_params(R_cw_all, t_cw_all, image_name_idx, meters_per_asset_unit)
            cam2world = cam2world[None, :]
            # print('cam2world',cam2world.shape)
            extri_opencv = closed_form_inverse_se3(cam2world)
            # assert np.allclose(extri_opencv_temp, extri_opencv, atol=1e-6), \
            #     f"Not equal:\n{extri_opencv_temp}\n{extri_opencv}"
            extri_opencv = extri_opencv[0, :3, :]

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
                filepath=image_filepath,
            )

            water_image = synthesize_underwater_image(
                image,
                depth_map,
                water_params,
                spatial_strength=spatial_strength,
                invalid_mask=~point_mask,
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
