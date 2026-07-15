# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import logging
import random
import glob

import cv2
import numpy as np
import torch.distributed as dist
from training.data.dataset_util import *
from training.data.base_dataset import BaseDataset
import csv


class RealVideoDataset(BaseDataset):
    def __init__(
            self,
            common_conf,
            split: str = "train",
            FLSEA_VI_DIR: str = 'datasets/flsea_vi',
            ONLINE_VIDEO_DIR: str = 'datasets/online_video',
            min_num_images: int = 24,
            len_train: int = 100000,
            len_test: int = 10000,
            expand_ratio: int = 8,
            video_classes=None,
            enable_extra_video=True,
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
        self.min_teacher_num_images = 24
        if video_classes is None:
            video_classes = ['pexel', 'pexel2', 'freepik', 'pixabay', 'vecteezy']
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.init_rotate_config()
        self.expand_ratio = expand_ratio
        self.FLSEA_VI_DIR = FLSEA_VI_DIR
        self.ONLINE_VIDEO_DIR = ONLINE_VIDEO_DIR
        self.min_num_images = min_num_images
        self.sequence_dict = {}
        self.no_label_dataset = True

        self.video_classes = video_classes  #

        self.load_flsea_vi_scene_data(self.FLSEA_VI_DIR)
        if split == "train":
            self.len_train = len_train
            self.select_scene(('red_sea/coral_table_loop', 'canyons/u_canyon'), opposite=True)

        elif split == "test":
            self.len_train = len_test
            self.select_scene(('red_sea/coral_table_loop', 'canyons/u_canyon'))

        self.flsea_vi_dict_len = len(self.sequence_dict)
        if enable_extra_video:
            for video_class in self.video_classes:
                self.load_online_video_scene_data(self.ONLINE_VIDEO_DIR, video_class)
        logging.info(f"FLSEA_VI_DIR is {self.FLSEA_VI_DIR}")
        logging.info(f"ONLINE_VIDEO_DIR is {self.ONLINE_VIDEO_DIR}")
        self.sequence_dict_len = len(self.sequence_dict)
        self.seq_keys = list(self.sequence_dict.keys())
        # print('self.seq_keys', self.seq_keys)

        self.depth_max = 80

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: FLSEA_VI_DIR size: {self.flsea_vi_dict_len}")
        logging.info(f"{status}: ONLINE_VIDEO_DIR size: {self.sequence_dict_len - self.flsea_vi_dict_len}")
        logging.info(f"{status}: FLSEA_VI_DIR length: {len(self)}")

    def select_scene(self, scene, opposite=False):
        if opposite:
            self.sequence_dict = {key: value for key, value in self.sequence_dict.items() if key not in scene}
        else:
            self.sequence_dict = {key: value for key, value in self.sequence_dict.items() if key in scene}

    def load_flsea_vi_scene_data(self, gt_root):
        total_images = 0
        for scene in os.listdir(gt_root):
            scene_root = os.path.join(gt_root, scene)
            if not os.path.isdir(scene_root):
                logging.debug('Skip non-scene entry: %s', scene)
                continue

            for obj in os.listdir(scene_root):
                key = f'{scene}/{obj}'

                obj_root = os.path.join(scene_root, obj)
                imgs_root = os.path.join(obj_root, "imgs")

                if not os.path.isdir(imgs_root):
                    logging.debug('Skip entry without imgs directory: %s', obj)
                    continue
                self.sequence_dict[key] = []
                all_imgs = []
                for root, _, files in os.walk(imgs_root):
                    for f in files:
                        if not f.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                            continue
                        img_path = os.path.join(root, f)
                        name_no_ext, _ = os.path.splitext(f)

                        try:
                            ts = int(name_no_ext)  # ()
                        except ValueError:
                            logging.debug('Fallback to mtime for image path: %s', img_path)
                            ts = os.path.getmtime(img_path)  # :

                        all_imgs.append((ts, img_path))
                total_images += len(all_imgs)

                all_imgs.sort(key=lambda x: x[0])

                for _, img_path in all_imgs:
                    # print('img_path', img_path)
                    self.sequence_dict[key].append((scene, obj, img_path))
        logging.info(f"FLSEA_VI_DIR has frame {total_images}")

    def _load_usable_keys(self, csv_path: str) -> set:
        """
         labels.csv, usable == 'y'  key 
        CSV ,:usable,key
        /: usable(y/n)  key(scene/obj)
        """
        usable_keys = set()
        total_rows = 0
        y_rows = 0
        n_rows = 0
        bad_rows = 0

        if not os.path.isfile(csv_path):
            print(f'[WARN] labels.csv not found: {csv_path}')
            return usable_keys

        with open(csv_path, 'r', newline='') as f:
            sample = f.read(4096)
            f.seek(0)
            has_header = ('usable' in sample and 'key' in sample)

            if has_header:
                reader = csv.DictReader(f)
                for row in reader:
                    total_rows += 1
                    usable = (row.get('usable') or '').strip().lower()
                    key = (row.get('key') or '').strip()
                    if usable == 'y' and key:
                        usable_keys.add(key)
                        y_rows += 1
                    elif usable == 'n' and key:
                        n_rows += 1
                    else:
                        bad_rows += 1
            else:
                reader = csv.reader(f, skipinitialspace=True)
                for cols in reader:
                    if not cols:
                        continue
                    total_rows += 1
                    cols = [c.strip() for c in cols if c is not None]

                    usable = None
                    key = None
                    for c in cols:
                        cl = c.lower()
                        if cl in ('y', 'n') and usable is None:
                            usable = cl
                        if '/' in c and key is None:
                            key = c

                    if usable == 'y' and key:
                        usable_keys.add(key)
                        y_rows += 1
                    elif usable == 'n' and key:
                        n_rows += 1
                    else:
                        bad_rows += 1

        logging.info('labels total rows : %d', total_rows)
        logging.info('labels usable=y   : %d', y_rows)
        logging.info('labels usable=n   : %d', n_rows)
        if bad_rows:
            logging.info('labels bad rows   : %d', bad_rows)
        logging.info('usable key count  : %d', len(usable_keys))
        return usable_keys

    def load_online_video_scene_data(self, video_root, video_class):
        total_images = 0
        class_video_root = os.path.join(video_root, video_class)
        labels_csv = os.path.join(class_video_root, 'labels.csv')
        usable_keys = self._load_usable_keys(labels_csv)
        class_video_path = os.path.join(class_video_root, 'out_frame')
        total_candidates = 0
        filtered_by_label_n_or_missing = 0

        seen_objs = set()  #  usable=y (>=36), scene 
        duplicated_count = 0  # 
        duplicated_objs = set()  #  obj 

        kept_keys = 0
        filtered_by_len = 0

        for scene in os.listdir(class_video_path):
            scene_root = os.path.join(class_video_path, scene)
            if not os.path.isdir(scene_root):
                logging.debug('Skip non-scene entry: %s', scene)
                continue

            for obj in os.listdir(scene_root):
                key = f'{scene}/{obj}'
                global_key = f'{video_class}/{scene}/{obj}'
                total_candidates += 1

                if key not in usable_keys:
                    filtered_by_label_n_or_missing += 1
                    continue

                if obj in seen_objs:
                    duplicated_count += 1
                    duplicated_objs.add(obj)
                    # print(f'skip duplicated obj "{obj}" in scene "{scene}" (key={key})')
                    continue

                obj_root = os.path.join(scene_root, obj)
                if not os.path.isdir(obj_root):
                    logging.debug('Skip non-directory entry: %s', obj)
                    continue

                all_imgs = []
                for root, _, files in os.walk(obj_root):
                    for f in files:
                        if not f.lower().endswith((
                                '.tiff', '.tif', '.png', '.jpg',
                                '.jpeg', '.bmp', '.gif'
                        )):
                            continue
                        img_path = os.path.join(root, f)
                        name_no_ext, _ = os.path.splitext(f)
                        try:
                            ts = int(name_no_ext)
                        except ValueError:
                            # print(f'false get time name {img_path}')
                            ts = os.path.getmtime(img_path)
                        all_imgs.append((ts, img_path))

                if len(all_imgs) < 12:
                    filtered_by_len += 1
                    continue
                total_images += len(all_imgs)
                all_imgs.sort(key=lambda x: x[0])

                seen_objs.add(obj)

                self.sequence_dict[global_key] = []
                for _, img_path in all_imgs:
                    self.sequence_dict[global_key].append((scene, obj, img_path))
                kept_keys += 1

        logging.info('================ final summary ================')
        logging.info(f'total candidates scanned         : {total_candidates}')
        logging.info(f'filtered by label (n or missing) : {filtered_by_label_n_or_missing}')
        logging.info(f'filtered by <36 imgs             : {filtered_by_len}')
        logging.info(f'kept keys                        : {kept_keys}')
        logging.info(f'duplicated occurrences skipped   : {duplicated_count}')
        logging.info(f"{video_class} has frame {total_images}")
        # print(f'duplicated obj names             : {len(duplicated_objs) if False else len(duplicated_objs)}')
        # if duplicated_objs:
        #     print('duplicated obj list              :', sorted(duplicated_objs))

    def init_rotate_config(self):
        self.ANGLES = [0, 90, 180, 270]
        # self.ANGLES = [0]

        self.R_LOOKUP = {
            0: np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=float),
            90: np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ], dtype=float),
            180: np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ], dtype=float),
            270: np.array([
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ], dtype=float),
        }
        self.ROTATE_LOOKUP = {
            0: lambda img: img,
            90: lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
            180: lambda img: cv2.rotate(img, cv2.ROTATE_180),
            270: lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        }

    def get_image_rotate(self, img):
        img = img.copy()
        theta = random.choice(self.ANGLES)
        img_rot = self.ROTATE_LOOKUP[theta](img)
        R_img = self.R_LOOKUP[theta]
        return img_rot, R_img, theta

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
        # print('seq_name',seq_name)
        full_sequence = self.sequence_dict[seq_index]

        num_images = len(full_sequence)
        # print('\n\n\n\nwater\n\n\n\n')
        # print('num_images', num_images)
        # print('ids1',ids)
        if ids is None:
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)
        # print('ids2',ids)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        teacher_ids, teacher_mask = self.get_teacher_ids(ids=ids,
                                                         target_n=max(self.min_teacher_num_images, 3 * len(ids)),
                                                         full_seq_num=num_images)
        indices = np.random.permutation(len(teacher_ids))

        teacher_ids = teacher_ids[indices]
        teacher_mask = teacher_mask[indices]
        # print('teacher_ids',teacher_ids)
        # print('teacher_mask',teacher_mask)
        #
        # print('expand_ratio',self.expand_ratio)
        # print('ids3',ids)
        index_mapping = self.build_index_mapping(ids=ids, teacher_ids=teacher_ids)
        # print('index_mapping',index_mapping)

        target_image_shape = self.get_target_shape(aspect_ratio)
        # target_image_shape=np.array((518,518))

        student_images = []
        R_matrixs = []
        thetas = []
        original_sizes = []
        image_paths = []
        teacher_images = [None] * len(teacher_ids)
        image_cache = {}

        for i, image_idx in enumerate(ids):
            teacher_pos = np.where(teacher_ids == image_idx)[0]
            scene, obj, img_path = full_sequence[image_idx]
            image = read_image_cv2(img_path, rgb=True)
            original_size = np.array(image.shape[:2])

            image_cache[image_idx] = (image, original_size)
            image_paths.append(img_path)
            teacher_image = self.resize_and_center_crop(image, target_h=target_image_shape[0],
                                                        target_w=target_image_shape[1])

            rotate_image, R_matrix, theta = self.get_image_rotate(teacher_image)
            # print('rotate_image',rotate_image.shape)
            student_image = cv2.resize(rotate_image, (target_image_shape[1], target_image_shape[0]))
            # student_image = self.resize_and_center_crop(rotate_image, target_h=target_image_shape[0],
            #                                             target_w=target_image_shape[1])
            if (student_image.shape[:2] != target_image_shape).any():
                logging.error(
                    f"Wrong shape for {seq_name}: expected {target_image_shape}, "
                    f"got {student_image.shape[:2]}"
                )
                continue

            student_images.append(student_image)
            R_matrixs.append(R_matrix)
            thetas.append(theta)
            original_sizes.append(original_size)

            # teacher_image = cv2.resize(image, (target_image_shape[1], target_image_shape[0]))


            for p in teacher_pos:
                teacher_images[p] = teacher_image

        for t_i, t_idx in enumerate(teacher_ids):
            if teacher_images[t_i] is not None:
                continue  # 

            if t_idx in image_cache:
                image, _ = image_cache[t_idx]
            else:
                scene, obj, img_path = full_sequence[t_idx]
                image = read_image_cv2(img_path, rgb=True)
                original_size = np.array(image.shape[:2])
                image_cache[t_idx] = (image, original_size)

            # teacher_image = cv2.resize(image, (target_image_shape[1], target_image_shape[0]))
            teacher_image = self.resize_and_center_crop(image, target_h=target_image_shape[0],
                                                        target_w=target_image_shape[1])
            teacher_images[t_i] = teacher_image
        set_name = "vkitti"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "teacher_ids": np.asarray(teacher_ids),
            'mapping': np.asarray(index_mapping),
            "frame_num": len(ids),
            "R_matrixs": R_matrixs,
            "images": student_images,
            "thetas": thetas,
            'image_paths': image_paths,
            'teacher_images': teacher_images,
            "original_sizes": original_sizes,
        }
        return batch

    def get_teacher_ids(self, ids, target_n, full_seq_num=None):

        """
         ids  target_n , index  ids 

        Args:
            ids (array-like[int]):  ids, = num_images
            target_n (int): , > len(ids)
            full_seq_num (int, optional): (), clip

        Returns:
            all_ids (np.ndarray[int]):  target_n  ids
            mask (np.ndarray[bool]):  target_n  mask,
                                     True  ids,False 
        """
        ids = np.asarray(ids, dtype=int)
        num_images = len(ids)

        if target_n <= num_images:
            raise ValueError(f"target_n ({target_n}) must be > num_images ({num_images})")

        num_new = target_n - num_images

        sorted_ids = np.sort(ids)
        segments = []
        for i in range(len(sorted_ids) - 1):
            a = sorted_ids[i]
            b = sorted_ids[i + 1]
            segments.append([a, b])

        new_ids = []

        for _ in range(num_new):
            valid_indices = [i for i, (a, b) in enumerate(segments) if b - a > 1]
            if not valid_indices:
                break  # ,

            best_idx = max(valid_indices, key=lambda i: segments[i][1] - segments[i][0])
            a, b = segments[best_idx]

            if b - a <= 1:
                continue

            mid = (a + b) // 2
            new_ids.append(mid)

            left = [a, mid]
            right = [mid, b]
            segments[best_idx:best_idx + 1] = [left, right]

        if len(new_ids) < num_new:
            remaining = num_new - len(new_ids)
            if full_seq_num is not None:
                low, high = 0, full_seq_num
            else:
                low = sorted_ids.min()
                high = sorted_ids.max() + 1  #  randint 
            extra = np.random.randint(low, high, size=remaining)
            new_ids.extend(extra.tolist())

        new_ids = np.asarray(new_ids[:num_new], dtype=int)

        if full_seq_num is not None:
            new_ids = np.clip(new_ids, 0, full_seq_num - 1)

        all_ids = np.concatenate([ids, new_ids], axis=0)

        mask = np.zeros(target_n, dtype=bool)
        mask[:num_images] = True  #  ids

        return all_ids, mask

    def build_index_mapping(self, ids, teacher_ids):
        """
         ids[i]  teacher_ids ( ID)
         ids 
        """
        ids = np.asarray(ids)
        teacher_ids = np.asarray(teacher_ids)

        mapping = []
        used = {}

        for x in ids:
            if x not in used:
                used[x] = 0

            matches = np.where(teacher_ids == x)[0]

            idx = matches[used[x]]
            mapping.append(idx)

            used[x] += 1

        return np.array(mapping, dtype=int)

    def resize_and_center_crop(self, image, target_h, target_w):
        h, w = image.shape[:2]  # 1440, 2560-->392 518

        scale = max(target_w / w, target_h / h)  #
        scale_w = target_w / w
        scale_h = target_h / h
        if scale_w >= scale_h:
            new_h, new_w = int(h * scale), target_w
        else:
            new_h, new_w = target_h, int(w * scale)
        # print('scale',scale)
        # print('target_w / w', target_w / w)
        # print('target_h / h', target_h / h)

        # new_w = int(w * scale)
        # new_h = int(h * scale)
        # print('new_w', new_w)
        # print('new_h', new_h)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2

        cropped = resized[start_y:start_y + target_h, start_x:start_x + target_w]

        return cropped
