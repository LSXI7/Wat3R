import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path


class UnderwaterDepthDataset(Dataset):
    def __init__(self, gt_root,
                 nums: int = 100,
                 skip=0,
                 shuffle_in_chunk: bool = True,
                 seed: int = 42,

                 ):

        self.min_depth = 1e-3
        self.max_depth = 30
        self.disp_name = 'flsea_stereo'
        self.filename_ls_path = gt_root
        self.nums = int(nums) if nums is not None else 0
        self.skip = int(skip) if skip is not None else 0
        if self.skip > 0:
            self.shuffle_in_chunk = bool(shuffle_in_chunk)
        else:
            self.shuffle_in_chunk = False
        self.seed = int(seed)

        # key: (scene, obj) -> list[(scene, obj, img_path, depth_path)]
        self.data_list = {}

        for scene in os.listdir(gt_root):
            scene_root = os.path.join(gt_root, scene)
            print('scene_root', scene_root)
            if not os.path.isdir(scene_root):
                print('no scene in {}'.format(scene))
                continue

            for obj in os.listdir(scene_root):
                obj_root = os.path.join(scene_root, obj)

                imgs_both_root = os.path.join(obj_root, "imgs")

                depth_both_root = os.path.join(obj_root, "depth")

                if not os.path.isdir(imgs_both_root) or not os.path.isdir(depth_both_root):
                    print('no file in {}'.format(obj))
                    continue

                for camera in os.listdir(imgs_both_root):
                    camera_obj = os.path.join(obj, camera)
                    key = (scene, camera_obj)
                    self.data_list.setdefault(key, [])

                    imgs_root = os.path.join(imgs_both_root, camera)
                    print('imgs_root', imgs_root)
                    depth_root = os.path.join(depth_both_root, camera)
                    print('depth_root', depth_root)
                    for root, _, files in os.walk(imgs_root):
                        files = [f for f in files if
                                 f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]
                        files = sorted(files, key=self._natural_key)

                        for f in files:
                            img_path = os.path.join(root, f)

                            rel = os.path.relpath(img_path, imgs_root)
                            rel_dir = os.path.dirname(rel)
                            filename = os.path.basename(rel)
                            name, _ = os.path.splitext(filename)
                            depth_filename = f"{name}_abs_depth.tif"
                            depth_path = os.path.join(depth_root, rel_dir, depth_filename)

                            if os.path.exists(depth_path):
                                self.data_list[key].append((scene, camera_obj, img_path, depth_path))

        for key in list(self.data_list.keys()):
            items = self.data_list[key]
            items = sorted(items, key=lambda x: self._natural_key(os.path.basename(x[2])))
            self.data_list[key] = items

        # ✅ 构建 index：一个 idx -> 一个 chunk
        # index item: (scene, obj, chunk_id, start, end)
        self.index = []
        total_images = 0
        keys_sorted = sorted(self.data_list.keys())

        for idx, (scene, obj) in enumerate(keys_sorted):
            items = self.data_list[(scene, obj)]
            n = len(items)
            total_images += n
            if n == 0:
                continue

            if self.nums and self.nums > 0 and n > self.nums:
                chunk_id = 0
                for s in range(0, n, self.nums):
                    e = min(s + self.nums, n)
                    self.index.append((scene, obj, chunk_id, s, e))
                    chunk_id += 1
            else:
                self.index.append((scene, obj, 0, 0, n))

        print(
            f'get {total_images} images, {len(keys_sorted)} (scene,obj) groups -> {len(self.index)} chunks (nums={self.nums})')

    def __len__(self):
        return len(self.index)


    def _get_chunk_items(self, idx):
        """
        获取第 idx 个 batch/chunk 的 items。
        这里和 __getitem__ 保持一致：支持 skip 和 shuffle_in_chunk。
        """
        scene, obj,chunk_id, start, end = self.index[idx]
        items = self.data_list[(scene, obj)][start:end]

        stride = self.skip + 1

        if self.skip > 0:
            items = items[::stride]

        if self.shuffle_in_chunk and len(items) > 1:
            rng = np.random.default_rng(self.seed + idx)
            perm = rng.permutation(len(items))
            items = [items[i] for i in perm]

        return scene, obj, items

    def save_batch_image_paths_txt(self, txt_path):
        """
        保存一个 txt 文件：
        每个 batch/chunk 一段，里面是该 batch 的 n 张图像路径。
        """
        os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)

        with open(txt_path, "w", encoding="utf-8") as f:
            for batch_idx in range(len(self)):
                scene, obj, items = self._get_chunk_items(batch_idx)

                image_paths = [item[2] for item in items]  # item = (scene, obj, img_path, depth_path)

                f.write(f"# batch={batch_idx}, scene={scene}, obj={obj}, n={len(image_paths)}\n")

                for img_path in image_paths:
                    f.write(img_path + "\n")

                f.write("\n")

    def __getitem__(self, idx):
        scene, obj, chunk_id, start, end = self.index[idx]
        items = self.data_list[(scene, obj)][start:end]

        stride = self.skip + 1
        # ✅ 1) stride 采样：skip=4 => stride=5 => 100/5=20（整除时刚好）
        if self.skip > 0:
            print('skip')
            items = items[::stride]

        # ✅ 2) 采样后随机打乱（让每次取出的序列顺序随机）
        if self.shuffle_in_chunk and len(items) > 1:
            print('shuffle')
            # 用 idx + torch.initial_seed() 做种子，兼容多 worker（不同 idx 也不同）
            rng = np.random.default_rng(self.seed + idx)  # 每个 idx(块) 固定打乱，可复现
            perm = rng.permutation(len(items))
            items = [items[i] for i in perm]

        scene_list = []
        obj_list = []
        depth_np_list = []
        depth_ts_list = []
        image_path_list = []
        depth_path_list = []
        valid_mask_list = []

        for scene_i, obj_i, img_path, depth_path in tqdm(items, desc="Processing", total=len(items)):
            try:
                depth = Image.open(depth_path)
                depth_np = np.array(depth, dtype=np.float32)
                if np.max(depth_np) == 0:
                    continue
                depth_ts = torch.from_numpy(depth_np)
                valid_mask = self._get_valid_mask(depth_ts)

                scene_list.append(scene_i)
                obj_list.append(obj_i)
                depth_np_list.append(depth_np)
                depth_ts_list.append(depth_ts)
                image_path_list.append(img_path)
                depth_path_list.append(depth_path)
                valid_mask_list.append(valid_mask)
            except Exception as e:
                with open("failed.txt", "a") as f:
                    f.write(f"[ERROR] scene={scene_i}, obj={obj_i}, depth={depth_path}, error={str(e)}\n")
                continue

        uid = f"{scene}__{obj}__{chunk_id:05d}__{start:06d}-{end:06d}"

        return {
            # "uid": uid,                 # ✅ 唯一标识（同scene/obj分块也不冲突）
            "scene": scene_list,  # ✅ 唯一值
            "obj": obj_list,  # ✅ 唯一值
            # "chunk_id": chunk_id,       # ✅ 第几个块
            # "range": (start, end),      # ✅ 在该(scene,obj)序列中的范围

            "depth_np": depth_np_list,  # list[np.ndarray]
            "depth_ts": depth_ts_list,  # list[torch.Tensor]
            "image_path": image_path_list,  # list[str]
            "depth_path": depth_path_list,  # list[str]
            "valid_mask": valid_mask_list,  # list[torch.BoolTensor]
        }

    def _get_valid_mask(self, depth: torch.Tensor):
        return torch.logical_and((depth > self.min_depth), (depth < self.max_depth)).bool()

    def create_dataloader(self, batch_size=1, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers)

    @staticmethod
    def _natural_key(s: str):
        # 自然排序：按数字大小排序
        parts = re.split(r'(\d+)', s)
        out = []
        for p in parts:
            out.append(int(p) if p.isdigit() else p.lower())
        return out

    # ------- 你原来的均匀采样工具保持不变 -------
    def even_indices(self, L: int, k: int):
        if k >= L:
            return list(range(L))
        if k <= 0:
            return []
        if k == 1:
            return [L // 2]
        return [round(i * (L - 1) / (k - 1)) for i in range(k)]

    def even_sample(self, seq, k):
        idx = self.even_indices(len(seq), k)
        return [seq[i] for i in idx]
