import os
import re
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class UnderwaterDepthDataset(Dataset):
    def __init__(self, gt_root,
                 nums: int = 100,
                 skip=0,
                 shuffle_in_chunk: bool = True,
                 seed: int = 42,

                 ):

        self.min_depth = 1e-3
        self.max_depth = 30
        self.disp_name = 'seathru'
        self.filename_ls_path = gt_root

        self.nums = int(nums) if nums is not None else 0
        self.skip = int(skip) if skip is not None else 0
        if self.skip > 0:
            self.shuffle_in_chunk = bool(shuffle_in_chunk)
        else:
            self.shuffle_in_chunk = False
        self.seed = int(seed)

        # key: (scene, obj)  value: list[(scene, obj, img_path, depth_path)]
        self.data_list = {}

        for scene in os.listdir(gt_root):
            scene_root = os.path.join(gt_root, scene)
            if not os.path.isdir(scene_root):
                print('no scene in {}'.format(scene))
                continue

            for obj in os.listdir(scene_root):
                obj_root = os.path.join(scene_root, obj)
                imgs_root = os.path.join(obj_root, "linearPNG")

                if obj != 'D4':
                    depth_root = os.path.join(obj_root, "depth")
                else:
                    depth_root = os.path.join(obj_root, "depth_resized")

                if not os.path.isdir(imgs_root) or not os.path.isdir(depth_root):
                    print('no file in {}'.format(obj))
                    continue

                key = (scene, obj)
                self.data_list.setdefault(key, [])

                for root, _, files in os.walk(imgs_root):
                    files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]

                    files = sorted(files, key=self._natural_key)

                    for f in files:
                        img_path = os.path.join(root, f)
                        rel = os.path.relpath(img_path, imgs_root)
                        rel_dir = os.path.dirname(rel)

                        filename = os.path.basename(rel)
                        name, _ = os.path.splitext(filename)

                        # T_S02951.png --> depthT_S02951.tif
                        depth_filename = 'depth' + name + '.tif'
                        depth_path = os.path.join(depth_root, rel_dir, depth_filename)

                        if os.path.exists(depth_path):
                            self.data_list[key].append((scene, obj, img_path, depth_path))

        # Build the index table: each index maps to one chunk.
        # self.index = list of (scene, obj, start, end)
        self.index = []
        total_images = 0

        keys = sorted(self.data_list.keys())
        for (scene, obj) in keys:
            items = self.data_list[(scene, obj)]
            # Sort within each group by image filename.
            items = sorted(items, key=lambda x: self._natural_key(os.path.basename(x[2])))
            self.data_list[(scene, obj)] = items

            n = len(items)
            total_images += n

            if self.nums and self.nums > 0 and n > self.nums:
                for s in range(0, n, self.nums):
                    e = min(s + self.nums, n)
                    self.index.append((scene, obj, s, e))
            else:
                self.index.append((scene, obj, 0, n))

        print(
            f'get {total_images} images, {len(keys)} (scene,obj) groups -> {len(self.index)} chunks (nums={self.nums})')

    def __len__(self):
        return len(self.index)

    def _get_chunk_items(self, idx):
        """
        Return the items for a batch/chunk index.
        This mirrors __getitem__, including skip and shuffle_in_chunk handling.
        """
        scene, obj, start, end = self.index[idx]
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
        Save one text block per batch/chunk with its image paths.
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
        scene, obj, start, end = self.index[idx]
        if obj == 'D5':
            self.min_depth = 1
        else:
            self.min_depth = 1e-3
        items = self.data_list[(scene, obj)][start:end]

        stride = self.skip + 1
        # Strided sampling: skip=4 means stride=5.
        if self.skip > 0:
            print('skip')
            items = items[::stride]

        # Optionally shuffle the sampled chunk.
        if self.shuffle_in_chunk and len(items) > 1:
            print('shuffle')
            rng = np.random.default_rng(self.seed + idx)
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
            depth = Image.open(depth_path)
            depth_np = np.array(depth, dtype=np.float32)
            depth_ts = torch.from_numpy(depth_np)
            valid_mask = self._get_valid_mask(depth_ts)

            scene_list.append(scene_i)
            obj_list.append(obj_i)
            depth_np_list.append(depth_np)
            depth_ts_list.append(depth_ts)
            image_path_list.append(img_path)
            depth_path_list.append(depth_path)
            valid_mask_list.append(valid_mask)

        return {
            "scene": scene_list,
            "obj": obj_list,
            "depth_np": depth_np_list,
            "depth_ts": depth_ts_list,
            "image_path": image_path_list,
            "depth_path": depth_path_list,
            "valid_mask": valid_mask_list,
        }

    def _get_valid_mask(self, depth: torch.Tensor):
        return torch.logical_and((depth > self.min_depth), (depth < self.max_depth)).bool()

    def create_dataloader(self, batch_size=1, num_workers=0):
        # Chunks may have different lengths, so batch_size=1 is recommended.
        return DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    @staticmethod
    def _natural_key(s: str):
        """
        Natural sort key that compares embedded numbers as integers.
        """
        parts = re.split(r'(\d+)', s)
        out = []
        for p in parts:
            if p.isdigit():
                out.append(int(p))
            else:
                out.append(p.lower())
        return out
