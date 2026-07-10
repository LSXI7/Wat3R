import os
import re
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class UnderwaterDepthDataset(Dataset):
    def __init__(self, gt_root,
                 nums: int = 100,  # ✅ 新增：每份最多nums张，0/None表示不分块
                 debug=False, debug2=False, no_water=False, undistort=False, debug3=False,
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

                # 遍历图片
                for root, _, files in os.walk(imgs_root):
                    # ✅ 只处理图片文件（可按需扩展）
                    files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]

                    # ✅ 按名字“自然排序”（包含数字排序）
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

        # ✅ 构建“索引表”：一个 idx 对应一个 chunk
        # self.index = list of (scene, obj, start, end)
        self.index = []
        total_images = 0

        # ✅ scene,obj 固定顺序
        keys = sorted(self.data_list.keys())
        for (scene, obj) in keys:
            items = self.data_list[(scene, obj)]
            # ✅ items 内部再按 image 文件名排序（双保险：防止不同root拼接顺序影响）
            items = sorted(items, key=lambda x: self._natural_key(os.path.basename(x[2])))
            self.data_list[(scene, obj)] = items

            n = len(items)
            total_images += n

            if self.nums and self.nums > 0 and n > self.nums:
                # 分块
                for s in range(0, n, self.nums):
                    e = min(s + self.nums, n)
                    self.index.append((scene, obj, s, e))
            else:
                # 不分块
                self.index.append((scene, obj, 0, n))

        print(
            f'get {total_images} images, {len(keys)} (scene,obj) groups -> {len(self.index)} chunks (nums={self.nums})')

    def __len__(self):
        return len(self.index)

    def _get_chunk_items(self, idx):
        """
        获取第 idx 个 batch/chunk 的 items。
        这里和 __getitem__ 保持一致：支持 skip 和 shuffle_in_chunk。
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
        scene, obj, start, end = self.index[idx]
        if obj == 'D5':
            self.min_depth = 1
        else:
            self.min_depth = 1e-3
        items = self.data_list[(scene, obj)][start:end]  # ✅ 这一份最多 nums 张

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

        # tqdm 建议只在调试用；训练会很慢
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
        # ✅ 由于每个chunk长度可能不同，建议 batch_size=1
        return DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    @staticmethod
    def _natural_key(s: str):
        """
        自然排序 key：把字符串中的数字提取出来按 int 排序
        例：T_S02951.png < T_S02952.png；T_S9.png < T_S10.png
        """
        parts = re.split(r'(\d+)', s)
        out = []
        for p in parts:
            if p.isdigit():
                out.append(int(p))
            else:
                out.append(p.lower())
        return out
