# Wat3R Evaluation

This directory contains the official Wat3R evaluation code for monocular depth,
multiview depth, multiview point-cloud reconstruction, and multiview camera pose
estimation.

## Data Structure

<p></p>
<details>
<summary><b>Organizing the Datasets</b></summary>

We recommend organizing the datasets in the following folder structure:

```text
evaluation/datasets/
├── FLSea_VI/
│   ├── canyons/
│   │   └── <sequence>/{imgs,depth}/
│   └── red_sea/
│       └── <sequence>/{imgs,depth}/
├── seathru/
│   └── <scene>/
│       └── <sequence>/{linearPNG,depth}/
├── flsea_stereo/
│   └── <scene>/
│       └── <sequence>/
│           ├── imgs/{LFT,RGT}/
│           └── depth/{LFT,RGT}/
├── SQUID/
│   └── <location>/
│       └── image_set_XX/
│           ├── LFT_*resizedUndistort.tif
│           ├── RGT_*resizedUndistort.tif
│           └── xyzPoints.mat
├── SeathruNeRF/
    └── <scene>/
        ├── Images_wb/ or images_wb/
        └── sparse/1/images.txt
└── water3D/
    └── <scene>/
        ├── images/
        └── output/
            ├── sparse/
            ├── stereo/depth_maps/
            └── fused.ply
```

Example symbolic links:

```bash
ln -s /path/to/FLSea_VI evaluation/datasets/FLSea_VI
ln -s /path/to/seathru evaluation/datasets/seathru
ln -s /path/to/flsea_stereo evaluation/datasets/flsea_stereo
ln -s /path/to/SQUID evaluation/datasets/SQUID
ln -s /path/to/SeathruNeRF_dataset evaluation/datasets/SeathruNeRF
ln -s /path/to/water3D evaluation/datasets/water3D
```

</details>

## Monocular Depth

Supported datasets: `flsea_vi`, `seathru`, `flsea_stereo`, and `squid`.
For the stereo datasets, the left and right images are evaluated independently.

```bash
python evaluation/evaluate_depth.py \
  --mode mono \
  --dataset flsea_vi \
  --checkpoint /path/to/wat3r.pt \
  --output-dir evaluation/outputs
  # --save-figs
```

## Multiview Depth

Supported datasets: `seathru_full` and `flsea_stereo_full`.

```bash
python evaluation/evaluate_depth.py \
  --mode multiview \
  --dataset seathru_full \
  --checkpoint /path/to/wat3r.pt \
  --output-dir evaluation/outputs \
  --skip 9
  # --save-figs
```

`--skip N` keeps one frame every `N+1` frames. Omit it to evaluate all frames in
a sequence chunk.

## Multiview Camera Pose

```bash
python evaluation/evaluate_pose.py \
  --dataset seathru_nerf \
  --checkpoint /path/to/wat3r.pt \
  --output-dir evaluation/outputs 
  # --save-figs
```

## Multiview Point Cloud

The Water3D point-cloud evaluation samples `--num-views` frames from each scene,
aligns the predicted point cloud to the COLMAP depth-derived point cloud with a
single weighted similarity transform, and reports accuracy, completion, and
normal consistency.

```bash
python evaluation/evaluate_point.py \
  --checkpoint /path/to/wat3r.pt \
  --dataset-root evaluation/datasets/water3D \
  --output-dir evaluation/outputs \
  --num-views 20 \
  --test-mode 2
  # --scene cv_1000       # can be repeated
  # --save-o3d
```

`--test-mode 2` evaluates point clouds reconstructed from the depth and camera
heads. `--test-mode 0` evaluates the point head directly.

## Run All Benchmarks

This reproduces the enabled commands from the original `test_one_model.sh`:

```bash
bash evaluation/run_all.sh /path/to/wat3r.pt evaluation/outputs
```

Use `--dataset-root /custom/path` on an individual command when a dataset is not
stored under `evaluation/datasets/`.
