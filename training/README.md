# Wat3R Training

This document describes how to train Wat3R with the default training config.

Default config:

```text
training/config/wat3r_training.yaml
```

Wat3R training is initialized from the original VGGT checkpoint. Download the
VGGT pretrained weights first, then set the path in the config:

```yaml
checkpoint:
  resume_checkpoint_path: /path/to/vggt.pt
```

## Installation

Run from the repository root:

```bash
conda activate py
pip install -e .
pip install -r training/requirements_train.txt
```

## Data Preparation

Create dataset symlinks under:

```text
training/datasets/
```

The symlink names should match the paths used in `training/config/wat3r_training.yaml`.

## Training

Run from the repository root:

```bash
torchrun --nproc_per_node=4 --nnodes=1 training/launch.py --config wat3r_training.yaml
```

Or run from the `training/` directory:

```bash
cd training
torchrun --nproc_per_node=4 --nnodes=1 launch.py --config wat3r_training.yaml
```
