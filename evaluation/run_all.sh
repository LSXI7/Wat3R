#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash evaluation/run_all.sh <checkpoint> <output_dir>"
  exit 1
fi

CHECKPOINT=$1
OUTPUT_DIR=$2

for dataset in flsea_vi seathru flsea_stereo squid; do
  python evaluation/evaluate_depth.py \
    --mode mono \
    --dataset "$dataset" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR"
done

for dataset in seathru_full flsea_stereo_full; do
  python evaluation/evaluate_depth.py \
    --mode multiview \
    --dataset "$dataset" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --skip 9

  python evaluation/evaluate_depth.py \
    --mode multiview \
    --dataset "$dataset" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR"
done

python evaluation/evaluate_point.py \
  --checkpoint "$CHECKPOINT" \
  --dataset-root evaluation/datasets/water3D \
  --output-dir "$OUTPUT_DIR" \
  --num-views 20 \
  --test-mode 2

python evaluation/evaluate_pose.py \
  --dataset seathru_nerf \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR"
