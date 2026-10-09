#!/usr/bin/env bash
set -euo pipefail

cd /root/pignn
mkdir -p out logs
exec /root/miniconda3/envs/pignn/bin/python -u controlled_error_geometry.py \
  --phase validation --ranks 4 8 16 32 64 --batch 16 \
  --parquet data/GBnetwork.parquet --out out/controlled_geometry_validation.json \
  --pignn ckpt/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt \
  --gridsfm ckpt/pfv2_gridsfm_GBnetwork_b26_best.pt \
  --graphkit ckpt/pfv2_graphkit_GBnetwork_b52_best.pt \
  --lumina ckpt/pfv2_lumina_GBnetwork_b32_best.pt
