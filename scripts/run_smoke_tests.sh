#!/usr/bin/env bash
set -euo pipefail

export MPLBACKEND=Agg

uv run python scripts/write_smoke_data.py
uv run python preprocess_oco2.py \
  --months 1701 \
  --components screen geometry state measured modeled wavelength \
  --output-dir outputs/smoke_preprocess \
  --smoke-test \
  --overwrite
uv run python train_forward.py \
  --band o2 \
  --sample_new \
  --smoke_test \
  --epochs 1 \
  --max_rows 16 \
  --no_aim \
  --device cpu \
  --output_dir outputs/smoke_forward_o2
uv run python train_inverse.py \
  --smoke_test \
  --epochs 1 \
  --max_rows 16 \
  --no_aim \
  --device cpu \
  --output_dir outputs/smoke_inverse
uv run python scripts/sanity_check_artifacts.py --allow-missing
