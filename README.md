# CO2-Profile-Retrieval

This repository contains the cleaned release version of the code used for the
paper revision. It keeps one final code path only:

- `train_forward.py`: trains the band-wise surrogate forward model.
- `train_inverse.py`: trains the probabilistic inverse retrieval model.
- `preprocess_oco2.py`: exports OCO-2 Level-2 diagnostic/full-physics HDF5
  files to the parquet files consumed by the training scripts.
- `oco2_surrogate/`: shared model, loss, preprocessing, data-loading, plotting,
  and forward-model wrapper utilities.
- `los_physics/`: differentiable line-of-sight radiance baseline used by the
  surrogate forward model and by inverse-model augmentation.

The original working-tree versions were selected from the plotting provenance
and renamed to the stable names above.

## Environment

This release is configured for `uv`:

```bash
uv sync
```

All commands below are written as `uv run ...` so they use the locked project
environment.

## Data and External Files

The scripts expect preprocessed OCO-2 parquet files and ABSCO/solar/reference
files to be available locally. Set these environment variables before running:

```bash
export OCO2_DATA_DIR=/path/to/LOS_OCO2_DATA
export OCO2_ABSCO_DIR=/path/to/absco
export OCO2_CONSTANT_DIR=/path/to/OCO2
export OCO2_L2DIA_DIR=/path/to/OCO2/L2DiaND
# Optional, if local preprocessed files use non-default names:
export OCO2_STATE_FILE_TEMPLATE='df_state_{yymm}.parquet'
export OCO2_LOS_FILE_TEMPLATE='df_los_{yymm}.parquet'
```

`OCO2_DATA_DIR` should contain files named like:

- `df_screen_YYMM.parquet`
- `df_geometry_YYMM.parquet`
- `df_state_YYMM.parquet`
- `df_los_YYMM.parquet`
- `df_modeled_YYMM.parquet`
- `df_measured_YYMM.parquet`

`OCO2_ABSCO_DIR` should contain `co2_v51.hdf`, `h2o_v51.hdf`,
`o2_v51.hdf`, and `oco_solar_model.h5`.

`OCO2_CONSTANT_DIR` should contain `index.json`, `L1BScND/`, and `L2DiaND/`
for initializing the LOS model constants.

Model weights, paper-test inputs, and real-data reproduction fixtures are not
included in this GitHub repository. They are available from the authors on
reasonable request. After receiving them, place model weights under
`artifacts/` and optional test/reproduction inputs under `examples/`.

## Preprocessing

Run from this project directory. For example, to export January and February
2017 from OCO-2 Level-2 diagnostic/full-physics HDF5 files:

```bash
python preprocess_oco2.py --months 1701 1702
```

By default this exports:

- `df_screen_YYMM.parquet`
- `df_geometry_YYMM.parquet`
- `df_state_YYMM.parquet`
- `df_measured_YYMM.parquet`
- `df_modeled_YYMM.parquet`
- `df_wavelength_YYMM.parquet`
- `df_los_YYMM.parquet`

To skip the LOS baseline step, omit `los` from `--components`:

```bash
python preprocess_oco2.py --months 1701 --components screen geometry state measured modeled wavelength
```

The LOS step uses `los_physics/` and therefore requires `OCO2_ABSCO_DIR` and
`OCO2_CONSTANT_DIR` in addition to the L2 diagnostic files.

A tiny non-LOS preprocessing smoke test is available without external OCO-2
files:

```bash
uv run python preprocess_oco2.py \
  --months 1701 \
  --components screen geometry state measured modeled wavelength \
  --output-dir outputs/smoke_preprocess \
  --smoke-test \
  --overwrite
```

Full LOS preprocessing remains an integration check because it needs the ABSCO,
solar, and OCO-2 constant files listed above.

## Training

Run from this project directory.

First generate the shared forward-model data and train the O2 model:

```bash
python train_forward.py --band o2 --sample_new
```

Then train the other bands using the shared data saved under
`status/train_forward_o2`:

```bash
python train_forward.py --band weak_co2
python train_forward.py --band strong_co2
```

The inverse model uses the trained forward-model checkpoints for CO2-shift
augmentation when `--sample_new` is used:

```bash
python train_inverse.py --sample_new
python train_inverse.py
```

By default, the inverse script looks for forward checkpoints
`149 174 199 224 249` under `status/train_forward_o2`,
`status/train_forward_weak_co2`, and `status/train_forward_strong_co2`.

For a CPU-safe smoke run:

```bash
uv run bash scripts/run_smoke_tests.sh
```

This writes synthetic fixtures to `examples/smoke_data/`, smoke outputs under
`outputs/`, and exercises preprocessing, forward training, inverse training,
and release-artifact path checks. On a typical workstation this should finish
in under a minute.

## Model Weights and Test Data

Model weights and paper-test data are distributed on request rather than stored
in this repository. The expected local layout is:

- `artifacts/forward/{band}/mlp_model_best.pth`
- `artifacts/forward/{band}/model_config.pkl`
- shared `artifacts/forward/scalers.pth`, `indices.pth`, and
  `df_retrieved_columns.pth`
- `artifacts/inverse/inverse_model_best_test.pth`
- `artifacts/inverse/scalers.pth`
- optional `examples/figure2_input/`
- optional `examples/2021_measured_sample1000/`

If you have access to the author's training output tree, the helper below can
import final pre-finetune model artifacts into the expected local paths:

```bash
uv run python scripts/import_release_artifacts.py \
  --source-root /mnt/d/chenwei/SS_OCO2/status
```

The importer expects:

- forward models from `new_los_train_09_final_o2`,
  `new_los_train_09_final_weak_co2`, and
  `new_los_train_09_final_strong_co2`
- inverse model from `uncert_train_17_final`

It writes:

- `artifacts/forward/{band}/mlp_model_best.pth`
- `artifacts/forward/{band}/model_config.pkl`
- shared `artifacts/forward/scalers.pth`, `indices.pth`, and
  `df_retrieved_columns.pth`
- `artifacts/inverse/inverse_model_best_test.pth`
- `artifacts/inverse/scalers.pth`
- `artifacts/MANIFEST.json` with source paths, file sizes, SHA256 hashes, and
  import time

These local artifact and example directories are ignored by Git.

## Paper-Result Reproduction

The repository contains scripts for two real-data reproduction checks. The
required model weights and input fixtures are available from the authors on
reasonable request.

Regenerate the manuscript Figure 2 radiance comparison from the requested
single-sounding source data and forward weights:

```bash
uv run python scripts/reproduce_figure2.py \
  --device cpu \
  --output-dir outputs/reproduction
```

The source data should live under `examples/figure2_input/` and include the original
geometry/state/LOS/modeled/measured/wavelength row for sounding
`2022120508103571`, plus the historical per-band radiance PDF provenance files.
The script writes `outputs/reproduction/figure2_radiance_2022120508103571.png`
and a metrics JSON file, plus
`outputs/reproduction/figure2_emulated_radiance_2022120508103571.parquet` with
the recomputed emulated radiance values.

Run the released inverse model on a random 1000-sounding 2021 measured-radiance
sample:

```bash
uv run python scripts/run_2021_measured_scatter.py \
  --device cpu \
  --sample-size 1000 \
  --seed 42 \
  --output-dir outputs/reproduction \
  --cache-input-dir examples/2021_measured_sample1000
```

This command reads real 2021 OCO-2 parquet inputs from
`/mnt/d/chenwei/LOS_OCO2_DATA`, uses `df_measured_21MM.parquet` radiances, runs
`artifacts/inverse/inverse_model_best_test.pth`, computes XCO2 in physical units,
and writes the sampled real input rows into `examples/2021_measured_sample1000/`.
After that cache exists, the same result can be regenerated without rereading the
external yearly parquet files:

```bash
uv run python scripts/run_2021_measured_scatter.py \
  --device cpu \
  --output-dir outputs/reproduction \
  --cache-input-dir examples/2021_measured_sample1000 \
  --use-cached-input
```

The script writes:

- `outputs/reproduction/scatter_2021_measured_sample1000.png`
- `outputs/reproduction/scatter_2021_measured_sample1000_results.parquet`
- `outputs/reproduction/scatter_2021_measured_sample1000_metrics.json`

With seed 42, the current 1000-point measured-radiance run gives RMSE
`1.143 ppm` and ME `0.080 ppm`.

`backtest_release.py` and `scripts/run_smoke_tests.sh` are smoke-path checks for
software wiring only. They use tiny synthetic fixtures and should not be cited
as paper-result reproduction.

## Release Verification

Expected end-to-end release checks:

```bash
uv sync
uv run bash scripts/run_smoke_tests.sh
# The following checks require requested artifacts and real-data fixtures:
uv run python scripts/reproduce_figure2.py --device cpu --output-dir outputs/reproduction
uv run python scripts/run_2021_measured_scatter.py --device cpu --sample-size 1000 --seed 42 --output-dir outputs/reproduction --cache-input-dir examples/2021_measured_sample1000
uv run python scripts/run_2021_measured_scatter.py --device cpu --output-dir outputs/reproduction --cache-input-dir examples/2021_measured_sample1000 --use-cached-input
```

## Main Dependencies

Python packages used by the final code path include:

- `torch`
- `numpy`
- `pandas`
- `pyarrow`
- `h5py`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `numba`
- `tqdm`
- `aim`
