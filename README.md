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
