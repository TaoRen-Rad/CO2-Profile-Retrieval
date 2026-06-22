#!/usr/bin/env python
"""Export OCO-2 full-physics diagnostic files to training parquet files.

The training scripts in this directory expect month-level parquet files with
names such as ``df_geometry_1701.parquet`` and ``df_state_1701.parquet``.
This script builds those files from OCO-2 Level-2 diagnostic/full-physics HDF5
files and can optionally compute the line-of-sight baseline radiance used by
the surrogate forward model.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm

from los_physics.constant import sh2vmr
from los_physics.predict import predict as predict_los
from oco2_surrogate.release import load_smoke_dataframes


BAND_NAMES = ["o2", "weak_co2", "strong_co2"]
RADIANCE_COLUMNS = [f"{band}_{i}" for band in BAND_NAMES for i in range(1016)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess OCO-2 L2 diagnostic HDF5 files to parquet files."
    )
    parser.add_argument(
        "--l2dia-dir",
        default=os.environ.get("OCO2_L2DIA_DIR", "data/OCO2/L2DiaND"),
        help="Root containing yearly L2DiaND directories, e.g. .../L2DiaND/2017/*.h5.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OCO2_DATA_DIR", "data/LOS_OCO2_DATA"),
        help="Directory for output parquet files.",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        required=True,
        help="Year-month values such as 1701 1702 or 201701 201702.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=["screen", "geometry", "state", "measured", "modeled", "wavelength", "los"],
        choices=["screen", "geometry", "state", "measured", "modeled", "wavelength", "signal", "los"],
        help="Components to export.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device used for LOS generation when --components includes los.",
    )
    parser.add_argument(
        "--los-batch-size",
        type=int,
        default=128,
        help="Batch size used inside LOS generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing component parquet files.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Write tiny synthetic non-LOS parquet outputs instead of reading L2DiaND files.",
    )
    return parser.parse_args()


def normalize_yymm(value: str) -> str:
    value = value.strip()
    if len(value) == 6 and value.startswith("20"):
        return value[2:]
    if len(value) == 4:
        return value
    raise ValueError(f"Expected YYMM or YYYYMM, got {value!r}")


def find_files_for_yymm(l2dia_dir: str | Path, yymm: str) -> list[str]:
    """Find L2DiaND files by observation date embedded in the filename."""
    year = f"20{yymm[:2]}"
    candidates = glob.glob(str(Path(l2dia_dir) / year / "*.h5"))
    pattern = re.compile(r"_(\d{6})_B")

    matched_files = []
    for fpath in candidates:
        match = pattern.search(Path(fpath).name)
        if match and match.group(1)[:4] == yymm:
            matched_files.append(fpath)
    return sorted(matched_files)


@jit(nopython=True, cache=True)
def fill_bands_fast(radiance, wavelength, sample_indexes, wl_ranges, n_sounding):
    output_data = np.zeros((n_sounding, 3, 1016))
    for band_idx in range(3):
        wl_min = wl_ranges[band_idx][0]
        wl_max = wl_ranges[band_idx][1]
        for sounding_idx in range(n_sounding):
            for spec_idx in range(wavelength.shape[1]):
                wl = wavelength[sounding_idx, spec_idx]
                if wl_min <= wl <= wl_max:
                    sample_idx = int(sample_indexes[sounding_idx, spec_idx])
                    if 0 <= sample_idx < 1016:
                        output_data[sounding_idx, band_idx, sample_idx] = radiance[
                            sounding_idx, spec_idx
                        ]
    return output_data


def load_screen_data(file: h5py.File, idx: np.ndarray, sounding_id: np.ndarray) -> pd.DataFrame:
    pre_name = "/RetrievalGeometry/retrieval_"
    screen_names = ["latitude", "longitude"]
    values = np.zeros((len(sounding_id), len(screen_names)))
    for col, name in enumerate(screen_names):
        values[:, col] = file[pre_name + name][idx]

    df_location = pd.DataFrame(values, columns=screen_names, index=sounding_id)
    co2_profile_uncert = file["/RetrievalResults/co2_profile_uncert"][idx]
    df_uncert = pd.DataFrame(
        co2_profile_uncert,
        columns=[f"co2_profile_uncert_{i:02d}" for i in range(20)],
        index=sounding_id,
    )
    return pd.concat([df_location, df_uncert], axis=1)


def load_geometry_data(file: h5py.File, idx: np.ndarray, sounding_id: np.ndarray) -> pd.DataFrame:
    pre_name = "/RetrievalGeometry/retrieval_"
    geometry_names = [
        "azimuth",
        "solar_azimuth",
        "solar_zenith",
        "zenith",
        "polarization_angle",
        "altitude",
        "relative_velocity",
        "slope",
        "solar_distance",
        "solar_relative_velocity",
    ]

    values = np.zeros((len(sounding_id), len(geometry_names)))
    for col, name in enumerate(geometry_names):
        values[:, col] = file[pre_name + name][idx]

    values[:, 0] = values[:, 0] - values[:, 1]
    values = np.delete(values, 1, axis=1)
    geometry_names.pop(1)
    geometry_names[0] = "relative_azimuth"

    cos_sin = np.zeros((len(sounding_id), 8))
    for i in range(4):
        cos_sin[:, 2 * i] = np.cos(np.radians(values[:, i]))
        cos_sin[:, 2 * i + 1] = np.sin(np.radians(values[:, i]))

    values = np.concatenate([cos_sin, values[:, 4:]], axis=1)
    columns = []
    for i in range(4):
        columns.append(f"cos_{geometry_names[i]}")
        columns.append(f"sin_{geometry_names[i]}")
    columns.extend(geometry_names[4:])

    return pd.DataFrame(values, columns=columns, index=sounding_id)


def _read_result(file: h5py.File, name: str, idx: np.ndarray) -> np.ndarray:
    return np.asarray(file[f"/RetrievalResults/{name}"][idx])


def load_state_data(file: h5py.File, idx: np.ndarray, sounding_id: np.ndarray) -> pd.DataFrame:
    df_aerosols = []
    for suffix in ["_apriori", ""]:
        aerosol = np.asarray(file[f"/AerosolResults/aerosol_param{suffix}"][idx])
        for i in range(8):
            bad = aerosol[:, i, 0] < -1e5
            aerosol[bad, i, 0] = -1e5
            aerosol[bad, i, 1] = 0.0
            aerosol[bad, i, 2] = 0.0
        aerosol[:, :, 0] = np.exp(-np.exp(aerosol[:, :, 0]))
        columns = [
            f"aerosol_{field}_{i}{suffix}"
            for i in range(8)
            for field in ["trans", "mu", "sigma"]
        ]
        df_aerosols.append(
            pd.DataFrame(aerosol.reshape(aerosol.shape[0], -1), columns=columns, index=sounding_id)
        )
    df_aerosol = pd.concat(df_aerosols, axis=1)

    brdf_names = ["weight", "weight_slope", "weight_quadratic"]
    brdf_values = np.zeros((len(sounding_id), len(brdf_names) * len(BAND_NAMES) * 2))
    brdf_columns = []
    col = 0
    for prefix in ["apriori_", ""]:
        for band in BAND_NAMES:
            for name in brdf_names:
                brdf_columns.append(f"brdf_{name}_{prefix}{band}")
                brdf_values[:, col] = file[f"/BRDFResults/brdf_{name}_{prefix}{band}"][idx]
                col += 1
    df_brdf = pd.DataFrame(brdf_values, columns=brdf_columns, index=sounding_id)

    dispersion_names = ["offset", "spacing"]
    dispersion_values = np.zeros((len(sounding_id), len(dispersion_names) * len(BAND_NAMES) * 2))
    dispersion_columns = []
    col = 0
    for prefix in ["apriori_", ""]:
        for name in dispersion_names:
            for band in BAND_NAMES:
                dispersion_columns.append(f"{name}_{prefix}{band}")
                dispersion_values[:, col] = file[
                    f"/DispersionResults/dispersion_{name}_{prefix}{band}"
                ][idx]
                col += 1

    retrieved_names = [
        "specific_humidity_profile_met",
        "h2o_scale_factor",
        "h2o_scale_factor_apriori",
        "co2_profile",
        "co2_profile_apriori",
        "temperature_profile_met",
        "vector_pressure_levels_met",
        "temperature_offset_fph",
        "temperature_offset_apriori_fph",
        "vector_pressure_levels",
        "vector_pressure_levels_apriori",
        "fluorescence_at_reference",
        "fluorescence_at_reference_apriori",
        "fluorescence_slope",
        "fluorescence_slope_apriori",
        "xco2_pressure_weighting_function",
        "xco2",
        "xco2_uncert",
    ]
    retrieved_names.extend(f"eof_{i}_scale_{band}" for i in range(1, 4) for band in BAND_NAMES)
    retrieved_names.extend(
        f"eof_{i}_scale_apriori_{band}" for i in range(1, 4) for band in BAND_NAMES
    )

    retrieved_arrays = []
    retrieved_columns = []
    for name in retrieved_names:
        data = _read_result(file, name, idx)
        if data.ndim == 1:
            retrieved_arrays.append(data.reshape(-1, 1))
            retrieved_columns.append(name)
        else:
            retrieved_arrays.append(data)
            retrieved_columns.extend(f"{name}_{i}" for i in range(data.shape[1]))

    df_state = pd.DataFrame(
        np.concatenate([dispersion_values, *retrieved_arrays], axis=1),
        columns=dispersion_columns + retrieved_columns,
        index=sounding_id,
    )
    df_state = pd.concat([df_state, df_aerosol, df_brdf], axis=1)
    df_state["outcome_flag"] = file["/RetrievalResults/outcome_flag"][idx]

    _append_interpolated_profiles(df_state)
    return df_state


def _append_interpolated_profiles(df_state: pd.DataFrame) -> None:
    pressure_met = df_state[[f"vector_pressure_levels_met_{i}" for i in range(72)]].values
    temp_met = df_state[[f"temperature_profile_met_{i}" for i in range(72)]].values
    humidity_met = df_state[[f"specific_humidity_profile_met_{i}" for i in range(72)]].values

    pressure_apriori = df_state[[f"vector_pressure_levels_apriori_{i}" for i in range(20)]].values
    pressure = df_state[[f"vector_pressure_levels_{i}" for i in range(20)]].values

    temp_offset = df_state["temperature_offset_fph"].values
    temp_offset_apriori = df_state["temperature_offset_apriori_fph"].values
    h2o_scale = df_state["h2o_scale_factor"].values
    h2o_scale_apriori = df_state["h2o_scale_factor_apriori"].values

    n_samples = len(df_state)
    temp_apriori = np.zeros((n_samples, 20))
    temp = np.zeros((n_samples, 20))
    h2o_apriori = np.zeros((n_samples, 20))
    h2o = np.zeros((n_samples, 20))

    for i in range(n_samples):
        temp_apriori[i] = (
            np.interp(pressure_apriori[i], pressure_met[i], temp_met[i]) + temp_offset_apriori[i]
        )
        temp[i] = np.interp(pressure[i], pressure_met[i], temp_met[i]) + temp_offset[i]
        h2o_apriori[i] = (
            sh2vmr(np.interp(pressure_apriori[i], pressure_met[i], humidity_met[i]))
            * h2o_scale_apriori[i]
        )
        h2o[i] = sh2vmr(np.interp(pressure[i], pressure_met[i], humidity_met[i])) * h2o_scale[i]

    drop_columns = (
        [f"temperature_profile_met_{i}" for i in range(72)]
        + [f"vector_pressure_levels_met_{i}" for i in range(72)]
        + ["temperature_offset_fph", "temperature_offset_apriori_fph"]
        + [f"specific_humidity_profile_met_{i}" for i in range(72)]
        + ["h2o_scale_factor", "h2o_scale_factor_apriori"]
    )
    df_state.drop(columns=drop_columns, inplace=True)
    df_state[[f"temperature_profile_{i}" for i in range(20)]] = temp
    df_state[[f"temperature_profile_apriori_{i}" for i in range(20)]] = temp_apriori
    df_state[[f"h2o_profile_apriori_{i}" for i in range(20)]] = h2o_apriori
    df_state[[f"h2o_profile_{i}" for i in range(20)]] = h2o


def load_spectral_data(file: h5py.File, idx: np.ndarray, sounding_id: np.ndarray) -> dict[str, pd.DataFrame]:
    modeled = file["/SpectralParameters/modeled_radiance"][idx]
    measured = file["/SpectralParameters/measured_radiance"][idx]
    wavelength = file["/SpectralParameters/wavelength"][idx]
    sample_indexes = file["/SpectralParameters/sample_indexes"][idx] - 1

    signal = np.column_stack(
        [
            file["/SpectralParameters/signal_o2_fph"][idx],
            file["/SpectralParameters/signal_weak_co2_fph"][idx],
            file["/SpectralParameters/signal_strong_co2_fph"][idx],
        ]
    )

    wl_ranges = np.array([[0.5, 1.0], [1.5, 1.7], [2.0, 2.3]])
    n_sounding = len(sounding_id)
    modeled = fill_bands_fast(modeled, wavelength, sample_indexes, wl_ranges, n_sounding)
    measured = fill_bands_fast(measured, wavelength, sample_indexes, wl_ranges, n_sounding)
    wavelength = fill_bands_fast(wavelength, wavelength, sample_indexes, wl_ranges, n_sounding)

    modeled[modeled == 0] = np.nan
    measured[measured == 0] = np.nan
    wavelength[wavelength == 0] = np.nan

    return {
        "measured": pd.DataFrame(measured.reshape(n_sounding, -1), columns=RADIANCE_COLUMNS, index=sounding_id),
        "modeled": pd.DataFrame(modeled.reshape(n_sounding, -1), columns=RADIANCE_COLUMNS, index=sounding_id),
        "wavelength": pd.DataFrame(wavelength.reshape(n_sounding, -1), columns=RADIANCE_COLUMNS, index=sounding_id),
        "signal": pd.DataFrame(
            signal,
            columns=[f"signal_{band}" for band in BAND_NAMES],
            index=sounding_id,
        ),
    }


def load_single_file(filename: str, components: set[str]) -> dict[str, pd.DataFrame]:
    with h5py.File(filename, "r") as file:
        land_fraction = file["/RetrievalGeometry/retrieval_land_fraction"][:]
        sounding_all = file["/RetrievalHeader/sounding_id"][:]
        idx = (land_fraction == 100.0) & (sounding_all % 10 == 1)

        if not np.any(idx):
            return {}

        sounding_id = file["/RetrievalHeader/sounding_id"][idx]
        results = {}
        if "screen" in components:
            results["screen"] = load_screen_data(file, idx, sounding_id)
        if "geometry" in components or "los" in components:
            results["geometry"] = load_geometry_data(file, idx, sounding_id)
        if "state" in components or "los" in components:
            results["state"] = load_state_data(file, idx, sounding_id)
        if components.intersection({"measured", "modeled", "wavelength", "signal"}):
            spectral = load_spectral_data(file, idx, sounding_id)
            for component in ["measured", "modeled", "wavelength", "signal"]:
                if component in components:
                    results[component] = spectral[component]
        return results


def export_month(
    yymm: str,
    l2dia_dir: str | Path,
    output_dir: str | Path,
    components: Iterable[str],
    device: str,
    los_batch_size: int,
    overwrite: bool,
) -> None:
    components = set(components)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_names = {
        "screen": f"df_screen_{yymm}.parquet",
        "geometry": f"df_geometry_{yymm}.parquet",
        "state": f"df_state_{yymm}.parquet",
        "measured": f"df_measured_{yymm}.parquet",
        "modeled": f"df_modeled_{yymm}.parquet",
        "wavelength": f"df_wavelength_{yymm}.parquet",
        "signal": f"df_signal_{yymm}.parquet",
        "los": f"df_los_{yymm}.parquet",
    }
    if not overwrite:
        components = {
            name for name in components if not (output_dir / output_names[name]).exists()
        }
    if not components:
        print(f"{yymm}: all requested outputs already exist")
        return

    files = find_files_for_yymm(l2dia_dir, yymm)
    if not files:
        print(f"{yymm}: no L2DiaND files found")
        return

    collectors: dict[str, list[pd.DataFrame]] = {name: [] for name in components.union({"geometry", "state"})}
    for file_path in tqdm(files, desc=f"Processing {yymm}", ncols=100):
        results = load_single_file(file_path, components)
        for key, value in results.items():
            collectors.setdefault(key, []).append(value)

    combined = {}
    for component in components.union({"geometry", "state"}):
        if collectors.get(component):
            combined[component] = pd.concat(collectors[component])

    if "los" in components:
        if "geometry" not in combined or "state" not in combined:
            raise RuntimeError("LOS generation requires geometry and state data.")
        combined["los"] = predict_los(
            combined["geometry"],
            combined["state"],
            device=device,
            batch_size=los_batch_size,
        )

    for component in components:
        if component in combined:
            out = output_dir / output_names[component]
            combined[component].to_parquet(out)
            print(f"Saved {component}: {out}")


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        if "los" in args.components:
            raise SystemExit("--smoke-test supports non-LOS components only; omit los.")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = load_smoke_dataframes()
        for month in args.months:
            yymm = normalize_yymm(month)
            component_frames = {
                "screen": pd.DataFrame(
                    {
                        "latitude": np.linspace(30.0, 31.0, len(frames["geo"])),
                        "longitude": np.linspace(120.0, 121.0, len(frames["geo"])),
                    },
                    index=frames["geo"].index,
                ),
                "geometry": frames["geo"],
                "state": pd.concat(
                    [
                        frames["retr"],
                        frames["apriori"],
                        frames["wf"],
                        pd.Series(1, index=frames["geo"].index, name="outcome_flag"),
                    ],
                    axis=1,
                ),
                "measured": frames["rad"],
                "modeled": frames["rad"],
                "wavelength": frames["rad"] * 0.0,
                "signal": pd.DataFrame(
                    {f"signal_{band}": 1.0 for band in BAND_NAMES},
                    index=frames["geo"].index,
                ),
            }
            for component in args.components:
                out = output_dir / f"df_{component}_{yymm}.parquet"
                if out.exists() and not args.overwrite:
                    continue
                component_frames[component].to_parquet(out)
                print(f"Saved smoke {component}: {out}")
        return
    months = [normalize_yymm(month) for month in args.months]
    for yymm in months:
        export_month(
            yymm=yymm,
            l2dia_dir=args.l2dia_dir,
            output_dir=args.output_dir,
            components=args.components,
            device=args.device,
            los_batch_size=args.los_batch_size,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
