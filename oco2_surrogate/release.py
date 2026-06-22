from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from .load_data import get_state_indices
from .preprocessor import StandardScaler


BANDS = ["o2", "weak_co2", "strong_co2"]
SPECTRAL_POINTS = 1016
FULL_RADIANCE_DIM = len(BANDS) * SPECTRAL_POINTS


def geometry_columns() -> List[str]:
    return [
        "cos_relative_azimuth",
        "sin_relative_azimuth",
        "cos_solar_zenith",
        "sin_solar_zenith",
        "cos_zenith",
        "sin_zenith",
        "cos_polarization_angle",
        "sin_polarization_angle",
        "altitude",
        "relative_velocity",
        "slope",
        "solar_distance",
        "solar_relative_velocity",
    ]


def retrieved_columns() -> List[str]:
    columns = []
    for prefix in ["co2_profile", "vector_pressure_levels", "temperature_profile", "h2o_profile"]:
        columns.extend(f"{prefix}_{i}" for i in range(20))
    for band in BANDS:
        columns.extend([f"offset_{band}", f"spacing_{band}"])
        columns.extend(
            [
                f"brdf_weight_{band}",
                f"brdf_weight_slope_{band}",
                f"brdf_weight_quadratic_{band}",
                f"eof_1_scale_{band}",
                f"eof_2_scale_{band}",
                f"eof_3_scale_{band}",
            ]
        )
    columns.extend(["fluorescence_at_reference", "fluorescence_slope"])
    return columns


def apriori_columns() -> List[str]:
    columns = []
    for prefix in [
        "co2_profile_apriori",
        "vector_pressure_levels_apriori",
        "temperature_profile_apriori",
        "h2o_profile_apriori",
    ]:
        columns.extend(f"{prefix}_{i}" for i in range(20))
    extra_count = len(retrieved_columns()) - len(columns)
    columns.extend(f"apriori_extra_{i}" for i in range(extra_count))
    return columns


def radiance_columns() -> List[str]:
    return [f"{band}_{i}" for band in BANDS for i in range(SPECTRAL_POINTS)]


def _frame(values: np.ndarray, columns: List[str]) -> pd.DataFrame:
    index = pd.Index(np.arange(1, values.shape[0] + 1), name="sounding_id")
    return pd.DataFrame(values.astype("float32"), columns=columns, index=index)


def make_smoke_dataframes(n_rows: int = 24, seed: int = 7) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_geo = len(geometry_columns())
    n_retr = len(retrieved_columns())
    n_apr = len(apriori_columns())
    n_wf = 20

    geo = rng.normal(0.0, 0.2, size=(n_rows, n_geo))
    geo[:, geometry_columns().index("solar_distance")] = 1.0 + rng.normal(0.0, 0.01, n_rows)
    retr = rng.normal(0.0, 0.05, size=(n_rows, n_retr))
    apriori = retr + rng.normal(0.0, 0.01, size=(n_rows, n_apr))
    wf = np.abs(rng.normal(1.0 / n_wf, 0.005, size=(n_rows, n_wf)))
    wf = wf / wf.sum(axis=1, keepdims=True)

    x = np.linspace(0, 2 * np.pi, FULL_RADIANCE_DIM, dtype=np.float32)
    base = np.sin(x)[None, :] * 0.02 + np.cos(0.25 * x)[None, :] * 0.01
    row_term = retr[:, :1] * 0.1 + geo[:, :1] * 0.03
    los = base + row_term + rng.normal(0.0, 0.002, size=(n_rows, FULL_RADIANCE_DIM))
    radiance = los + rng.normal(0.0, 0.001, size=(n_rows, FULL_RADIANCE_DIM))

    return {
        "geo": _frame(geo, geometry_columns()),
        "retr": _frame(retr, retrieved_columns()),
        "apriori": _frame(apriori, apriori_columns()),
        "wf": _frame(wf, [f"xco2_pressure_weighting_function_{i}" for i in range(n_wf)]),
        "los": _frame(los, radiance_columns()),
        "rad": _frame(radiance, radiance_columns()),
    }


def write_smoke_data(base_dir: str | Path = "examples/smoke_data", n_rows: int = 24) -> None:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    frames = make_smoke_dataframes(n_rows=n_rows)
    for name, df in frames.items():
        df.to_parquet(base / f"{name}.parquet", index=True)
    metadata = {
        "description": "Synthetic fixture data for release smoke tests.",
        "rows": n_rows,
        "radiance_columns": FULL_RADIANCE_DIM,
    }
    (base / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_smoke_dataframes(base_dir: str | Path = "examples/smoke_data") -> Dict[str, pd.DataFrame]:
    base = Path(base_dir)
    if not (base / "geo.parquet").exists():
        write_smoke_data(base)
    return {
        "geo": pd.read_parquet(base / "geo.parquet"),
        "retr": pd.read_parquet(base / "retr.parquet"),
        "apriori": pd.read_parquet(base / "apriori.parquet"),
        "wf": pd.read_parquet(base / "wf.parquet"),
        "los": pd.read_parquet(base / "los.parquet"),
        "rad": pd.read_parquet(base / "rad.parquet"),
    }


def _tensor(df: pd.DataFrame, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(df.to_numpy(copy=True)).to(dtype)


def make_forward_smoke_datasets(
    max_rows: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[TensorDataset, TensorDataset, Dict[str, Dict[str, List[int]]], Dict[str, StandardScaler], np.ndarray]:
    frames = load_smoke_dataframes()
    if max_rows:
        frames = {name: df.iloc[:max_rows].copy() for name, df in frames.items()}

    train_n = max(4, int(len(frames["geo"]) * 0.75))
    train = {name: df.iloc[:train_n].copy() for name, df in frames.items()}
    test = {name: df.iloc[train_n:].copy() for name, df in frames.items()}
    if len(test["geo"]) == 0:
        test = {name: df.iloc[-2:].copy() for name, df in frames.items()}

    geo_scaler = StandardScaler().fit(_tensor(train["geo"], dtype))
    retr_scaler = StandardScaler().fit(_tensor(train["retr"], dtype))
    rad_scaler = StandardScaler().fit(_tensor(train["rad"], dtype))

    def dataset(subset: Dict[str, pd.DataFrame]) -> TensorDataset:
        return TensorDataset(
            geo_scaler.transform(_tensor(subset["geo"], dtype)),
            retr_scaler.transform(_tensor(subset["retr"], dtype)),
            rad_scaler.transform(_tensor(subset["los"], dtype)),
            rad_scaler.transform(_tensor(subset["rad"], dtype)),
        )

    all_state_indices = {
        band: get_state_indices(frames["retr"].columns.tolist(), band) for band in BANDS
    }
    all_nnan_indices = {
        band: list(range(SPECTRAL_POINTS)) for band in BANDS
    }
    indices = {"all_state_indices": all_state_indices, "all_nnan_indices": all_nnan_indices}
    scalers = {"geometry": geo_scaler, "retrieved": retr_scaler, "radiance": rad_scaler}
    return dataset(train), dataset(test), indices, scalers, frames["retr"].columns.to_numpy()


def load_smoke_inverse_dfs(max_rows: int | None = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    frames = load_smoke_dataframes()
    if max_rows:
        frames = {name: df.iloc[:max_rows].copy() for name, df in frames.items()}
    split = max(4, int(len(frames["geo"]) * 0.75))
    train = {
        "df_geo_train": frames["geo"].iloc[:split].copy(),
        "df_retr_train": frames["retr"].iloc[:split].copy(),
        "df_apriori_train": frames["apriori"].iloc[:split].copy(),
        "df_wf_train": frames["wf"].iloc[:split].copy(),
        "df_rad_train": frames["rad"].iloc[:split].copy(),
    }
    test = {
        "df_geo_test": frames["geo"].iloc[split:].copy(),
        "df_retr_test": frames["retr"].iloc[split:].copy(),
        "df_apriori_test": frames["apriori"].iloc[split:].copy(),
        "df_wf_test": frames["wf"].iloc[split:].copy(),
        "df_rad_test": frames["rad"].iloc[split:].copy(),
    }
    if len(test["df_geo_test"]) == 0:
        test = {key: value.iloc[-2:].copy() for key, value in train.items()}
        test = {key.replace("_train", "_test"): value for key, value in test.items()}
    return train, test
