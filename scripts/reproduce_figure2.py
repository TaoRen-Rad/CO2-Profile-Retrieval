#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oco2_surrogate.forward_models import BAND_NAME_TO_INDEX, load_forward_model
from oco2_surrogate.load_data import RADIANCE_SCALER, state2retrieved


BANDS = [
    ("o2", "O$_2$", 0, 1016),
    ("weak_co2", "Weak CO$_2$", 1016, 2032),
    ("strong_co2", "Strong CO$_2$", 2032, 3048),
]
RADIANCE_DISPLAY_SCALE = 1e20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate manuscript Figure 2 radiance comparison from release inputs."
    )
    parser.add_argument("--input-dir", default="examples/figure2_input")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--output-dir", default="outputs/reproduction")
    parser.add_argument("--sounding-id", type=int, default=2022120508103571)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_row(input_dir: Path, name: str, sounding_id: int) -> pd.DataFrame:
    path = input_dir / f"{name}_{sounding_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if sounding_id not in df.index:
        raise ValueError(f"{sounding_id} not found in {path}")
    return df.loc[[sounding_id]]


def predict_forward_radiance(
    artifact_root: Path,
    geometry: pd.DataFrame,
    retrieved: pd.DataFrame,
    los: pd.DataFrame,
    device: str,
) -> np.ndarray:
    geo_tensor = torch.from_numpy(geometry.to_numpy(copy=True)).to(device, dtype=torch.float32)
    ret_tensor = torch.from_numpy(retrieved.to_numpy(copy=True)).to(device, dtype=torch.float32)
    los_tensor = torch.from_numpy(los.to_numpy(copy=True)).to(device, dtype=torch.float32)
    output = torch.full_like(los_tensor, torch.nan)

    for band, _, start, end in BANDS:
        info = load_forward_model(
            band=band,
            base_name="train_forward",
            data_dir=str(artifact_root / "forward"),
            device=device,
        )
        model = info["model"].to(device)
        with torch.no_grad():
            output[:, start:end] = model.predict(geo_tensor, ret_tensor, los_tensor[:, start:end])

    return output.cpu().numpy()


def plot_figure(
    modeled: np.ndarray,
    measured: np.ndarray,
    predicted: np.ndarray,
    wavelength: np.ndarray,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
        }
    )
    fig = plt.figure(figsize=(6.5, 4.0))
    gs = GridSpec(
        3,
        3,
        left=0.105,
        right=0.99,
        bottom=0.14,
        top=0.9,
        wspace=0.16,
        hspace=0.2,
        height_ratios=[1.0, 0.5, 0.5],
    )
    dash_a = (0, (2, 2))
    dash_b = (2, (2, 2))

    for col, (_, title, start, end) in enumerate(BANDS):
        label = modeled[start:end].astype(float)
        pred = predicted[start:end].astype(float)
        meas = measured[start:end].astype(float)
        wave = wavelength[start:end].astype(float)
        pred[np.isnan(label)] = np.nan
        meas[np.isnan(label)] = np.nan
        if np.all(np.isnan(wave)):
            x = np.arange(end - start)
            xlabel = "Band Spectral Index"
        else:
            x = wave
            xlabel = r"Wavelength [$\mu$m]"
        finite_x = np.isfinite(x)
        x_min = np.nanmin(x[finite_x])
        x_max = np.nanmax(x[finite_x])

        ax1 = fig.add_subplot(gs[0, col])
        ax1.plot(x, meas * RADIANCE_DISPLAY_SCALE, label="Observed", alpha=0.8, linewidth=0.5, color="gray")
        ax1.plot(x, label * RADIANCE_DISPLAY_SCALE, label="Full-Physics", alpha=0.8, linewidth=0.5, linestyle=dash_a, color="red")
        ax1.plot(x, pred * RADIANCE_DISPLAY_SCALE, label="Emulated", alpha=0.8, linewidth=0.5, linestyle=dash_b, color="blue")
        ax1.set_title(title)
        ax1.set_xlim(x_min, x_max)
        ax1.set_xticklabels([])
        ax1.grid(True, alpha=0.3)
        ymax = np.nanmax(label * RADIANCE_DISPLAY_SCALE)
        if np.isfinite(ymax) and ymax > 0:
            ax1.set_ylim(0, ymax * 1.1)
        if col == 0:
            ax1.text(
                -0.22,
                0.5,
                r"$I$ [$\mathrm{ph\;s^{-1}\,m^{-2}\,sr^{-1}\,\mu m^{-1}}$]",
                va="center",
                ha="center",
                transform=ax1.transAxes,
                rotation=90,
            )
        ax1.legend(loc="lower right", fontsize=7)

        ax2 = fig.add_subplot(gs[1, col])
        scale = np.nanmax(label)
        if np.isfinite(scale) and scale > 0:
            ax2.plot(x, (label - pred) / scale * 100.0, linestyle="--", linewidth=0.5, color="red")
        ax2.axhline(0, color="black", linestyle="--", alpha=0.8, linewidth=0.6)
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xticklabels([])
        ax2.grid(True, alpha=0.3)
        if col == 0:
            ax2.set_ylabel(r"$\epsilon$ [\%]")

        ax3 = fig.add_subplot(gs[2, col])
        scale_meas = np.nanmax(np.abs(meas))
        if np.isfinite(scale_meas) and scale_meas > 0:
            ax3.plot(x, (label - meas) / scale_meas * 100.0, linestyle=dash_a, linewidth=0.5, color="red")
            ax3.plot(x, (pred - meas) / scale_meas * 100.0, linestyle=dash_b, linewidth=0.5, color="blue")
        ax3.axhline(0, color="black", linestyle="--", alpha=0.8, linewidth=0.6)
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_xlabel(xlabel)
        ax3.grid(True, alpha=0.3)
        if col == 0:
            ax3.set_ylabel(r"$\epsilon$ [\%]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    artifact_root = Path(args.artifact_root)
    output_dir = Path(args.output_dir)
    sid = args.sounding_id

    geometry = load_row(input_dir, "geometry", sid)
    state = load_row(input_dir, "state", sid)
    los = load_row(input_dir, "los", sid)
    modeled = load_row(input_dir, "modeled", sid) / RADIANCE_SCALER
    measured = load_row(input_dir, "measured", sid) / RADIANCE_SCALER
    wavelength = load_row(input_dir, "wavelength", sid)
    retrieved = state2retrieved(state)

    predicted = predict_forward_radiance(artifact_root, geometry, retrieved, los, args.device)
    output_png = output_dir / f"figure2_radiance_{sid}.png"
    emulated_path = output_dir / f"figure2_emulated_radiance_{sid}.parquet"
    pd.DataFrame(predicted, index=geometry.index, columns=modeled.columns).to_parquet(emulated_path)
    plot_figure(
        modeled.iloc[0].to_numpy(),
        measured.iloc[0].to_numpy(),
        predicted[0],
        wavelength.iloc[0].to_numpy(),
        output_png,
    )

    metrics = {}
    for band, _, start, end in BANDS:
        label = modeled.iloc[0, start:end].to_numpy(dtype=float)
        pred = predicted[0, start:end].astype(float)
        finite = np.isfinite(label) & np.isfinite(pred)
        metrics[band] = {
            "finite_points": int(finite.sum()),
            "rmse_full_physics_minus_emulated_scaled": float(np.sqrt(np.mean((label[finite] - pred[finite]) ** 2))),
            "mae_full_physics_minus_emulated_scaled": float(np.mean(np.abs(label[finite] - pred[finite]))),
        }
    result = {
        "sounding_id": sid,
        "source": str(input_dir),
        "artifact_root": str(artifact_root),
        "output_png": str(output_png),
        "emulated_radiance_parquet": str(emulated_path),
        "metrics": metrics,
    }
    metrics_path = output_dir / f"figure2_radiance_{sid}_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
