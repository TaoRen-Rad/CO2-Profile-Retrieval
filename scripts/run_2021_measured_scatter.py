#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oco2_surrogate.load_data import RADIANCE_SCALER, state2apriori, state2retrieved, state2wf
from oco2_surrogate.preprocessor import StandardScaler
from train_inverse import InverseRetrievalModel, build_radiance_features, compute_xco2_from_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run released inverse model on a random 2021 measured-radiance sample."
    )
    parser.add_argument("--data-dir", default="/mnt/d/chenwei/LOS_OCO2_DATA")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--output-dir", default="outputs/reproduction")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--months", nargs="*", default=[f"21{m:02d}" for m in range(1, 13)])
    parser.add_argument(
        "--cache-input-dir",
        default="examples/2021_measured_sample1000",
        help="Project-local cache for the sampled real input rows.",
    )
    parser.add_argument(
        "--use-cached-input",
        action="store_true",
        help="Run from --cache-input-dir instead of resampling external parquet files.",
    )
    return parser.parse_args()


def tensor_from_df(df: pd.DataFrame) -> torch.Tensor:
    return torch.from_numpy(df.to_numpy(copy=True)).to(torch.float32)


def make_scaler(state: dict) -> StandardScaler:
    scaler = StandardScaler()
    scaler.register_buffer("mean_", state["mean_"])
    scaler.register_buffer("scale_", state["scale_"])
    return scaler


def load_inverse_model(artifact_root: Path, device: torch.device) -> tuple[InverseRetrievalModel, dict]:
    scalers = torch.load(artifact_root / "inverse" / "scalers.pth", map_location="cpu")
    state_dict = torch.load(artifact_root / "inverse" / "inverse_model_best_test.pth", map_location="cpu")
    model = InverseRetrievalModel(
        geo_dim=scalers["geometry"]["mean_"].numel(),
        apriori_dim=scalers["retrieved"]["mean_"].numel(),
        rad_dim=scalers["radiance"]["mean_"].numel() * 2,
        retr_dim=scalers["retrieved"]["mean_"].numel(),
        co2_indices=scalers["co2_indices"],
        pressure_indices=scalers["pressure_indices"],
        h2o_indices=scalers["h2o_indices"],
        hidden_dim=state_dict["fusion.out.bias"].numel(),
        co2_mean=scalers["retrieved"]["mean_"][scalers["co2_indices"]],
        co2_scale=scalers["retrieved"]["scale_"][scalers["co2_indices"]],
        wf_mean=scalers["wf"]["mean_"],
        wf_scale=scalers["wf"]["scale_"],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, scalers


def collect_candidate_ids(data_dir: Path, months: Iterable[str]) -> pd.Index:
    ids = []
    for yymm in months:
        state_path = data_dir / f"df_state_1206_{yymm}.parquet"
        if not state_path.exists():
            raise FileNotFoundError(state_path)
        df_state = pd.read_parquet(state_path, columns=["outcome_flag"])
        good = df_state.index[df_state["outcome_flag"] == 1]
        ids.append(pd.Index(good))
    return ids[0].append(ids[1:]) if len(ids) > 1 else ids[0]


def split_ids_by_month(ids: pd.Index) -> dict[str, list[int]]:
    by_month: dict[str, list[int]] = {}
    for sid in ids.astype(str):
        yymm = sid[2:6]
        by_month.setdefault(yymm, []).append(int(sid))
    return by_month


def load_sample_frames(data_dir: Path, ids: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_month = split_ids_by_month(ids)
    geo_frames = []
    state_frames = []
    rad_frames = []
    for yymm, month_ids in sorted(by_month.items()):
        idx = pd.Index(month_ids)
        geo = pd.read_parquet(data_dir / f"df_geometry_{yymm}.parquet")
        state = pd.read_parquet(data_dir / f"df_state_1206_{yymm}.parquet")
        measured = pd.read_parquet(data_dir / f"df_measured_{yymm}.parquet")
        geo_frames.append(geo.loc[idx])
        state_frames.append(state.loc[idx])
        rad_frames.append(measured.loc[idx])
    geometry = pd.concat(geo_frames).loc[ids]
    state = pd.concat(state_frames).loc[ids]
    measured = pd.concat(rad_frames).loc[ids] / RADIANCE_SCALER
    return geometry, state, measured


def write_input_cache(
    cache_dir: Path,
    geometry: pd.DataFrame,
    state: pd.DataFrame,
    measured: pd.DataFrame,
    metadata: dict,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    geometry.to_parquet(cache_dir / "geometry.parquet", index=True)
    state.to_parquet(cache_dir / "state.parquet", index=True)
    measured.to_parquet(cache_dir / "measured_radiance_scaled.parquet", index=True)
    pd.DataFrame({"sounding_id": geometry.index.astype(str)}).to_csv(
        cache_dir / "sounding_ids.csv",
        index=False,
    )
    (cache_dir / "MANIFEST.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_input_cache(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    required = [
        cache_dir / "geometry.parquet",
        cache_dir / "state.parquet",
        cache_dir / "measured_radiance_scaled.parquet",
        cache_dir / "MANIFEST.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cached input files: {missing}")
    metadata = json.loads((cache_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    return (
        pd.read_parquet(cache_dir / "geometry.parquet"),
        pd.read_parquet(cache_dir / "state.parquet"),
        pd.read_parquet(cache_dir / "measured_radiance_scaled.parquet"),
        metadata,
    )


def run_inference(
    model: InverseRetrievalModel,
    scalers: dict,
    geometry: pd.DataFrame,
    state: pd.DataFrame,
    measured: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    retrieved = state2retrieved(state)
    apriori = state2apriori(state)
    wf = state2wf(state)

    geometry_scaler = make_scaler(scalers["geometry"])
    retrieved_scaler = make_scaler(scalers["retrieved"])
    radiance_scaler = make_scaler(scalers["radiance"])

    geo_scaled = geometry_scaler.transform(tensor_from_df(geometry))
    apr_scaled = retrieved_scaler.transform(tensor_from_df(apriori))
    rad_feat = build_radiance_features(tensor_from_df(measured), radiance_scaler)

    pred_means = []
    pred_sigmas = []
    pred_wfs = []
    with torch.no_grad():
        for start in range(0, len(geometry), batch_size):
            end = min(start + batch_size, len(geometry))
            retr_pred, wf_pred = model(
                geo_scaled[start:end].to(device),
                apr_scaled[start:end].to(device),
                rad_feat[start:end].to(device),
            )
            pred_means.append(retr_pred[:, : model.retr_dim].cpu())
            pred_sigmas.append(torch.exp(retr_pred[:, model.retr_dim :]).cpu())
            pred_wfs.append(wf_pred.cpu())

    retr_mean_scaled = torch.cat(pred_means, dim=0)
    retr_sigma_scaled = torch.cat(pred_sigmas, dim=0)
    wf_pred_scaled = torch.cat(pred_wfs, dim=0)

    co2_idx = scalers["co2_indices"]
    retr_mean_phys = retr_mean_scaled * scalers["retrieved"]["scale_"] + scalers["retrieved"]["mean_"]
    retr_sigma_phys = retr_sigma_scaled * scalers["retrieved"]["scale_"]
    wf_pred_phys = wf_pred_scaled * scalers["wf"]["scale_"] + scalers["wf"]["mean_"]
    xco2_pred = compute_xco2_from_profiles(retr_mean_phys[:, co2_idx], wf_pred_phys)
    xco2_sigma = torch.sqrt(torch.sum((wf_pred_phys * retr_sigma_phys[:, co2_idx]) ** 2, dim=1, keepdim=True))

    retrieved_raw = tensor_from_df(retrieved)
    wf_raw = tensor_from_df(wf)
    xco2_target = compute_xco2_from_profiles(retrieved_raw[:, co2_idx], wf_raw)

    return pd.DataFrame(
        {
            "sounding_id": geometry.index.astype(str),
            "xco2_target": xco2_target.squeeze().numpy(),
            "xco2_pred": xco2_pred.squeeze().numpy(),
            "xco2_sigma": xco2_sigma.squeeze().numpy(),
            "xco2_target_ppm": xco2_target.squeeze().numpy() * 1e6,
            "xco2_pred_ppm": xco2_pred.squeeze().numpy() * 1e6,
            "xco2_sigma_ppm": xco2_sigma.squeeze().numpy() * 1e6,
        },
        index=geometry.index,
    )


def write_scatter(results: pd.DataFrame, metrics: dict, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    x = results["xco2_target_ppm"].to_numpy()
    y = results["xco2_pred_ppm"].to_numpy()
    ax.scatter(x, y, s=8, alpha=0.55, edgecolors="none")
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("OCO-2 L2 XCO2 [ppm]")
    ax.set_ylabel("Model retrieval XCO2 [ppm]")
    ax.text(
        0.98,
        0.02,
        f"RMSE: {metrics['rmse_ppm']:.3f} ppm\nME: {metrics['mean_error_ppm']:.3f} ppm\nN: {metrics['n']}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    artifact_root = Path(args.artifact_root)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_cached_input:
        geometry, state, measured, cache_metadata = load_input_cache(cache_dir)
        sampled_ids = pd.Index(geometry.index.astype(np.int64), name="sounding_id")
    else:
        candidates = collect_candidate_ids(data_dir, args.months)
        rng = np.random.default_rng(args.seed)
        if len(candidates) < args.sample_size:
            raise ValueError(f"Only {len(candidates)} candidate soundings for sample size {args.sample_size}")
        sampled_ids = pd.Index(rng.choice(candidates.to_numpy(), size=args.sample_size, replace=False))
        sampled_ids = pd.Index(sampled_ids.astype(np.int64), name="sounding_id")
        geometry, state, measured = load_sample_frames(data_dir, sampled_ids)
        cache_metadata = {
            "description": "Real 2021 OCO-2 measured-radiance input rows sampled for release reproduction.",
            "data_dir": str(data_dir),
            "months": list(args.months),
            "seed": int(args.seed),
            "sample_size": int(args.sample_size),
            "radiance_file_pattern": "df_measured_{yymm}.parquet",
            "radiance_units": "scaled by 1e20, matching inverse training inputs",
        }
        write_input_cache(cache_dir, geometry, state, measured, cache_metadata)
    device = torch.device(args.device)
    model, scalers = load_inverse_model(artifact_root, device)
    results = run_inference(model, scalers, geometry, state, measured, device, args.batch_size)
    err_ppm = results["xco2_pred_ppm"] - results["xco2_target_ppm"]
    metrics = {
        "n": int(len(results)),
        "seed": int(args.seed),
        "sample_size": int(args.sample_size),
        "months": list(args.months),
        "data_dir": str(data_dir),
        "artifact_root": str(artifact_root),
        "radiance_source": "measured",
        "input_cache_dir": str(cache_dir),
        "used_cached_input": bool(args.use_cached_input),
        "rmse_ppm": float(np.sqrt(np.mean(err_ppm.to_numpy() ** 2))),
        "mean_error_ppm": float(np.mean(err_ppm.to_numpy())),
    }

    results_path = output_dir / "scatter_2021_measured_sample1000_results.parquet"
    metrics_path = output_dir / "scatter_2021_measured_sample1000_metrics.json"
    figure_path = output_dir / "scatter_2021_measured_sample1000.png"
    results.to_parquet(results_path, index=True)
    metrics["results_path"] = str(results_path)
    metrics["figure_path"] = str(figure_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_scatter(results, metrics, figure_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
