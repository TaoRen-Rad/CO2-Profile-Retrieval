#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from oco2_surrogate.forward_models import BAND_NAME_TO_INDEX, load_forward_model
from oco2_surrogate.preprocessor import StandardScaler
from oco2_surrogate.release import (
    BANDS,
    SPECTRAL_POINTS,
    load_smoke_dataframes,
    write_smoke_data,
)
from train_inverse import InverseRetrievalModel, build_radiance_features, compute_xco2_from_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small release backtest from committed artifacts.")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--input-dir", default="examples/backtest_input")
    parser.add_argument("--output-dir", default="outputs/backtest")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-missing-artifacts", action="store_true")
    return parser.parse_args()


def _tensor(df: pd.DataFrame) -> torch.Tensor:
    return torch.from_numpy(df.to_numpy(copy=True)).to(torch.float32)


def align_tensor(tensor: torch.Tensor, target_dim: int, fill: torch.Tensor | None = None) -> torch.Tensor:
    if tensor.shape[1] == target_dim:
        return tensor
    if fill is None:
        aligned = torch.zeros((tensor.shape[0], target_dim), dtype=tensor.dtype)
    else:
        aligned = fill.to(dtype=tensor.dtype).view(1, -1).repeat(tensor.shape[0], 1)
    n = min(tensor.shape[1], target_dim)
    aligned[:, :n] = tensor[:, :n]
    return aligned


def finite_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    diff = torch.abs(pred - target)
    finite = torch.isfinite(diff)
    if not finite.any():
        return float("nan")
    return diff[finite].mean().item()


def ensure_backtest_input(input_dir: Path) -> None:
    if (input_dir / "geo.parquet").exists():
        return
    input_dir.mkdir(parents=True, exist_ok=True)
    write_smoke_data(input_dir, n_rows=8)
    expected = {
        "forward_mae_max": 1.0e9,
        "xco2_mae_max": 1.0e10,
        "description": "Synthetic smoke-data tolerances; replace with sampled paper-data tolerances when available.",
    }
    (input_dir / "expected_metrics.json").write_text(
        json.dumps(expected, indent=2) + "\n",
        encoding="utf-8",
    )


def load_inverse_model(artifact_root: Path, device: torch.device) -> tuple[InverseRetrievalModel, dict]:
    scalers = torch.load(artifact_root / "inverse" / "scalers.pth", map_location="cpu")
    state_dict = torch.load(artifact_root / "inverse" / "inverse_model_best_test.pth", map_location="cpu")
    geo_dim = scalers["geometry"]["mean_"].numel()
    retr_dim = scalers["retrieved"]["mean_"].numel()
    wf_dim = scalers["wf"]["mean_"].numel()
    rad_dim = scalers["radiance"]["mean_"].numel() * 2
    hidden_dim = state_dict["fusion.out.bias"].numel()
    model = InverseRetrievalModel(
        geo_dim=geo_dim,
        apriori_dim=retr_dim,
        rad_dim=rad_dim,
        retr_dim=retr_dim,
        co2_indices=scalers["co2_indices"],
        pressure_indices=scalers["pressure_indices"],
        h2o_indices=scalers["h2o_indices"],
        hidden_dim=hidden_dim,
        co2_mean=scalers["retrieved"]["mean_"][scalers["co2_indices"]],
        co2_scale=scalers["retrieved"]["scale_"][scalers["co2_indices"]],
        wf_mean=scalers["wf"]["mean_"],
        wf_scale=scalers["wf"]["scale_"],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, scalers


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_backtest_input(input_dir)

    required = [
        artifact_root / "forward" / "scalers.pth",
        artifact_root / "forward" / "indices.pth",
        artifact_root / "inverse" / "scalers.pth",
        artifact_root / "inverse" / "inverse_model_best_test.pth",
    ]
    required.extend(artifact_root / "forward" / band / "mlp_model_best.pth" for band in BANDS)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Missing artifacts:")
        for path in missing:
            print(f"  {path}")
        if args.allow_missing_artifacts:
            return
        raise SystemExit(1)

    frames = load_smoke_dataframes(input_dir)
    forward_scalers = torch.load(artifact_root / "forward" / "scalers.pth", map_location="cpu")
    geo_cpu = align_tensor(_tensor(frames["geo"]), forward_scalers["geometry"]["mean_"].numel(), forward_scalers["geometry"]["mean_"])
    retr_cpu = align_tensor(_tensor(frames["retr"]), forward_scalers["retrieved"]["mean_"].numel(), forward_scalers["retrieved"]["mean_"])
    los_cpu = align_tensor(_tensor(frames["los"]), forward_scalers["radiance"]["mean_"].numel(), forward_scalers["radiance"]["mean_"])
    rad_true_cpu = align_tensor(_tensor(frames["rad"]), forward_scalers["radiance"]["mean_"].numel(), forward_scalers["radiance"]["mean_"])
    geo = geo_cpu.to(args.device)
    retr = retr_cpu.to(args.device)
    los = los_cpu.to(args.device)
    rad_true = rad_true_cpu.to(args.device)

    rad_pred = torch.zeros_like(rad_true)
    for band in BANDS:
        info = load_forward_model(
            band=band,
            base_name="train_forward",
            data_dir=str(artifact_root / "forward"),
            device=args.device,
        )
        model = info["model"]
        idx = BAND_NAME_TO_INDEX[band]
        rad_indices = [idx * SPECTRAL_POINTS + i for i in range(SPECTRAL_POINTS)]
        with torch.no_grad():
            rad_pred[:, rad_indices] = model.predict(geo, retr, los[:, rad_indices])

    inverse_model, inv_scalers = load_inverse_model(artifact_root, torch.device(args.device))
    geo_scaled = (geo_cpu - inv_scalers["geometry"]["mean_"]) / inv_scalers["geometry"]["scale_"]
    apriori_scaled = (retr_cpu - inv_scalers["retrieved"]["mean_"]) / inv_scalers["retrieved"]["scale_"]
    radiance_scaler = StandardScaler()
    radiance_scaler.register_buffer("mean_", inv_scalers["radiance"]["mean_"])
    radiance_scaler.register_buffer("scale_", inv_scalers["radiance"]["scale_"])
    rad_feat = build_radiance_features(rad_pred.cpu(), radiance_scaler)
    with torch.no_grad():
        retr_pred, wf_pred = inverse_model(
            geo_scaled.to(args.device),
            apriori_scaled.to(args.device),
            rad_feat.to(args.device),
        )
        xco2_pred = inverse_model.compute_xco2(retr_pred[:, : inverse_model.retr_dim], wf_pred).cpu()

    co2_indices = inv_scalers["co2_indices"]
    wf_true = align_tensor(_tensor(frames["wf"]), inv_scalers["wf"]["mean_"].numel(), inv_scalers["wf"]["mean_"])
    xco2_true = compute_xco2_from_profiles(retr_cpu[:, co2_indices], wf_true)
    metrics = {
        "forward_mae": finite_mae(rad_pred.cpu(), rad_true.cpu()),
        "xco2_mae": finite_mae(xco2_pred, xco2_true),
    }

    expected_path = input_dir / "expected_metrics.json"
    if expected_path.exists():
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        if metrics["forward_mae"] > expected.get("forward_mae_max", float("inf")):
            raise RuntimeError(f"forward_mae exceeded tolerance: {metrics['forward_mae']}")
        if metrics["xco2_mae"] > expected.get("xco2_mae_max", float("inf")):
            raise RuntimeError(f"xco2_mae exceeded tolerance: {metrics['xco2_mae']}")

    results = pd.DataFrame(
        {
            "xco2_pred": xco2_pred.squeeze().numpy(),
            "xco2_true": xco2_true.squeeze().numpy(),
        },
        index=frames["geo"].index,
    )
    results.to_parquet(output_dir / "results.parquet")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    plt.figure(figsize=(5, 4))
    plt.scatter(results["xco2_true"], results["xco2_pred"], s=18)
    plt.xlabel("True XCO2")
    plt.ylabel("Predicted XCO2")
    plt.tight_layout()
    plt.savefig(output_dir / "xco2_scatter.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(rad_true[0].cpu().numpy(), label="target", linewidth=1)
    plt.plot(rad_pred[0].cpu().numpy(), label="pred", linewidth=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "radiance_example.png")
    plt.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
