#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REQUIRED = [
    "artifacts/forward/scalers.pth",
    "artifacts/forward/indices.pth",
    "artifacts/forward/df_retrieved_columns.pth",
    "artifacts/forward/o2/mlp_model_best.pth",
    "artifacts/forward/o2/model_config.pkl",
    "artifacts/forward/weak_co2/mlp_model_best.pth",
    "artifacts/forward/weak_co2/model_config.pkl",
    "artifacts/forward/strong_co2/mlp_model_best.pth",
    "artifacts/forward/strong_co2/model_config.pkl",
    "artifacts/inverse/inverse_model_best_test.pth",
    "artifacts/inverse/scalers.pth",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check release artifact paths.")
    parser.add_argument("--allow-missing", action="store_true", help="Report missing artifacts without failing.")
    args = parser.parse_args()

    missing = [path for path in REQUIRED if not Path(path).exists()]
    if missing:
        print("Missing release artifacts:")
        for path in missing:
            print(f"  {path}")
        if not args.allow_missing:
            raise SystemExit(1)
    else:
        print("All release artifact paths are present.")


if __name__ == "__main__":
    main()
