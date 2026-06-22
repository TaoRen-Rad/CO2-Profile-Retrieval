#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


FORWARD_SOURCES = {
    "o2": "new_los_train_09_final_o2",
    "weak_co2": "new_los_train_09_final_weak_co2",
    "strong_co2": "new_los_train_09_final_strong_co2",
}
INVERSE_SOURCE = "uncert_train_17_final"
SHARED_FORWARD_FILES = ["scalers.pth", "indices.pth", "df_retrieved_columns.pth"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import release model artifacts into stable paths.")
    parser.add_argument(
        "--source-root",
        default="/mnt/d/chenwei/SS_OCO2/status",
        help="Root containing historical status artifact directories.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts",
        help="Destination artifact root committed through Git LFS.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destination files.")
    return parser.parse_args()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, dst: Path, overwrite: bool) -> dict:
    if not src.exists():
        raise FileNotFoundError(f"Missing source artifact: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination exists, pass --overwrite to replace: {dst}")
    shutil.copy2(src, dst)
    return {
        "source": str(src),
        "path": str(dst),
        "size": dst.stat().st_size,
        "sha256": sha256sum(dst),
    }


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    entries = []

    shared_source = source_root / FORWARD_SOURCES["o2"]
    for filename in SHARED_FORWARD_FILES:
        entries.append(
            copy_file(
                shared_source / filename,
                output_root / "forward" / filename,
                overwrite=args.overwrite,
            )
        )

    for band, source_name in FORWARD_SOURCES.items():
        source_dir = source_root / source_name
        dest_dir = output_root / "forward" / band
        for filename in ["mlp_model_best.pth", "model_config.pkl"]:
            entries.append(copy_file(source_dir / filename, dest_dir / filename, args.overwrite))

    inverse_source = source_root / INVERSE_SOURCE
    for filename in ["inverse_model_best_test.pth", "scalers.pth"]:
        entries.append(
            copy_file(
                inverse_source / filename,
                output_root / "inverse" / filename,
                overwrite=args.overwrite,
            )
        )

    manifest = {
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "artifacts": entries,
    }
    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Imported {len(entries)} files into {output_root}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
