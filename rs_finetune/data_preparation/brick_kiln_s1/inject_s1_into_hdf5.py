#!/usr/bin/env python3
"""Atomically add corrected Sentinel-1 sidecars to Brick-Kiln HDF5 files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np


VH_KEY = "13 - VH"
VV_KEY = "14 - VV"
VALID_KEY = "S1 valid mask"
PROVENANCE_ATTR = "s1_provenance_json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject corrected VV/VH arrays into Brick-Kiln HDF5 files."
    )
    parser.add_argument("optical_dir", type=Path)
    parser.add_argument("s1_dir", type=Path, help="Directory of best-scene S1 sidecars")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_dataset(source: h5py.Dataset, target: h5py.File, name: str) -> None:
    target.create_dataset(
        name,
        data=source[()],
        dtype=source.dtype,
        chunks=source.chunks,
        compression=source.compression,
        compression_opts=source.compression_opts,
        shuffle=source.shuffle,
        fletcher32=source.fletcher32,
    )


def inject_one(task: tuple[str, str, bool]) -> dict[str, str]:
    optical_name, sidecar_name, overwrite = task
    optical = Path(optical_name)
    sidecar = Path(sidecar_name)
    temporary = optical.with_name(f".{optical.name}.s1tmp-{os.getpid()}")

    try:
        with h5py.File(optical, "r") as current:
            present = all(key in current for key in (VH_KEY, VV_KEY, VALID_KEY))
            if present and not overwrite:
                return {"sample": optical.stem, "status": "already-present"}
            if any(key in current for key in (VH_KEY, VV_KEY, VALID_KEY)) and not overwrite:
                raise RuntimeError(f"Partial S1 data already exists in {optical}")

        shutil.copy2(optical, temporary)
        with h5py.File(sidecar, "r") as source, h5py.File(temporary, "r+") as target:
            for key in (VH_KEY, VV_KEY, VALID_KEY):
                if key in target:
                    del target[key]
            copy_dataset(source["VH"], target, VH_KEY)
            copy_dataset(source["VV"], target, VV_KEY)
            copy_dataset(source["valid"], target, VALID_KEY)
            target.attrs[PROVENANCE_ATTR] = source.attrs["provenance_json"]
            target.attrs["s1_bbox"] = source.attrs["bbox"]
            target.attrs["s1_crs"] = source.attrs["crs"]
            target.attrs["s1_transform"] = source.attrs["transform"]
            target.attrs["s1_product"] = "best_single_scene"
            target.flush()

        with h5py.File(sidecar, "r") as source, h5py.File(temporary, "r") as target:
            checks = (
                np.array_equal(source["VH"][()], target[VH_KEY][()], equal_nan=True),
                np.array_equal(source["VV"][()], target[VV_KEY][()], equal_nan=True),
                np.array_equal(source["valid"][()], target[VALID_KEY][()]),
                target.attrs[PROVENANCE_ATTR] == source.attrs["provenance_json"],
            )
            if not all(checks):
                raise RuntimeError(f"Post-write validation failed for {optical}")

        os.replace(temporary, optical)
        return {"sample": optical.stem, "status": "injected"}
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    optical_files = sorted(args.optical_dir.glob("*.hdf5"))
    sidecar_files = {path.name: path for path in args.s1_dir.glob("*.hdf5")}
    optical_names = {path.name for path in optical_files}
    missing = [path.name for path in optical_files if path.name not in sidecar_files]
    extras = sorted(set(sidecar_files) - optical_names)
    if missing or extras:
        raise RuntimeError(
            f"Filename mismatch: {len(missing)} missing sidecars, {len(extras)} extra sidecars"
        )

    tasks = [(str(path), str(sidecar_files[path.name]), args.overwrite) for path in optical_files]
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(inject_one, tasks, chunksize=8), start=1):
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            if index % 500 == 0 or index == len(tasks):
                print(json.dumps({"processed": index, "total": len(tasks), "counts": counts}), flush=True)


if __name__ == "__main__":
    main()
