#!/usr/bin/env python3
"""Export aligned x-So2Sat features from the frozen TerraMind S2-head model."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from terratorch import BACKBONE_REGISTRY

dataset_path = Path(__file__).parent / "change_detection_pytorch" / "datasets" / "so2sat.py"
dataset_spec = importlib.util.spec_from_file_location("geocrossbench_so2sat", dataset_path)
dataset_module = importlib.util.module_from_spec(dataset_spec)
assert dataset_spec.loader is not None
dataset_spec.loader.exec_module(dataset_module)
So2SatDataset = dataset_module.So2SatDataset


OPTICAL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
VIEWS = {
    "rgb": ["B02", "B03", "B04"],
    "s2": OPTICAL_BANDS,
    "s1": ["VV", "VH"],
    "ns1s2": ["B8A", "B11", "B12"],
}


def terramind_input(x: torch.Tensor, bands: list[str]) -> dict[str, torch.Tensor]:
    if bands == ["VV", "VH"]:
        return {"S1GRD": x}
    optical = x.new_zeros(x.shape[0], len(OPTICAL_BANDS), x.shape[2], x.shape[3])
    for source_index, band in enumerate(bands):
        optical[:, OPTICAL_BANDS.index(band)] = x[:, source_index]
    return {"S2L2A": optical}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    model = BACKBONE_REGISTRY.build(
        "terramind_v1_base",
        pretrained=True,
        modalities=["S2L2A", "S1GRD"],
        img_size=224,
        patch_size=16,
        bands={
            "S2L2A": [
                "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2",
                "RED_EDGE_3", "NIR", "NIR_NARROW", "SWIR_1", "SWIR_2",
            ],
            "S1GRD": ["VV", "VH"],
        },
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        for view, bands in VIEWS.items():
            dataset = So2SatDataset(split=split, bands=bands, img_size=224)
            if args.max_samples is not None:
                dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            features, labels = [], []
            with torch.inference_mode():
                for x, y, _metadata in loader:
                    inputs = terramind_input(x.to(device, non_blocking=True), bands)
                    outputs = model(inputs)
                    feats = outputs[-1].mean(dim=1)
                    features.append(feats.detach().cpu().reshape(len(y), -1).numpy())
                    labels.append(y.numpy())
            np.savez_compressed(
                args.output_dir / f"{split}__{view}.npz",
                features=np.concatenate(features),
                labels=np.concatenate(labels),
                sample_ids=np.arange(len(dataset)),
            )


if __name__ == "__main__":
    main()
