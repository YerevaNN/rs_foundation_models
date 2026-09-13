#!/usr/bin/env python3
"""Export aligned frozen x-so2sat features for the saturation probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

RS_FINETUNE_ROOT = Path(__file__).resolve().parents[1]
if str(RS_FINETUNE_ROOT) not in sys.path:
    sys.path.insert(0, str(RS_FINETUNE_ROOT))

from change_detection_pytorch.datasets import So2SatDataset
from train_classifier import Classifier
from utils import create_collate_fn


VIEWS = {
    "rgb": ["B02", "B03", "B04"],
    "s2": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    "s1": ["VV", "VH"],
    "ns1s2": ["B8A", "B11", "B12"],
}


def pad_channels(x: torch.Tensor, count: int) -> torch.Tensor:
    if x.shape[1] > count:
        raise ValueError(f"Input has {x.shape[1]} channels, expected at most {count}")
    if x.shape[1] == count:
        return x
    zeros = x.new_zeros(x.shape[0], count - x.shape[1], x.shape[2], x.shape[3])
    return torch.cat([x, zeros], dim=1)


def prepare_input(model: Classifier, x: torch.Tensor, bands: list[str]) -> torch.Tensor:
    name = model.backbone_name.lower()
    weights = model.backbone_weights.lower()
    # Channel-aware/wavelength-aware encoders consume the requested channels directly.
    if any(token in name for token in ("cvit", "dofa", "clay", "panopticon")):
        return x
    if "anysat" in name:
        return pad_channels(x, 3 if len(bands) <= 3 else (10 if len(bands) <= 10 else 12))
    if "prithvi" in name:
        return pad_channels(x, model.encoder.patch_embed.proj.in_channels)
    if "satlas" in weights or model.enable_multiband_input:
        return pad_channels(x, model.multiband_channel_count)
    return pad_channels(x, 3)


def load_model(args: argparse.Namespace, cfg: dict) -> Classifier:
    model = Classifier(
        backbone_name=cfg["backbone"],
        backbone_weights=cfg["encoder_weights"],
        in_features=cfg["in_features"],
        num_classes=17,
        lr=0.0,
        scheduler="cosine",
        checkpoint_path=str(args.checkpoint),
        only_head=True,
        warmup_steps=0,
        eta_min=0.0,
        warmup_start_lr=0.0,
        weight_decay=0.0,
        mixup=False,
        bands=VIEWS["s2"],
        enable_multiband_input=True,
        multiband_channel_count=12,
        shared_proj=args.shared_proj,
        add_ch_embed=args.add_ch_embed,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional per-split limit for smoke tests")
    parser.add_argument("--shared-proj", action="store_true")
    parser.add_argument("--add-ch-embed", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.model_config.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, cfg).to(device)
    head = model.encoder.head if not hasattr(model, "classifier") else model.classifier
    captured: list[torch.Tensor] = []

    def capture_input(_module, inputs):
        captured.append(inputs[0].detach().cpu())

    hook = head.register_forward_pre_hook(capture_input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for split in ("train", "valid", "test"):
            for view, bands in VIEWS.items():
                dataset = So2SatDataset(split=split, bands=bands, img_size=args.image_size)
                if args.max_samples is not None:
                    dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=create_collate_fn("classification"),
                    pin_memory=True,
                )
                model.bands = list(bands)
                features, labels = [], []
                with torch.inference_mode():
                    for x, y, metadata in loader:
                        captured.clear()
                        x = prepare_input(model, x.to(device, non_blocking=True), bands)
                        if "dofa" in cfg["backbone"].lower():
                            model(x, list(metadata[0]["waves"]))
                        elif "clay" in cfg["backbone"].lower():
                            model(x, metadata)
                        else:
                            model(x)
                        if len(captured) != 1:
                            raise RuntimeError(f"Expected one pre-logit tensor, captured {len(captured)}")
                        features.append(captured[0].reshape(len(y), -1).numpy())
                        labels.append(y.numpy())
                np.savez_compressed(
                    args.output_dir / f"{split}__{view}.npz",
                    features=np.concatenate(features),
                    labels=np.concatenate(labels),
                    sample_ids=np.arange(len(dataset)),
                )
    finally:
        hook.remove()


if __name__ == "__main__":
    main()
