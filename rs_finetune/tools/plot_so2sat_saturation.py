#!/usr/bin/env python3
"""Plot the x-So2Sat S1 saturation experiment as a full-width dot chart."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "resnet-50": "ResNet-50",
    "vit-b": "ViT-B",
    "ibot": "iBOT",
    "dinov2": "DINOv2",
    "dinov3": "DINOv3",
    "chivit": "χViT",
    "dofa": "DOFA",
    "croma": "CROMA",
    "anysat": "AnySat",
    "prithvi": "Prithvi",
    "SatlasNet": "SatlasNet",
    "terrafm": "TerraFM",
    "TerraMind": "TerraMind",
    "panopticon": "Panopticon",
}

REGIMES = ("rgb", "s2", "mixture", "s1")


def load_means(results_dir: Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for path in sorted(results_dir.glob("*.csv")):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            values[row["train_regime"]].append(float(row["s1_test_accuracy"]))
        model = DISPLAY_NAMES.get(rows[0]["model"], rows[0]["model"])
        results[model] = {regime: float(np.mean(values[regime])) for regime in REGIMES}
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = load_means(args.results_dir)
    models = sorted(results, key=lambda model: results[model]["s1"], reverse=True)
    x = np.arange(len(models), dtype=float)

    colors = {
        "rgb": "#78A6D8",
        "s2": "#D6AC72",
        "mixture": "#666666",
        "s1": "#152238",
    }
    markers = {"rgb": "o", "s2": "s", "mixture": "D", "s1": "o"}
    labels = {"rgb": "RGB", "s2": "Sentinel-2", "mixture": "Mixture", "s1": "Sentinel-1"}
    offsets = {"rgb": -0.12, "s2": 0.12, "mixture": 0.0, "s1": 0.0}

    fig, ax = plt.subplots(figsize=(7.15, 2.85))
    for index, model in enumerate(models):
        source_best = max(results[model]["rgb"], results[model]["s2"])
        ax.plot(
            [index, index],
            [source_best, results[model]["s1"]],
            color="#CBD2D9",
            linewidth=2.0,
            zorder=1,
        )

    for regime in REGIMES:
        y = [results[model][regime] for model in models]
        ax.scatter(
            x + offsets[regime],
            y,
            s=27 if regime != "s1" else 32,
            marker=markers[regime],
            color=colors[regime],
            edgecolor="white",
            linewidth=0.55,
            label=labels[regime],
            zorder=3 if regime != "s1" else 4,
        )

    random_baseline = 100.0 / 17.0
    ax.axhline(
        random_baseline,
        color="#9A4F4F",
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.text(
        -0.35,
        random_baseline - 0.75,
        f"random ({random_baseline:.2f})",
        color="#8A4141",
        fontsize=7.2,
        ha="left",
        va="top",
    )

    ax.set_ylabel("Sentinel-1 test accuracy (%)", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=42, ha="right", rotation_mode="anchor", fontsize=7.2)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(0, 40)
    ax.set_yticks(np.arange(0, 41, 10))
    ax.tick_params(axis="y", labelsize=7.5, length=3)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.grid(axis="y", color="#E7E9EC", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#A7ADB4")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=4,
        frameon=False,
        fontsize=7.7,
        handletextpad=0.35,
        columnspacing=1.4,
    )
    ax.text(
        0.285,
        1.16,
        "Training bands:",
        transform=ax.transAxes,
        fontsize=7.7,
        ha="right",
        va="center",
    )

    fig.tight_layout(pad=0.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
