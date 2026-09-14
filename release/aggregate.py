"""Aggregate measured benchmark runs using the GeoCrossBench reporting protocol."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

TRANSFERS = [
    "RGB->RGB",
    "S2->S2",
    "RGB->SAR",
    "S2->SAR",
    "RGB->N'S1S2",
    "RGB->RGBN",
    "S2->S2+SAR",
]
SETTINGS = {
    "in_distribution": ["RGB->RGB", "S2->S2"],
    "no_overlap": ["RGB->SAR", "S2->SAR", "RGB->N'S1S2"],
    "superset": ["RGB->RGBN", "S2->S2+SAR"],
}


def summarize_seed_values(values):
    return {
        "mean": statistics.mean(values),
        "seed_std": statistics.stdev(values) if len(values) > 1 else None,
    }


def aggregate(rows, datasets, models, seeds):
    cells = defaultdict(dict)
    for row in rows:
        if row["status"] != "measured":
            raise ValueError("Only measured results may enter an aggregate")
        if (
            row["dataset"] not in datasets
            or row["model"] not in models
            or row["transfer"] not in TRANSFERS
        ):
            raise ValueError("Result is outside the declared comparison grid")
        if not row.get("source"):
            raise ValueError("Every measured result must identify its evidence source")
        seed, score = int(row["seed"]), float(row["score"])
        if seed not in seeds or not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError("Invalid seed or score; scores must be percentages in [0,100]")
        key = row["dataset"], row["model"], row["transfer"]
        if seed in cells[key]:
            raise ValueError(f"Duplicate result: {key}, seed={seed}")
        cells[key][seed] = score

    missing = [
        {
            "dataset": dataset,
            "model": model,
            "transfer": transfer,
            "missing_seeds": sorted(set(seeds) - set(cells[dataset, model, transfer])),
        }
        for dataset in datasets
        for model in models
        for transfer in TRANSFERS
        if set(cells[dataset, model, transfer]) != set(seeds)
    ]

    per_dataset = []
    for (dataset, model, transfer), values in sorted(cells.items()):
        if values:
            per_dataset.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "transfer": transfer,
                    "n": len(values),
                    "mean": statistics.mean(values.values()),
                    "seed_std": (
                        statistics.stdev(values.values()) if len(values) > 1 else None
                    ),
                }
            )

    summaries = []
    if not missing:
        for model in models:
            transfer_by_seed = {
                transfer: [
                    statistics.mean(
                        cells[dataset, model, transfer][seed] for dataset in datasets
                    )
                    for seed in seeds
                ]
                for transfer in TRANSFERS
            }
            setting_by_seed = {
                setting: [
                    statistics.mean(
                        transfer_by_seed[transfer][seed_index]
                        for transfer in transfers
                    )
                    for seed_index in range(len(seeds))
                ]
                for setting, transfers in SETTINGS.items()
            }
            overall_by_seed = [
                statistics.mean(
                    setting_by_seed[setting][seed_index] for setting in SETTINGS
                )
                for seed_index in range(len(seeds))
            ]
            summaries.append(
                {
                    "model": model,
                    "transfers": {
                        transfer: summarize_seed_values(values)
                        for transfer, values in transfer_by_seed.items()
                    },
                    "settings": {
                        setting: summarize_seed_values(values)
                        for setting, values in setting_by_seed.items()
                    },
                    "overall": summarize_seed_values(overall_by_seed),
                }
            )

    return {
        "complete": not missing,
        "missing": missing,
        "per_dataset": per_dataset,
        "summaries": summaries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--grid", type=Path, required=True, help="JSON with datasets, models, and seeds"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grid = json.loads(args.grid.read_text())
    if any(
        not grid[key] or len(grid[key]) != len(set(grid[key]))
        for key in ("datasets", "models", "seeds")
    ):
        parser.error("Grid dimensions must be nonempty and unique")

    with args.csv.open(newline="") as stream:
        result = aggregate(list(csv.DictReader(stream)), **grid)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["complete"]:
        raise SystemExit(
            "Incomplete measured grid: see missing cells; no aggregate rankings produced"
        )


if __name__ == "__main__":
    main()
