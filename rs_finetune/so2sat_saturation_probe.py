#!/usr/bin/env python3
"""Fit size-controlled frozen-feature probes for the x-so2sat S1 oracle test.

Expected input is one ``.npz`` per split and band view.  Each file contains
``features`` (N x D), ``labels`` (N,), and optionally ``sample_ids`` (N,).
The four views must describe the same split in the same sample order.

The balanced mixture has exactly N examples, like every single-view regime.
Within each class, samples are shuffled and divided as evenly as possible
among RGB, S2, S1, and N'S1S2.  Every physical training sample therefore
appears exactly once in the mixture, under one randomly assigned view.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VIEWS = ("rgb", "s2", "s1", "ns1s2")
REGIMES = ("rgb", "s2", "s1", "mixture")


def load_view(root: Path, split: str, view: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = root / f"{split}__{view}.npz"
    data = np.load(path, allow_pickle=False)
    features = np.asarray(data["features"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
    sample_ids = (
        np.asarray(data["sample_ids"]).astype(str)
        if "sample_ids" in data
        else np.arange(len(labels)).astype(str)
    )
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError(f"Malformed feature file: {path}")
    return features, labels, sample_ids


def load_split(root: Path, split: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    views = {view: load_view(root, split, view) for view in VIEWS}
    ref_y, ref_ids = views[VIEWS[0]][1:]
    for view in VIEWS[1:]:
        _, y, ids = views[view]
        if not np.array_equal(y, ref_y) or not np.array_equal(ids, ref_ids):
            raise ValueError(f"{split}/{view} is not aligned with {split}/{VIEWS[0]}")
    dims = {x.shape[1] for x, _, _ in views.values()}
    if len(dims) != 1:
        raise ValueError(f"Feature dimensions differ across {split} views: {sorted(dims)}")
    return views


def balanced_mixture(
    views: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    labels = views["rgb"][1]
    selected_x: list[np.ndarray] = []
    selected_y: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for view, view_indices in zip(VIEWS, np.array_split(indices, len(VIEWS))):
            selected_x.append(views[view][0][view_indices])
            selected_y.append(labels[view_indices])
    x = np.concatenate(selected_x)
    y = np.concatenate(selected_y)
    order = rng.permutation(len(y))
    if len(y) != len(labels):
        raise AssertionError("The size-matched mixture must contain exactly N examples")
    return x[order], y[order]


def train_matrix(
    train: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], regime: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if regime == "mixture":
        return balanced_mixture(train, seed)
    return train[regime][0], train[regime][1]


def fit_probe(x: np.ndarray, y: np.ndarray, c: float, seed: int) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        ),
    ).fit(x, y)


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    train = load_split(args.features, "train")
    valid = load_split(args.features, "valid")
    test = load_split(args.features, "test")
    test_x, test_y, _ = test["s1"]
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for regime in REGIMES:
            train_x, train_y = train_matrix(train, regime, seed)
            valid_x, valid_y = train_matrix(valid, regime, seed)
            best_c, best_valid = None, -np.inf
            for c in args.c_values:
                probe = fit_probe(train_x, train_y, c, seed)
                score = accuracy_score(valid_y, probe.predict(valid_x))
                if score > best_valid:
                    best_c, best_valid = c, score
            probe = fit_probe(train_x, train_y, float(best_c), seed)
            test_score = accuracy_score(test_y, probe.predict(test_x))
            rows.append(
                {
                    "model": args.model,
                    "seed": seed,
                    "train_regime": regime,
                    "train_size": len(train_y),
                    "best_c": best_c,
                    "valid_accuracy": 100 * best_valid,
                    "s1_test_accuracy": 100 * test_score,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 322, 456, 789])
    parser.add_argument("--c-values", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    args = parser.parse_args()
    rows = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for regime in REGIMES:
        values = [float(row["s1_test_accuracy"]) for row in rows if row["train_regime"] == regime]
        summary[regime] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
