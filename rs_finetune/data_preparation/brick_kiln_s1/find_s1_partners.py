#!/usr/bin/env python3
"""Find spatially aligned Sentinel-1 partners for GeoBench Brick-Kiln chips.

The Brick-Kiln Sentinel-2 inputs are temporal mean composites made from imagery
between 2018-10-01 and 2019-05-31. They therefore do not have one acquisition
date. This program searches Sentinel-1 RTC scenes over that interval, reads only
the requested chip from each cloud-optimized GeoTIFF, and ranks candidates by
valid coverage and optical/SAR edge correspondence.

If a CSV containing recovered reference dates is supplied, the search is
restricted to +/- ``--date-window-days`` around each date instead.

The source HDF5 files are never modified. Each output HDF5 contains the selected
VV/VH arrays and complete provenance; a CSV records all evaluated candidates.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import pickle
import random
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import h5py
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from scipy.ndimage import sobel


PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
COLLECTION = "sentinel-1-rtc"
DEFAULT_START = date(2018, 10, 1)
DEFAULT_END = date(2019, 5, 31)
INSECURE_TLS = False
S2_KEYS = {
    "blue": "02 - Blue",
    "green": "03 - Green",
    "red": "04 - Red",
    "nir": "08 - NIR",
}


@dataclass(frozen=True)
class Grid:
    crs: str
    original_transform: Affine
    north_up_transform: Affine
    bbox: tuple[float, float, float, float]
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find and crop Sentinel-1 RTC partners for Brick-Kiln HDF5 chips."
    )
    p.add_argument("input", type=Path, help="HDF5 file or directory containing *.hdf5")
    p.add_argument("output", type=Path, help="Output directory (source files are untouched)")
    p.add_argument(
        "--dates-csv",
        type=Path,
        help="Optional CSV with sample_id and date (YYYY-MM-DD) columns",
    )
    p.add_argument("--date-window-days", type=int, default=8)
    p.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    p.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    p.add_argument(
        "--max-candidates",
        type=int,
        default=12,
        help="Maximum scenes evaluated per sample; 0 evaluates every scene",
    )
    p.add_argument(
        "--mode",
        choices=("best", "median", "both"),
        default="both",
        help="Write the best single scene, temporal median, or both",
    )
    p.add_argument("--limit", type=int, default=0, help="Process at most this many files")
    p.add_argument("--num-shards", type=int, default=1, help="Split a large run into this many shards")
    p.add_argument("--shard-index", type=int, default=0, help="Zero-based shard to process")
    p.add_argument("--seed", type=int, default=0, help="Stable tie-breaking/subsampling seed")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Search and report without reading imagery")
    p.add_argument("--retries", type=int, default=4)
    p.add_argument(
        "--insecure-tls",
        action="store_true",
        help="Accept the cluster's TLS interception certificate (also passed to GDAL)",
    )
    return p.parse_args()


def http_json(url: str, body: dict[str, Any] | None, retries: int) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"User-Agent": "GeoCrossBench-S1-partner-finder/1.0"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            context = ssl._create_unverified_context() if INSECURE_TLS else None
            with urllib.request.urlopen(req, timeout=90, context=context) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


class Signer:
    def __init__(self, retries: int):
        self.retries = retries
        self.tokens: dict[tuple[str, str], tuple[str, datetime]] = {}

    def sign(self, href: str) -> str:
        parsed = urlparse(href)
        if not parsed.netloc.endswith(".blob.core.windows.net"):
            return href
        account = parsed.netloc.split(".", 1)[0]
        container = parsed.path.lstrip("/").split("/", 1)[0]
        key = (account, container)
        cached = self.tokens.get(key)
        refresh_before = datetime.now(timezone.utc) + timedelta(minutes=5)
        if cached is None or cached[1] <= refresh_before:
            payload = http_json(f"{PC_SAS}/{account}/{container}", None, self.retries)
            expiry_text = payload["msft:expiry"].replace("Z", "+00:00")
            expiry = datetime.fromisoformat(expiry_text)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            self.tokens[key] = (payload["token"], expiry)
        token = self.tokens[key][0]
        separator = "&" if parsed.query else "?"
        return f"{href}{separator}{token}"


def load_dates(path: Path | None) -> dict[str, date]:
    if path is None:
        return {}
    result: dict[str, date] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sample_id = (row.get("sample_id") or row.get("id") or row.get("filename") or "").strip()
            raw_date = (row.get("date") or row.get("s2_date") or row.get("reference_date") or "").strip()
            if not sample_id or not raw_date:
                continue
            result[Path(sample_id).stem] = date.fromisoformat(raw_date[:10])
    return result


def unpack_metadata(h5: h5py.File) -> dict[str, Any]:
    raw = h5.attrs.get("pickle")
    if raw is None:
        raise ValueError("missing HDF5 'pickle' metadata attribute")
    if isinstance(raw, np.void):
        raw = bytes(raw)
    if isinstance(raw, str):
        # GeoBench V1 stores repr(pickle_bytes), rather than the bytes directly.
        if raw.startswith(("b'", 'b"')):
            raw = ast.literal_eval(raw)
        else:
            raw = raw.encode("latin1")
    return pickle.loads(raw)


def grid_from_h5(h5: h5py.File) -> Grid:
    metadata = unpack_metadata(h5)
    band_meta = metadata[S2_KEYS["red"]]
    transform = Affine(*band_meta["transform"][:6])
    crs = str(band_meta["crs"])
    height, width = h5[S2_KEYS["red"]].shape

    corners = [
        transform * (0, 0),
        transform * (width, 0),
        transform * (0, height),
        transform * (width, height),
    ]
    xs, ys = zip(*corners)
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return Grid(
        crs=crs,
        original_transform=transform,
        north_up_transform=from_bounds(*bbox, width, height),
        bbox=bbox,
        width=width,
        height=height,
    )


def optical_reference(h5: h5py.File) -> np.ndarray:
    arrays = []
    for name in ("red", "green", "blue", "nir"):
        a = np.asarray(h5[S2_KEYS[name]], dtype=np.float32)
        # Brick-Kiln arrays already display north-up. The legacy affine's
        # positive y step does not describe their in-file row order.
        arrays.append(robust_scale(a))
    return np.mean(arrays, axis=0)


def robust_scale(a: np.ndarray) -> np.ndarray:
    valid = np.isfinite(a)
    if not valid.any():
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.nanpercentile(a[valid], [2, 98])
    if not np.isfinite(lo + hi) or hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)


def edge_magnitude(a: np.ndarray) -> np.ndarray:
    a = robust_scale(a)
    return np.hypot(sobel(a, axis=0, mode="nearest"), sobel(a, axis=1, mode="nearest"))


def edge_score(optical: np.ndarray, vv_db: np.ndarray, vh_db: np.ndarray, valid: np.ndarray) -> float:
    sar = 0.5 * (robust_scale(vv_db) + robust_scale(vh_db))
    a, b = edge_magnitude(optical), edge_magnitude(sar)
    mask = valid & np.isfinite(a) & np.isfinite(b)
    if mask.sum() < max(64, mask.size // 4):
        return float("nan")
    av, bv = a[mask], b[mask]
    if av.std() == 0 or bv.std() == 0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def search_items(grid: Grid, start: date, end: date, retries: int) -> list[dict[str, Any]]:
    return search_items_cached(grid, start, end, retries, {})[0]


def search_items_cached(
    grid: Grid,
    start: date,
    end: date,
    retries: int,
    cache: dict[tuple[Any, ...], list[dict[str, Any]]],
    cell_degrees: float = 0.25,
) -> tuple[list[dict[str, Any]], str]:
    west, south, east, north = grid.bbox
    cell_x = math.floor(((west + east) / 2) / cell_degrees)
    cell_y = math.floor(((south + north) / 2) / cell_degrees)
    key = (cell_x, cell_y, start.isoformat(), end.isoformat())
    query_bbox = [
        cell_x * cell_degrees,
        cell_y * cell_degrees,
        (cell_x + 1) * cell_degrees,
        (cell_y + 1) * cell_degrees,
    ]
    body = {
        "collections": [COLLECTION],
        "bbox": query_bbox,
        "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        "limit": 1000,
    }
    if key not in cache:
        cache[key] = http_json(f"{PC_STAC}/search", body, retries).get("features", [])
    features = cache[key]
    filtered = [
        item
        for item in features
        if "vv" in item.get("assets", {})
        and "vh" in item.get("assets", {})
        and {p.upper() for p in item.get("properties", {}).get("sar:polarizations", [])} >= {"VV", "VH"}
        and len(item.get("bbox", [])) >= 4
        and item["bbox"][0] <= west
        and item["bbox"][1] <= south
        and item["bbox"][2] >= east
        and item["bbox"][3] >= north
    ]
    return filtered, ":".join(map(str, key))


def stable_candidate_subset(
    items: list[dict[str, Any]], maximum: int, selection_key: str, seed: int
) -> list[dict[str, Any]]:
    items = sorted(items, key=lambda x: (x["properties"]["datetime"], x["id"]))
    if maximum <= 0 or len(items) <= maximum:
        return items

    # Preserve both orbit directions and sample dates across the whole interval.
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(item["properties"].get("sat:orbit_state", "unknown"), []).append(item)
    chosen: list[dict[str, Any]] = []
    for group in groups.values():
        quota = max(1, round(maximum * len(group) / len(items)))
        indices = np.linspace(0, len(group) - 1, min(quota, len(group)), dtype=int)
        chosen.extend(group[i] for i in indices)
    if len(chosen) > maximum:
        digest = hashlib.sha256(f"{selection_key}:{seed}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        rng.shuffle(chosen)
        chosen = chosen[:maximum]
    elif len(chosen) < maximum:
        seen = {x["id"] for x in chosen}
        remaining = [x for x in items if x["id"] not in seen]
        chosen.extend(remaining[: maximum - len(chosen)])
    return sorted(chosen, key=lambda x: (x["properties"]["datetime"], x["id"]))


class AssetReader:
    """Keep recently used COGs open so neighboring crops reuse downloaded blocks."""

    def __init__(self, max_open: int = 32):
        self.max_open = max_open
        options = dict(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
            GDAL_HTTP_MULTIRANGE="YES",
            GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
            VSI_CACHE="TRUE",
            VSI_CACHE_SIZE=8_388_608,
            GDAL_CACHEMAX=268_435_456,
        )
        if INSECURE_TLS:
            options["GDAL_HTTP_UNSAFESSL"] = "YES"
        self.env = rasterio.Env(**options)
        self.sources: OrderedDict[str, rasterio.DatasetReader] = OrderedDict()

    def __enter__(self) -> "AssetReader":
        self.env.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for source in self.sources.values():
            source.close()
        self.sources.clear()
        self.env.__exit__(exc_type, exc, traceback)

    def source(self, href: str) -> rasterio.DatasetReader:
        if href in self.sources:
            source = self.sources.pop(href)
            self.sources[href] = source
            return source
        source = rasterio.open(href)
        self.sources[href] = source
        while len(self.sources) > self.max_open:
            _, old = self.sources.popitem(last=False)
            old.close()
        return source

    def read(self, href: str, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
        src = self.source(href)
        with WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid.north_up_transform,
            width=grid.width,
            height=grid.height,
            resampling=Resampling.bilinear,
            nodata=np.nan,
        ) as vrt:
            masked = vrt.read(1, masked=True, out_dtype="float32")
        data = np.asarray(masked.filled(np.nan), dtype=np.float32)
        # Sentinel-1 RTC uses zero for uncovered/nodata pixels in some assets.
        valid = ~np.ma.getmaskarray(masked) & np.isfinite(data) & (data > 0)
        return data, valid


def linear_to_db(a: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(np.maximum(a, 1e-10))).astype(np.float32)


def evaluate_item(
    item: dict[str, Any], signer: Signer, reader: AssetReader, grid: Grid, optical: np.ndarray
) -> dict[str, Any]:
    vv, vv_valid = reader.read(signer.sign(item["assets"]["vv"]["href"]), grid)
    vh, vh_valid = reader.read(signer.sign(item["assets"]["vh"]["href"]), grid)
    valid = vv_valid & vh_valid
    vv_db, vh_db = linear_to_db(vv), linear_to_db(vh)
    nonblack = valid & ((vv_db > -35.0) | (vh_db > -42.0))
    props = item["properties"]
    return {
        "item_id": item["id"],
        "datetime": props.get("datetime"),
        "orbit_state": props.get("sat:orbit_state"),
        "relative_orbit": props.get("sat:relative_orbit"),
        "platform": props.get("platform"),
        "coverage": float(valid.mean()),
        "nonblack_fraction": float(nonblack.mean()),
        "edge_score": edge_score(optical, vv_db, vh_db, valid),
        "vv_linear": vv,
        "vh_linear": vh,
        "vv_db": vv_db,
        "vh_db": vh_db,
        "valid": valid,
    }


def score_key(result: dict[str, Any]) -> tuple[float, float]:
    edge = result["edge_score"]
    return (result["coverage"], -math.inf if not np.isfinite(edge) else edge)


def serializable_result(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}


def write_h5(
    path: Path,
    grid: Grid,
    arrays: dict[str, np.ndarray],
    provenance: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, array in arrays.items():
            f.create_dataset(name, data=array, compression="gzip")
        f.attrs["provenance_json"] = json.dumps(
            provenance,
            sort_keys=True,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
        f.attrs["crs"] = grid.crs
        f.attrs["transform"] = tuple(grid.original_transform)[:6]
        f.attrs["bbox"] = grid.bbox


def iter_inputs(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    else:
        def natural_key(p: Path) -> tuple[Any, ...]:
            parts = []
            for token in p.stem.replace("-", "_").split("_"):
                parts.append(int(token) if token.isdigit() else token)
            return tuple(parts)

        yield from sorted(path.rglob("*.hdf5"), key=natural_key)


def main() -> int:
    global INSECURE_TLS
    args = parse_args()
    INSECURE_TLS = args.insecure_tls
    if INSECURE_TLS:
        print("warning: TLS certificate verification is disabled", file=sys.stderr)
    args.output.mkdir(parents=True, exist_ok=True)
    dates = load_dates(args.dates_csv)
    signer = Signer(args.retries)
    search_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    files = list(iter_inputs(args.input))
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    # Contiguous shards keep neighboring grid cells together, maximizing COG
    # block reuse within each process.
    shard_size = math.ceil(len(files) / args.num_shards) if files else 0
    shard_start = args.shard_index * shard_size
    files = files[shard_start : shard_start + shard_size]
    if args.limit:
        files = files[: args.limit]

    with AssetReader() as reader:
      for index, source in enumerate(files, 1):
        sample_id = source.stem
        best_path = args.output / "best" / f"{sample_id}.hdf5"
        median_path = args.output / "median" / f"{sample_id}.hdf5"
        wanted = [best_path] if args.mode == "best" else [median_path]
        if args.mode == "both":
            wanted = [best_path, median_path]
        if not args.overwrite and all(p.exists() for p in wanted):
            print(f"[{index}/{len(files)}] skip {sample_id}: outputs exist", flush=True)
            continue

        with h5py.File(source, "r") as h5:
            grid = grid_from_h5(h5)
            optical = optical_reference(h5)

        reference_date = dates.get(sample_id)
        if reference_date:
            start = reference_date - timedelta(days=args.date_window_days)
            end = reference_date + timedelta(days=args.date_window_days)
        else:
            start, end = args.start, args.end

        items, selection_key = search_items_cached(
            grid, start, end, args.retries, search_cache
        )
        candidates = stable_candidate_subset(
            items, args.max_candidates, selection_key, args.seed
        )
        print(
            f"[{index}/{len(files)}] {sample_id}: {len(items)} scenes, "
            f"evaluating {len(candidates)} ({start}..{end})",
            flush=True,
        )
        if args.dry_run:
            for item in candidates:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "source": str(source),
                        "reference_date": reference_date.isoformat() if reference_date else "",
                        "search_start": start.isoformat(),
                        "search_end": end.isoformat(),
                        "item_id": item["id"],
                        "datetime": item["properties"].get("datetime"),
                        "orbit_state": item["properties"].get("sat:orbit_state"),
                        "relative_orbit": item["properties"].get("sat:relative_orbit"),
                        "coverage": "",
                        "edge_score": "",
                        "selected_best": "",
                    }
                )
            continue

        evaluated = []
        for item in candidates:
            try:
                evaluated.append(evaluate_item(item, signer, reader, grid, optical))
            except Exception as exc:
                print(f"  warning: {item['id']}: {exc}", file=sys.stderr, flush=True)
        usable = [
            x for x in evaluated
            if x["coverage"] >= 0.95 and x["nonblack_fraction"] >= 0.90
        ]
        if not usable:
            print(
                "  warning: no candidate passed >=95% valid coverage and "
                ">=90% nonblack signal; leaving output absent for a clean retry",
                file=sys.stderr,
            )
            continue
        best = max(usable, key=score_key)

        common = {
            "sample_id": sample_id,
            "source_hdf5": str(source),
            "reference_date": reference_date.isoformat() if reference_date else None,
            "search_start": start.isoformat(),
            "search_end": end.isoformat(),
            "collection": COLLECTION,
            "stac_api": PC_STAC,
            "selection": "maximum coverage then edge correlation among candidates with >=95% valid coverage and >=90% nonblack signal",
            "candidate_count_found": len(items),
            "candidate_count_evaluated": len(evaluated),
            "stored_north_up_to_match_source_array_display": True,
        }
        if args.mode in ("best", "both"):
            write_h5(
                best_path,
                grid,
                {"VV": best["vv_db"], "VH": best["vh_db"], "valid": best["valid"].astype("uint8")},
                {**common, "product": "best_single_scene", "selected": serializable_result(best)},
            )
        if args.mode in ("median", "both"):
            # Do not mix viewing geometries: ascending/descending passes and
            # relative orbits have different incidence and layover directions.
            orbit_groups: dict[tuple[str, Any], list[dict[str, Any]]] = {}
            for result in usable:
                key = (result["orbit_state"] or "unknown", result["relative_orbit"])
                orbit_groups.setdefault(key, []).append(result)
            composites = []
            for (orbit_state, relative_orbit), group in orbit_groups.items():
                vv_stack = np.stack(
                    [np.where(x["valid"], x["vv_linear"], np.nan) for x in group]
                )
                vh_stack = np.stack(
                    [np.where(x["valid"], x["vh_linear"], np.nan) for x in group]
                )
                vv_median = linear_to_db(np.nanmedian(vv_stack, axis=0))
                vh_median = linear_to_db(np.nanmedian(vh_stack, axis=0))
                valid_median = np.isfinite(vv_median) & np.isfinite(vh_median)
                composites.append(
                    {
                        "orbit_state": orbit_state,
                        "relative_orbit": relative_orbit,
                        "items": group,
                        "VV": vv_median,
                        "VH": vh_median,
                        "valid": valid_median,
                        "coverage": float(valid_median.mean()),
                        "edge_score": edge_score(optical, vv_median, vh_median, valid_median),
                    }
                )
            selected_composite = max(composites, key=score_key)
            write_h5(
                median_path,
                grid,
                {
                    "VV": selected_composite["VV"],
                    "VH": selected_composite["VH"],
                    "valid": selected_composite["valid"].astype("uint8"),
                },
                {
                    **common,
                    "product": "same-relative-orbit_temporal_median_in_linear_power_then_db",
                    "orbit_state": selected_composite["orbit_state"],
                    "relative_orbit": selected_composite["relative_orbit"],
                    "included_items": [
                        serializable_result(x) for x in selected_composite["items"]
                    ],
                    "edge_score": selected_composite["edge_score"],
                },
            )

        for result in evaluated:
            rows.append(
                {
                    "sample_id": sample_id,
                    "source": str(source),
                    "reference_date": reference_date.isoformat() if reference_date else "",
                    "search_start": start.isoformat(),
                    "search_end": end.isoformat(),
                    **serializable_result(result),
                    "selected_best": result["item_id"] == best["item_id"],
                }
            )

    report_name = (
        "candidates.csv"
        if args.num_shards == 1
        else f"candidates-shard-{args.shard_index:04d}-of-{args.num_shards:04d}.csv"
    )
    report = args.output / report_name
    if rows:
        fieldnames = list(rows[0])
        with report.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {report} ({len(rows)} candidate rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
