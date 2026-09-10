# Brick-Kiln Sentinel-1 partner finder

The original Brick-Kiln Sentinel-2 chips are spatial crops from a temporal mean
composite made over 2018-10-01 through 2019-05-31. A chip therefore has a known
date range, but not one acquisition date. The partner finder searches public
Sentinel-1 RTC scenes over that range, crops each scene to the exact HDF5
footprint, and ranks candidates by valid coverage and optical/SAR edge
correspondence.

The script explicitly handles a legacy metadata inconsistency in this dataset:
the stored arrays already display north-up even though their affine metadata
has a positive vertical step. Remote Sentinel-1 data are reprojected directly
to a north-up grid and are not flipped. The source HDF5 files are never changed.

Each candidate must have at least 95% valid footprint coverage and 90% usable
backscatter. Zero RTC power is treated as nodata. Candidate ranking first
maximizes valid coverage and then the Pearson correlation between Sobel edge
magnitudes of robustly scaled optical RGB--NIR and SAR VV--VH averages. The
selection uses no labels.

Install Python 3.10+ with `h5py`, `numpy`, `rasterio`, `scipy`, and `affine`,
then run:

```bash
python find_s1_partners.py /path/to/x-brick-kiln /path/to/output \
  --mode both --max-candidates 12
```

Use `--limit 5` for an initial visual audit. Use `--dry-run` to inspect scene
availability without reading imagery.

The script queries the Microsoft Planetary Computer STAC API and reads only the
required windows from Sentinel-1 RTC cloud-optimized GeoTIFFs. It does not
download whole source tiles. Use `--insecure-tls` only on a host whose outbound
proxy uses a locally signed TLS certificate.

If the discarded per-sample reference dates are recovered, provide a CSV with
`sample_id,date` columns:

```bash
/home/hrant/.conda/envs/rs_finetune/bin/python find_s1_partners.py INPUT OUTPUT \
  --dates-csv brick_kiln_reference_dates.csv --date-window-days 8
```

Outputs:

- `best/<sample>.hdf5`: best single acquisition, VV/VH in dB.
- `median/<sample>.hdf5`: temporal median from one relative orbit in linear
  power, then converted to dB. Ascending and descending viewing geometries are
  never mixed.
- `candidates.csv`: every evaluated scene, orbit, coverage, score, and selection.

The temporal median is useful for visual inspection because it suppresses
speckle and is conceptually closer to the temporal mean optical input. It should
only replace benchmark data after a full visual and quantitative audit.

Use `--num-shards N --shard-index I` to divide the dataset into contiguous,
non-overlapping shards for parallel execution. Contiguous shards improve reuse
of remote COG blocks for geographically neighboring crops. Existing complete
outputs are skipped, and an example with no candidate passing the quality
checks remains absent so a later run can retry it.

## Add reviewed S1 data to the public HDF5 files

After reviewing the matches, inject the selected best-scene arrays into a copy
of the Brick-Kiln dataset:

```bash
python inject_s1_into_hdf5.py /path/to/x-brick-kiln /path/to/output/best
```

This atomically adds `13 - VH`, `14 - VV`, `S1 valid mask`, and prefixed S1
provenance attributes to each HDF5 file. Every copied array is validated before
the destination file is replaced, and completed files are skipped when the
command is resumed.
