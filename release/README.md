# GeoCrossBench release candidate

This directory is the minimal release package for GeoCrossBench. It pins the public data and ChiViT checkpoint, records the benchmark protocol and splits, and provides checksum and measured-result aggregation tools. It is still a candidate: [RELEASE_READINESS.md](RELEASE_READINESS.md) lists the remaining publication blockers.

## Download verified assets

```sh
python release/download.py --dataset all --output downloads --list
python release/download.py --dataset x-eurosat --output downloads
python release/download.py --chivit --output weights/chivit
```

Downloads are first written as `.partial`, verified against the recorded byte count and checksum, and then promoted. `datasets.json` pins nine Harvard Dataverse records and 1,372 unrestricted files. Harvey provides two tasks, yielding ten tasks. `chivit.json` pins the byte-identical public checkpoint at `yerevann/ChiViT`.

## Protocol and results

`protocol.json` is the normative release-candidate specification for band sets, transfers, task metrics, seeds, HPO grids, and aggregation. `grid.json` declares the 10-task by 28-model/regime by 7-transfer by 5-seed result grid.

Result CSV files use:

```
dataset,model,transfer,seed,score,status,source
```

Scores are percentage points. `status` must be `measured`; imputed values are rejected. `source` must identify the run-level evidence. Aggregate rankings are withheld when any declared grid cell is missing.

```sh
python release/aggregate.py runs.csv --grid release/grid.json --output coverage.json
python -m unittest discover -s release -p 'test_*.py'
```

The aggregator computes every setting within each seed, then reports the mean and sample standard deviation across the five seed aggregates. Overall AVG gives equal weight to the three settings.

## Splits and preprocessing

`splits/` contains the recovered relative splits and available statistics. The GeoBench-derived JSON partitions match the public files semantically, and OSCD split membership matches the public files. The Harvey files are the author-confirmed benchmark split, but this version still needs publication as an immutable Dataverse revision. `preprocessing-constants.json` indexes constants recovered from the implementation; it does not replace the per-model adaptation documentation.

## ChiViT and TerraMind

`CHIVIT_MODEL_CARD.md` is a publication-ready draft except for the license decision. `chivit-training.json` records checkpoint-derived evidence for the 400M-sample run.

TerraMind remains separate by author decision. `TERRAMIND.md` pins branch `terramind` at commit `af2a54f7177c6bb5c68291f019de321b1c308fad`. Do not merge it into the main benchmark branch.
