# Public-release readiness

**Status: blocked.** The package is useful and reviewable, but it does not yet support the manuscript claim that the complete benchmark is publicly reproducible.

## Ready

- Nine Dataverse records pin 1,372 unrestricted files, including versions, sizes, checksums, and publisher-declared licenses.
- The public `yerevann/ChiViT` checkpoint is pinned at revision `dad320e530f45964a03b2a4c863e65217a989dfd`. Its 1,539,025,955-byte payload has SHA-256 `0fd0913d069bc236183c1bcb025c8319f62578af45fc814aeb8d9a828bb4ef6f`, exactly matching the evaluated checkpoint.
- Relative splits, preprocessing constants, the five-seed grid, the three-setting aggregation rule, and measured-only aggregation utilities are packaged.
- TerraMind is pinned to its separate branch and has not been merged.

## Publication blockers

1. **Final results provenance (authors).** Produce one measured-only CSV covering the declared 9,800 cells. Each row should trace to a run, selected checkpoint, code revision, and configuration. Reconcile the audit's 295 changed cells and 915 cells absent from the fresh exporter. Do not publish imputed cells as runs.
2. **Portable benchmark runtime (Hakob).** The recovered implementation still contains cluster paths and incomplete requirements. Replace them with configuration, lock the final environment, and run clean train/evaluate smoke tests for classification, segmentation, and change detection.
3. **Harvey release version (authors).** Publish the confirmed split as a new Dataverse version; then update its version, file IDs, and checksums in `datasets.json`.
4. **ChiViT license (Hrant).** Add an explicit checkpoint license to `yerevann/ChiViT` and the model card. No license was inferred in this candidate.
5. **Backbone manifest (Hakob).** Pin the URL, immutable revision, checksum, license, and loading procedure for every evaluated backbone and auxiliary package.
6. **Payload parity (Hakob).** Download and extract all pinned datasets, then compare them with the exact benchmark inputs used by the reported runs.
7. **TerraMind parity (Hakob).** Lock the environment for `af2a54f` and verify the reported outputs from that branch, without merging it.

## TMLR anonymous review package

Keep the named GitHub, Hugging Face, Dataverse, and arXiv links out of the double-blind review manuscript. Submit a source archive without `.git` history, author names, usernames, organization URLs, cluster paths, acknowledgements, or other identity-bearing metadata. Give reviewers an anonymous artifact link if TMLR's submission form supports one. Keep immutable checksums and technical version identifiers, with a private mapping to the named public records for later release.

The named GitHub repository can remain public, but citing it from the anonymous manuscript would reveal the authors. After review, replace the anonymous artifact references with the named GitHub, Dataverse, and `yerevann/ChiViT` records.
