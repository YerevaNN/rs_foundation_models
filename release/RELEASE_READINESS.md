# Public-release readiness

**Status: blocked.** The package is useful and reviewable, but it does not yet support the manuscript claim that the complete benchmark is publicly reproducible.

## Ready

- Nine Dataverse records pin 1,372 unrestricted files, including versions, sizes, checksums, and publisher-declared licenses.
- The public `yerevann/ChiViT` checkpoint is pinned at revision `dad320e530f45964a03b2a4c863e65217a989dfd`. Its 1,539,025,955-byte payload has SHA-256 `0fd0913d069bc236183c1bcb025c8319f62578af45fc814aeb8d9a828bb4ef6f`, exactly matching the evaluated checkpoint. Apache-2.0 metadata and license text were published in model-repository commit `9d98acccc47015b1951a812baf0e0793a7423fa7`.
- Relative splits, preprocessing constants, the five-seed grid, the three-setting aggregation rule, and measured-only aggregation utilities are packaged.
- TerraMind is pinned to its separate branch and has not been merged. A 118-package hashed CPU environment builds the wrapper and completes a finite forward pass; 690/700 historical exported values match the live sheet.

## Publication blockers

1. **Final results provenance (authors).** Produce one measured-only CSV covering the declared 9,800 cells. Each row should trace to a run, selected checkpoint, code revision, and configuration. Reconcile the audit's 295 changed cells and 915 cells absent from the fresh exporter. Do not publish imputed cells as runs.
2. **Portable benchmark runtime (Hakob).** The recovered implementation still contains cluster paths and incomplete requirements. Replace them with configuration, lock the final environment, and run clean train/evaluate smoke tests for classification, segmentation, and change detection.
3. **Harvey release version (authors).** Version 1.0 has the same 930-sample universe and 375/94/461 sizes, but only 151/18/232 samples remain in the same train/validation/test partition as the confirmed benchmark split. Publish the confirmed relative split as a new Dataverse version; then update its version, file IDs, and checksums in `datasets.json`. See `harvey-split-comparison.json`.
4. **Backbone parity (Hakob).** `backbones.json` covers all 14 models with loading instructions. χViT is byte-verified; seven entries pin public filename-matched candidates; six remain unresolved. Hash the cluster caches, recover the missing Panopticon integration, resolve the iBOT MillionAID-versus-ImageNet inconsistency, and pin floating timm/torch.hub loaders.
5. **Payload parity (Hakob).** Download and extract all pinned datasets, then compare them with the exact benchmark inputs used by the reported runs.
6. **TerraMind parity (Hakob).** The new CPU environment passes wrapper construction and a forward test, and 690/700 historical values match. Recover provenance for the ten changed Sen1Floods11 cells, hash the TerraMind cache, and run task-level checkpoint evaluations on CUDA without merging the branch.

## TMLR anonymous review package

Keep the named GitHub, Hugging Face, Dataverse, and arXiv links out of the double-blind review manuscript. Submit a source archive without `.git` history, author names, usernames, organization URLs, cluster paths, acknowledgements, or other identity-bearing metadata. Give reviewers an anonymous artifact link if TMLR's submission form supports one. Keep immutable checksums and technical version identifiers, with a private mapping to the named public records for later release.

The named GitHub repository can remain public, but citing it from the anonymous manuscript would reveal the authors. After review, replace the anonymous artifact references with the named GitHub, Dataverse, and `yerevann/ChiViT` records.
