---
library_name: pytorch
pipeline_tag: image-feature-extraction
license: apache-2.0
---

# ChiViT

ChiViT is a band-agnostic vision transformer pretrained for remote-sensing representation learning and evaluated as a supporting baseline in GeoCrossBench.

## Checkpoint

- Repository: `yerevann/ChiViT`
- Pinned revision: `dad320e530f45964a03b2a4c863e65217a989dfd`
- File: `checkpoint.pth`
- Size: 1,539,025,955 bytes
- SHA-256: `0fd0913d069bc236183c1bcb025c8319f62578af45fc814aeb8d9a828bb4ef6f`

The checksum matches the checkpoint used by the recovered GeoCrossBench evaluator.

## Training summary

The saved checkpoint metadata records a 400,000,000-sample budget, 781,200 completed optimizer iterations, batch size 16 per GPU, eight workers, and gradient accumulation of four. This corresponds to 399,974,400 processed samples. The model used self-supervised pretraining with band sampling over remote-sensing imagery, including paired optical and SAR data.

The standalone repository launcher specifies 500M samples and is not the final run configuration. See `chivit-training.json` for the checkpoint-derived record.

## Evaluation and limitations

ChiViT was evaluated on GeoCrossBench across in-distribution, no-overlap, and superset band-transfer settings. Benchmark results mix task-specific metrics; consult `protocol.json` before interpreting aggregate scores.

A post-hoc audit found that nearly all images in the GeoBench BigEarthNet test split appeared without labels in ChiViT's self-supervised pretraining corpus. BigEarthNet labels were not used for pretraining. Report this overlap whenever publishing BigEarthNet results and avoid treating that task as a clean unseen-data test.

The released checkpoint does not establish performance on future sensors or satellites beyond the band and modality shifts measured by GeoCrossBench.

## Intended use

Use this checkpoint for research on remote-sensing representation learning and cross-band transfer. Downstream users must validate sensor calibration, preprocessing, spatial alignment, and task suitability for their own data.

## License

The ChiViT checkpoint is released under the Apache License 2.0. See LICENSE in the model repository.
