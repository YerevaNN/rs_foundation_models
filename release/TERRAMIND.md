# TerraMind: separate branch, no merge

Hrant's decision on 2026-09-05: document the TerraMind implementation; do not merge it.

- Repository: https://github.com/YerevaNN/rs_foundation_models
- Branch: `terramind`
- Pinned commit: `af2a54f7177c6bb5c68291f019de321b1c308fad`
- Evaluators: `rs_finetune/eval_bands_cls.py`, `eval_bands_seg.py`, `eval_bands_cd.py`.
- Encoder: `rs_finetune/change_detection_pytorch/encoders/terramind.py`.
- Adaptation helpers: `rs_finetune/utils_terramind_adapt.py`.

The branch modifies shared encoder registration, dataset handling, dense-prediction wrappers, training scripts, and evaluation scripts. A trial merge-tree comparison against public main reported eleven file conflicts. It also imports TerraTorch through shared encoder registration. These facts support preserving separation, but do not establish the author's original reason for creating the branch.

Keep its execution environment separate from the main benchmark environment. The exact historical dependency versions and correspondence to `terramind_eval_results_full.csv` still need verification. The optional bundled backend/launcher prepared during this audit has been withdrawn from the release candidate and archived locally; no GitHub branch merge or push occurred.


## Reproduction environment audit

The branch's `rs_finetune/requirements.txt` is not usable for TerraMind: it pins `timm==0.4.12`, while TerraTorch 1.1.1 requires `timm>=1.0.15`, and it omits TerraTorch and shared-import dependencies. The release candidate therefore provides:

- `terramind-environment.in`: direct Python 3.11 CPU smoke-test requirements;
- `terramind-requirements.lock`: 121 transitive distributions with hashes;
- `test_terramind_smoke.py`: package-version, wrapper-construction, band-mapping, and forward-pass checks.

The local `rpe_index` extension must be compiled from `rs_finetune/change_detection_pytorch/encoders/rpe_ops` after installing the lock. The clean smoke test used Python 3.11.13, PyTorch 2.7.1 CPU, TerraTorch 1.1.1, timm 1.0.20, and Transformers 4.57.3. It constructed the 85,544,448-parameter `terramind_v1_base` wrapper and produced a finite `1 x 768` RGB output. This validates a new CPU reproduction environment; it does not recover the unrecorded historical cluster environment or validate CUDA training.

## Result parity audit

`terramind-result-verification.json` maps the 700 rows in the historical TerraMind export uniquely to the 700 current TerraMind spreadsheet cells. After rounding to the displayed precision, 690 cells match and ten Sen1Floods11 cells differ. The current values for those ten cells need the newer run logs and checkpoint provenance before full result parity can be claimed. The remaining 690 cells are value-level matches, but rerunning them still requires the datasets and selected downstream checkpoints.

### Smoke-test commands

```sh
uv python install 3.11.13
uv venv --python 3.11.13 .venv-terramind
uv pip sync --python .venv-terramind/bin/python --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match release/terramind-requirements.lock
uv pip install --python .venv-terramind/bin/python --no-build-isolation ./rs_finetune/change_detection_pytorch/encoders/rpe_ops
.venv-terramind/bin/python release/test_terramind_smoke.py --branch-source ./rs_finetune
```

The smoke test uses `pretrained=False`; loading the 1.5 GB pretrained base artifact and reproducing downstream metrics remain separate parity checks. The recorded local result is in `terramind-smoke-result.json`.
