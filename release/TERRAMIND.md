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
