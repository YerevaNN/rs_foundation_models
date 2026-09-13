# x-So2Sat S1 saturation probe

This experiment freezes each backbone, trains linear probes on four x-So2Sat
training regimes, and evaluates every probe on the same Sentinel-1 test set.
The regimes are RGB, full Sentinel-2, Sentinel-1, and a balanced mixture of
RGB, Sentinel-2, Sentinel-1, and N'S1S2 views.

The mixture is size matched. Within every class, the physical training samples
are shuffled and divided among the four views, so each sample appears exactly
once and the mixture contains the same number of labels as a single-view
regime. Five assignments are produced with seeds 42, 123, 322, 456, and 789.

`export_so2sat_frozen_features.py` loads the frozen S2-head checkpoint and
exports aligned features for every split and view. `so2sat_saturation_probe.py`
selects logistic-regression regularization on the validation split and reports
Sentinel-1 test accuracy. The Slurm launcher runs the 13 backbones on the main
branch; TerraMind remains in its separate environment and uses the same
feature-file and probe interfaces.

Before submission, create the Slurm log directory:

```bash
mkdir -p /mnt/weka/hrant/geocrossbench/so2sat-saturation/logs
sbatch rs_finetune/scripts_cluster/submit_so2sat_saturation_probe.sh
```
