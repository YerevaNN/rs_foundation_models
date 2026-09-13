#!/bin/bash -l
#SBATCH --job-name=so2sat-oracle
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-12
#SBATCH --output=/mnt/weka/hrant/geocrossbench/so2sat-saturation/logs/%A_%a.log

set -euo pipefail

repo=${GCB_REPO:-$HOME/rs_foundation_models_saturation}
output_root=${GCB_OUTPUT:-/mnt/weka/hrant/geocrossbench/so2sat-saturation}
checkpoint_root=${GCB_CHECKPOINTS:-/mnt/weka/akhosrovyan/ckpt_rs_finetune/classification/so2sat}

models=(SatlasNet anysat chivit croma dinov2 dinov3 dofa ibot prithvi resnet-50 terrafm vit-b panopticon)
configs=(swin-B-satlas-ms anysat cvit croma dinov2 dinov3 dofa ibot-B prithvi timm_resnet50 terrafm timm_vit-b panopticon)
lrs=(5e-3 3e-3 1e-3 1e-3 1e-3 3e-3 1e-3 4e-4 4e-3 1e-3 5e-4 1e-3 1e-4)

i=${SLURM_ARRAY_TASK_ID}
model=${models[$i]}
config=${configs[$i]}
lr=${lrs[$i]}
experiment="x-so2sat_${model}_s2_head"
checkpoint="$checkpoint_root/$experiment/seed42_bs64_ep50_lr${lr}/best-model.ckpt"

mkdir -p "$output_root/logs" "$output_root/features/$model" "$output_root/results"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /home/hrant/.conda/envs/rs_finetune
cd "$repo/rs_finetune"

python tools/export_so2sat_frozen_features.py \
  --model-config "configs/${config}.json" \
  --checkpoint "$checkpoint" \
  --output-dir "$output_root/features/$model" \
  --image-size 224 \
  --batch-size 128

python tools/so2sat_saturation_probe.py \
  --model "$model" \
  --features "$output_root/features/$model" \
  --output "$output_root/results/$model.csv"
