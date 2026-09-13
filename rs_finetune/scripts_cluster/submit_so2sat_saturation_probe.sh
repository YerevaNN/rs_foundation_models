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
export CHANNELVIT_REPO=${CHANNELVIT_REPO:-/home/hrant/vendor/ChannelViT}

models=(SatlasNet anysat chivit croma dinov2 dinov3 dofa ibot prithvi resnet-50 terrafm vit-b panopticon)
configs=(swin-B-satlas-ms anysat cvit-pretrained croma dinov2 dinov3 dofa ibot-B prithvi timm_resnet50 terrafm timm_vit-b panopticon)
lrs=(5e-3 3e-3 1e-3 1e-3 1e-3 3e-3 1e-3 4e-4 4e-3 1e-3 5e-4 1e-3 1e-4)
image_sizes=(224 224 224 120 224 224 224 224 224 224 224 224 224)
channel_counts=(10 10 10 12 12 12 12 12 12 10 12 10 10)

i=${SLURM_ARRAY_TASK_ID}
model=${models[$i]}
config=${configs[$i]}
lr=${lrs[$i]}
image_size=${image_sizes[$i]}
channel_count=${channel_counts[$i]}
experiment="x-so2sat_${model}_s2_head"
checkpoint="$checkpoint_root/$experiment/seed42_bs64_ep50_lr${lr}/best-model.ckpt"

mkdir -p "$output_root/logs" "$output_root/features/$model" "$output_root/results"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /home/hrant/.conda/envs/rs_finetune
cd "$repo/rs_finetune"

extra_export_args=()
if [[ -n "${GCB_MAX_SAMPLES:-}" ]]; then
  extra_export_args+=(--max-samples "$GCB_MAX_SAMPLES")
fi
if [[ "$model" == "chivit" ]]; then
  extra_export_args+=(--shared-proj --add-ch-embed)
fi

python tools/export_so2sat_frozen_features.py \
  --model-config "configs/${config}.json" \
  --checkpoint "$checkpoint" \
  --output-dir "$output_root/features/$model" \
  --image-size "$image_size" \
  --multiband-channel-count "$channel_count" \
  --batch-size 128 \
  "${extra_export_args[@]}"

python tools/so2sat_saturation_probe.py \
  --model "$model" \
  --features "$output_root/features/$model" \
  --output "$output_root/results/$model.csv"
