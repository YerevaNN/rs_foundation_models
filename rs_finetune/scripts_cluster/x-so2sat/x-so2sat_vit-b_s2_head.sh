#!/bin/bash -l
#SBATCH --job-name=tmlr_so2sat_vitb_s2_h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=50:00:00
#SBATCH --partition=all
#SBATCH --output=/mnt/weka/akhosrovyan/logs_geocrossbench/tmlr_x-so2sat_vit-b_s2_head_%j.log
#SBATCH --array=0-7

set -euo pipefail

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate rs_finetune

lrs=(1e-4 1e-3 3e-4 3e-3 5e-4 5e-3 4e-4 4e-3)
: "${SLURM_ARRAY_TASK_ID:=0}"
if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -ge "${#lrs[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range [0,$((${#lrs[@]}-1))]"
  exit 1
fi
lr=${lrs[$SLURM_ARRAY_TASK_ID]}
seed=42
RDZV_PORT=$((40000 + (RANDOM % 20000)))
MASTER_PORT=$((20000 + (RANDOM % 20000)))

python \
  train_classifier.py \
  --experiment_name \
  so2sat/x-so2sat_vit-b_s2_head/seed${seed}_bs64_ep50_lr${lr} \
  --dataset_name \
  so2sat \
  --in_features \
  768 \
  --backbone \
  timm_vit-b \
  --encoder_weights \
  imagenet \
  --batch_size \
  64 \
  --accumulate_grad_batches \
  4 \
  --optimizer \
  adamw \
  --scheduler \
  cosine \
  --epoch \
  50 \
  --lr \
  $lr \
  --bands \
  B02 \
  B03 \
  B04 \
  B05 \
  B06 \
  B07 \
  B08 \
  B8A \
  B11 \
  B12 \
  --seed \
  $seed \
  --image_size \
  224 \
  --base_dir \
  /nfs/ap/mnt/frtn/rs-multiband/geobench/classification_v1.0/m-so2sat \
  --checkpoint_path \
  /mnt/weka/akhosrovyan/ckpt_rs_finetune/classification/so2sat/x-so2sat_vit-b_s2_head \
  --only_head \
  --enable_multiband_input \
  --multiband_channel_count \
  12

python \
  eval_bands_cls.py \
  --model_config \
  ./configs/timm_vit-b.json \
  --dataset_config \
  ./configs/so2sat.json \
  --checkpoint_path \
  /mnt/weka/akhosrovyan/ckpt_rs_finetune/classification/so2sat/x-so2sat_vit-b_s2_head/seed${seed}_bs64_ep50_lr${lr}/best-model.ckpt \
  --img_size \
  224 \
  --filename \
  /mnt/weka/akhosrovyan/logs_geocrossbench/cls/TMLR_x-so2sat_vit-b_s2_head_ \
  --bands \
  '[["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"], ["VV", "VH"], ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "VV", "VH"]]' \
  --enable_multiband_input \
  --multiband_channel_count \
  12
