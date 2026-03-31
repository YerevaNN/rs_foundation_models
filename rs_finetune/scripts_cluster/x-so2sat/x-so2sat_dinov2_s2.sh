#!/bin/bash
set -euo pipefail
#SBATCH --job-name=tmlr_so2sat_dinov2_s2_f
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=50:00:00
#SBATCH --partition=all
#SBATCH --output=/nfs/ap/mnt/frtn/logs_anna/tmlr_x-so2sat_dinov2_s2_full_%j.log
#SBATCH --array=0-0
seeds=(42 123 322 456 789)
: "${SLURM_ARRAY_TASK_ID:=0}"
if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -ge "${#seeds[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range [0,$((${#seeds[@]}-1))]"
  exit 1
fi
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
RDZV_PORT=$((40000 + (RANDOM % 20000)))
MASTER_PORT=$((20000 + (RANDOM % 20000)))

python \
  train_classifier.py \
  --experiment_name \
  TMLR_x-so2sat_dinov2_s2_full_${seed} \
  --dataset_name \
  so2sat \
  --in_features \
  768 \
  --backbone \
  dinov2 \
  --encoder_weights \
  imagenet \
  --batch_size \
  16 \
  --accumulate_grad_batches \
  4 \
  --optimizer \
  adamw \
  --scheduler \
  cosine \
  --epoch \
  1 \
  --lr \
  1e-4 \
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
  /nfs/h100/raid/rs/ckpt_rs_finetune \
  --enable_multiband_input \
  --multiband_channel_count \
  12

python \
  eval_bands_cls.py \
  --model_config \
  ./configs/dinov2.json \
  --dataset_config \
  ./configs/so2sat.json \
  --checkpoint_path \
  /nfs/h100/raid/rs/ckpt_rs_finetune/classification/TMLR_x-so2sat_dinov2_s2_full_${seed}/best-model.ckpt \
  --img_size \
  224 \
  --filename \
  logs_ICLR/cls/TMLR_x-so2sat_dinov2_s2_full_ \
  --bands \
  '[["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"], ["VV", "VH"], ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "VV", "VH"]]' \
  --enable_multiband_input \
  --multiband_channel_count \
  12
