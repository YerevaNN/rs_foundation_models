#!/bin/bash
set -euo pipefail
#SBATCH --job-name=tmlr_eurosat_SatlasNet_rgb_h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=50:00:00
#SBATCH --partition=all
#SBATCH --output=/nfs/ap/mnt/frtn/logs_anna/tmlr_x-eurosat_SatlasNet_rgb_head_%j.log
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
  TMLR_x-eurosat_SatlasNet_rgb_head_${seed} \
  --dataset_name \
  m_eurosat \
  --in_features \
  1024 \
  --backbone \
  Swin-B \
  --encoder_weights \
  satlas \
  --batch_size \
  64 \
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
  --seed \
  $seed \
  --image_size \
  224 \
  --base_dir \
  /nfs/ap/mnt/frtn/rs-multiband/ben/classification_v1.0.0/m-eurosat/ \
  --checkpoint_path \
  /nfs/h100/raid/rs/ckpt_rs_finetune \
  --only_head \
  --checkpoint_path \
  /nfs/ap/mnt/frtn/rs-base-models/satlas_model/sentinel2_swinb_si_rgb.pth

python \
  eval_bands_cls.py \
  --model_config \
  ./configs/swin-B-satlas.json \
  --dataset_config \
  ./configs/m_eurosat.json \
  --checkpoint_path \
  /nfs/h100/raid/rs/ckpt_rs_finetune/classification/TMLR_x-eurosat_SatlasNet_rgb_head_${seed}/best-model.ckpt \
  --img_size \
  224 \
  --filename \
  logs_ICLR/cls/TMLR_x-eurosat_SatlasNet_rgb_head_ \
  --bands \
  '[["B02", "B03", "B04"], ["VV", "VH"], ["B8A", "B11", "B12"], ["B02", "B03", "B04", "B08"]]'
