#!/bin/bash
set -euo pipefail
#SBATCH --job-name=tmlr_brick-ki_SatlasNet_s2_f
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=50:00:00
#SBATCH --partition=all
#SBATCH --output=/nfs/ap/mnt/frtn/logs_anna/tmlr_x-brick-kiln_SatlasNet_s2_full_%j.log
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
  TMLR_x-brick-kiln_SatlasNet_s2_full_${seed} \
  --dataset_name \
  m_brick \
  --in_features \
  1024 \
  --backbone \
  Swin-B \
  --encoder_weights \
  satlas_ms \
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
  /nfs/h100/raid/rs/geobench/brick-kiln/ \
  --checkpoint_path \
  /nfs/h100/raid/rs/ckpt_rs_finetune \
  --enable_multiband_input \
  --multiband_channel_count \
  10

python \
  eval_bands_cls.py \
  --model_config \
  ./configs/swin-B-satlas-ms.json \
  --dataset_config \
  ./configs/m_brick.json \
  --checkpoint_path \
  /nfs/h100/raid/rs/ckpt_rs_finetune/classification/TMLR_x-brick-kiln_SatlasNet_s2_full_${seed}/best-model.ckpt \
  --img_size \
  224 \
  --filename \
  logs_ICLR/cls/TMLR_x-brick-kiln_SatlasNet_s2_full_ \
  --bands \
  '[["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"], ["VV", "VH"], ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "VV", "VH"]]' \
  --enable_multiband_input \
  --multiband_channel_count \
  10
