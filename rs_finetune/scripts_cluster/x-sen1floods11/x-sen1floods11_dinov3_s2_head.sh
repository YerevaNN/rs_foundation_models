#!/bin/bash
set -euo pipefail
#SBATCH --job-name=tmlr_sen1floo_dinov3_s2_h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=50:00:00
#SBATCH --partition=all
#SBATCH --output=/nfs/ap/mnt/frtn/logs_anna/tmlr_x-sen1floods11_dinov3_s2_head_%j.log
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

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv-endpoint=localhost:${RDZV_PORT} \
  train_segmenter.py \
  --seed \
  $seed \
  --experiment_name \
  TMLR_x-sen1floods11_dinov3_s2_head_${seed} \
  --dataset_name \
  sen1floods11 \
  --dataset_path \
  /nfs/ap/mnt/frtn/rs-multiband/sen1floods11/sen1floods11 \
  --metadata_path \
  /nfs/ap/mnt/frtn/rs-multiband/sen1floods11_metadata \
  --backbone \
  dinov3_vitb16 \
  --encoder_weights \
  imagenet \
  --batch_size \
  8 \
  --weight_decay \
  0.05 \
  --lr \
  1e-4 \
  --lr_sched \
  warmup_cosine \
  --bands \
  B04 \
  B03 \
  B02 \
  B05 \
  B06 \
  B07 \
  B08 \
  B8A \
  B11 \
  B12 \
  --max_epochs \
  1 \
  --loss_type \
  ce \
  --img_size \
  224 \
  --upernet_width \
  256 \
  --classes \
  2 \
  --freeze_encoder \
  --enable_multiband_input \
  --multiband_channel_count \
  12

python \
  eval_bands_seg.py \
  --model_config \
  ./configs/dinov3.json \
  --dataset_config \
  ./configs/sen1floods11.json \
  --checkpoint_path \
  /nfs/h100/raid/rs/ckpt_rs_finetune/segmentation/TMLR_x-sen1floods11_dinov3_s2_head_${seed}/best_model.pth \
  --size \
  224 \
  --classes \
  2 \
  --filename \
  logs_ICLR/seg/TMLR_x-sen1floods11_dinov3_s2_head_ \
  --upernet_width \
  256 \
  --bands \
  '[["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"], ["VV", "VH"], ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "VV", "VH"]]' \
  --enable_multiband_input \
  --multiband_channel_count \
  12
