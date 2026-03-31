#!/bin/bash
set -euo pipefail
#SBATCH --job-name=tmlr_harvey-f_resnet50_s2_h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=50:00:00
#SBATCH --partition=all
#SBATCH --output=/nfs/ap/mnt/frtn/logs_anna/tmlr_x-harvey-flood_resnet-50_s2_head_%j.log
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
  train_change.py \
  --experiment_name \
  TMLR_x-harvey-flood_resnet-50_s2_head_${seed} \
  --mode \
  vanilla \
  --dataset_name \
  harvey \
  --dataset_path \
  /nfs/h100/raid/rs/harvey/harvey \
  --metadata_path \
  /nfs/h100/raid/rs/metadata_harvey \
  --backbone \
  timm_resnet50 \
  --encoder_weights \
  imagenet \
  --fusion \
  diff \
  --lr_sched \
  warmup_cosine \
  --warmup_steps \
  20 \
  --weight_decay \
  0.0005 \
  --lr \
  5e-4 \
  --warmup_lr \
  0.000001 \
  --bands \
  B2 \
  B3 \
  B4 \
  B5 \
  B6 \
  B7 \
  B8 \
  B8A \
  B11 \
  B12 \
  --batch_size \
  8 \
  --max_epochs \
  1 \
  --img_size \
  224 \
  --seed \
  $seed \
  --upernet_width \
  64 \
  --freeze_encoder \
  --enable_multiband \
  --multiband_channel_count \
  12

python \
  eval_bands_cd.py \
  --model_config \
  ./configs/timm_resnet50.json \
  --dataset_config \
  ./configs/harvey.json \
  --checkpoint_path \
  /nfs/h100/raid/rs/ckpt_rs_finetune/change_detection/TMLR_x-harvey-flood_resnet-50_s2_head_${seed}/best_model.pth \
  --size \
  224 \
  --filename \
  logs_ICLR/cd/TMLR_x-harvey-flood_resnet-50_s2_head_ \
  --bands \
  '[["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"], ["vv", "vh"], ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "vv", "vh"]]' \
  --enable_multiband \
  --multiband_channel_count \
  12
