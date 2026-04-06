#!/bin/bash -l
#SBATCH --job-name=tmlr_harvey-b_prithvi_s2_f
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=50:00:00
#SBATCH --partition=all
#SBATCH --output=/mnt/weka/akhosrovyan/logs_geocrossbench/tmlr_x-harvey-building_prithvi_s2_full_%j.log
#SBATCH --array=0-23

set -euo pipefail

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate rs_finetune

lrs=(1e-4 1e-5 3e-4 3e-5 5e-4 5e-5 6e-4 6e-5)
upernet_widths=(1 2 3)
total_configs=$((${#lrs[@]} * ${#upernet_widths[@]}))
: "${SLURM_ARRAY_TASK_ID:=0}"
if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -ge "$total_configs" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range [0,$((total_configs-1))]"
  exit 1
fi
lr_idx=$((SLURM_ARRAY_TASK_ID / ${#upernet_widths[@]}))
upernet_idx=$((SLURM_ARRAY_TASK_ID % ${#upernet_widths[@]}))
lr=${lrs[$lr_idx]}
upernet_width=${upernet_widths[$upernet_idx]}
seed=42
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
  TMLR_x-harvey-building_prithvi_s2_full_seed${seed}_bs8_ep100_lr${lr}_uw${upernet_width} \
  --dataset_name \
  harvey \
  --dataset_path \
  /nfs/h100/raid/rs/harvey/harvey \
  --metadata_path \
  /nfs/h100/raid/rs/metadata_harvey \
  --backbone \
  prithvi \
  --encoder_weights \
  HLS-2 \
  --batch_size \
  8 \
  --weight_decay \
  0.0005 \
  --lr \
  $lr \
  --lr_sched \
  warmup_cosine \
  --warmup_steps \
  20 \
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
  --max_epochs \
  100 \
  --loss_type \
  ce \
  --img_size \
  224 \
  --upernet_width \
  $upernet_width \
  --classes \
  2 \
  --enable_multiband_input \
  --multiband_channel_count \
  12

python \
  eval_bands_seg.py \
  --model_config \
  ./configs/prithvi.json \
  --dataset_config \
  ./configs/harvey.json \
  --checkpoint_path \
  /nfs/h100/raid/rs/ckpt_rs_finetune/segmentation/TMLR_x-harvey-building_prithvi_s2_full_seed${seed}_bs8_ep100_lr${lr}_uw${upernet_width}/best_model.pth \
  --size \
  224 \
  --classes \
  2 \
  --filename \
  logs_ICLR/seg/TMLR_x-harvey-building_prithvi_s2_full_ \
  --upernet_width \
  $upernet_width \
  --bands \
  '[["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"], ["VV", "VH"], ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "VV", "VH"]]' \
  --enable_multiband_input \
  --multiband_channel_count \
  12
