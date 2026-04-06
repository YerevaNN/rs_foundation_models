#!/bin/bash -l
#SBATCH --job-name=tmlr_sen1floo_SatlasNet_rgb_h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=50:00:00
#SBATCH --partition=all
#SBATCH --output=/mnt/weka/akhosrovyan/logs_geocrossbench/tmlr_x-sen1floods11_SatlasNet_rgb_head_%j.log
#SBATCH --array=0-23

set -euo pipefail

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate rs_finetune

lrs=(1e-4 1e-3 3e-4 3e-3 5e-4 5e-3 4e-4 4e-3)
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
  TMLR_x-sen1floods11_SatlasNet_rgb_head_seed${seed}_bs8_ep100_lr${lr}_uw${upernet_width} \
  --dataset_name \
  sen1floods11 \
  --dataset_path \
  /nfs/ap/mnt/frtn/rs-multiband/sen1floods11/sen1floods11 \
  --metadata_path \
  /nfs/ap/mnt/frtn/rs-multiband/sen1floods11_metadata \
  --backbone \
  Swin-B \
  --encoder_weights \
  satlas_ms \
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
  B02 \
  B03 \
  B04 \
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
  --freeze_encoder

python \
  eval_bands_seg.py \
  --model_config \
  ./configs/swin-B-satlas-ms.json \
  --dataset_config \
  ./configs/sen1floods11.json \
  --checkpoint_path \
  /nfs/h100/raid/rs/ckpt_rs_finetune/segmentation/TMLR_x-sen1floods11_SatlasNet_rgb_head_seed${seed}_bs8_ep100_lr${lr}_uw${upernet_width}/best_model.pth \
  --size \
  224 \
  --classes \
  2 \
  --filename \
  logs_ICLR/seg/TMLR_x-sen1floods11_SatlasNet_rgb_head_ \
  --upernet_width \
  $upernet_width \
  --bands \
  '[["B02", "B03", "B04"], ["VV", "VH"], ["B8A", "B11", "B12"], ["B02", "B03", "B04", "B08"]]'
