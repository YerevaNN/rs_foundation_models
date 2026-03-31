#!/bin/bash
set -euo pipefail
#SBATCH --job-name=tmlr_SA-crop-_anysat_rgb_f
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=50:00:00
#SBATCH --partition=all
#SBATCH --output=/nfs/ap/mnt/frtn/logs_anna/tmlr_x-SA-crop-type_anysat_rgb_full_%j.log
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
  TMLR_x-SA-crop-type_anysat_rgb_full_${seed} \
  --dataset_name \
  crop \
  --dataset_path \
  /nfs/h100/raid/rs/geobench/sa_crop_type \
  --metadata_path \
  /nfs/h100/raid/rs/geobench/sa_crop_type \
  --backbone \
  anysat \
  --encoder_weights \
  anysat \
  --batch_size \
  8 \
  --weight_decay \
  0.05 \
  --lr \
  1e-4 \
  --lr_sched \
  warmup_cosine \
  --bands \
  B02 \
  B03 \
  B04 \
  --max_epochs \
  1 \
  --loss_type \
  ce \
  --img_size \
  224 \
  --upernet_width \
  256 \
  --classes \
  10

python \
  eval_bands_seg.py \
  --model_config \
  ./configs/anysat.json \
  --dataset_config \
  ./configs/sa_crop.json \
  --checkpoint_path \
  /nfs/h100/raid/rs/ckpt_rs_finetune/segmentation/TMLR_x-SA-crop-type_anysat_rgb_full_${seed}/best_model.pth \
  --size \
  224 \
  --classes \
  10 \
  --filename \
  logs_ICLR/seg/TMLR_x-SA-crop-type_anysat_rgb_full_ \
  --upernet_width \
  256 \
  --bands \
  '[["B02", "B03", "B04"], ["VV", "VH"], ["B8A", "B11", "B12"], ["B02", "B03", "B04", "B08"]]'
