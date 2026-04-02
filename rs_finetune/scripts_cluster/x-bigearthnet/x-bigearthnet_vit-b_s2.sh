#!/bin/bash -l
#SBATCH --job-name=tmlr_bigearth_vitb_s2_f
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=50:00:00
#SBATCH --partition=all
#SBATCH --output=/mnt/weka/akhosrovyan/logs_geocrossbench/tmlr_x-bigearthnet_vit-b_s2_full_%j.log
#SBATCH --array=0-7

set -euo pipefail

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate rs_finetune

lrs=(1e-4 1e-5 3e-4 3e-5 5e-4 5e-5 6e-4 6e-5)
: "${SLURM_ARRAY_TASK_ID:=0}"
if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -ge "${#lrs[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range [0,$((${#lrs[@]}-1))]"
  exit 1
fi
lr=${lrs[$SLURM_ARRAY_TASK_ID]}
seed=42
RDZV_PORT=$((40000 + (RANDOM % 20000)))
MASTER_PORT=$((20000 + (RANDOM % 20000)))
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

python \
  train_classifier.py \
  --experiment_name \
  bigearthnet/x-bigearthnet_vit-b_s2/seed${seed}_bs64_ep50_lr${lr} \
  --dataset_name \
  m_ben \
  --in_features \
  768 \
  --backbone \
  timm_vit-b \
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
  /nfs/ap/mnt/frtn/rs-multiband/m_ben \
  --enable_multiband_input \
  --multiband_channel_count \
  12

python \
  eval_bands_cls.py \
  --model_config \
  ./configs/timm_vit-b.json \
  --dataset_config \
  ./configs/m_ben.json \
  --checkpoint_path \
  /mnt/weka/akhosrovyan/ckpt_rs_finetune/classification/bigearthnet/x-bigearthnet_vit-b_s2 \
  --img_size \
  224 \
  --filename \
  /mnt/weka/akhosrovyan/logs_geocrossbench/cls/TMLR_x-bigearthnet_vit-b_s2_full_ \
  --bands \
  '[["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"], ["VV", "VH"], ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "VV", "VH"]]' \
  --enable_multiband_input \
  --multiband_channel_count \
  12
