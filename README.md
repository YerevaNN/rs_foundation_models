# 🛰️ GeoCrossBench and χViT: A Benchmark and Model for Cross-Satellite Generalization in Remote Sensing

## 📊 GeoCrossBench Benchmark

`GeoCrossBench` is a new evaluation protocol that tests:
1.  In-distribution performance.
2.  Generalization to satellites with no band overlap.
3.  Generalization to satellites with additional bands with respect to the training set.

## ⚙️ Setup

Before running the scripts, make sure you have all the required dependencies installed.

```bash
# Example of setting up the environment
pip install -r requirements.txt
```

## 📦 Weights and Datasets

### 🎯 χViT Weights

You can download the pretrained `χViT` weights from the following link:

- [χViT Pretrained Weights](https://huggingface.co/akhosrovyan/ChiViT/tree/main)

### 📥 Datasets

The datasets for the `GeoCrossBench` benchmark can be downloaded from the following link:

- [GeoCrossBench Datasets](https://dataverse.harvard.edu/dataverse/geocrossbench/)

## 🚀 Training

Below are the training scripts for classification, semantic segmentation, and change detection tasks.

### 🏷️ Classification

```bash
python train_classifier.py --experiment_name "terrafm" --dataset_name "m_ben" \
        --in_features "768" --backbone "terrafm-base"  --encoder_weights "terrafm_base" --batch_size 64 \
        --optimizer "adamw"  --scheduler "cosine" --epoch 50  --lr 1e-3 --bands B02 B03 B04  \
        --seed 42 --image_size 224
```

### 🖼️ Semantic Segmentation

```bash
torchrun --nnodes=1 --nproc_per_node=1 --rdzv-endpoint=localhost:39189 train_segmenter.py \
  --experiment_name "terrafm_seg" --backbone 'terrafm-base' --encoder_weights 'terrafm_base' \
  --loss_type ce --in_channels 18 --lr_sched 'warmup_cosine' --warmup_steps 20 --weight_decay 0.0005 \
  --lr 6e-4 --warmup_lr 0.000001 --dataset_name 'harvey' --dataset_path '/your/path/to/harvey' \
  --bands B2 B3 B4 --batch_size 8 --max_epochs 100 --img_size 224 --seed 42 --upernet_width 64
```

### 🔍 Change Detection

```bash
torchrun --nnodes=1 --nproc_per_node=1 --rdzv-endpoint=localhost:29189 train_change.py \
  --experiment_name "terrafm_change" --mode 'vanilla' --backbone 'terrafm-base' --encoder_weights 'terrafm_base' \
  --fusion 'diff' --lr_sched 'warmup_cosine' --warmup_steps 20 --weight_decay 0.0005 \
  --lr 5e-4 --warmup_lr 0.000001 --dataset_name 'harvey' --dataset_path '/your/path/to/harvey' \
  --bands B2 B3 B4 --batch_size 8 --max_epochs 100 --img_size 224 --seed 42 --upernet_width 64
```

## 📈 Evaluation

Below are the evaluation scripts for classification, semantic segmentation, and change detection tasks.

### 🏷️ Classification

```bash
python eval_bands_cls.py --model_config './configs/terrafm.json' --dataset_config './configs/m_ben.json' \
    --checkpoint_path "/your/path/to/checkpoint" --img_size  224
```

### 🖼️ Semantic Segmentation

```bash
python eval_bands_seg.py --model_config './configs/terrafm.json' --dataset_config './configs/harvey.json' \
  --checkpoint_path "/your/path/to/checkpoint" \
  --size 224 --bands '[["B2", "B3", "B4"], ["B5","B3","B4"], ["B6", "B5", "B4"], ["B8A", "B11", "B12"], ["vh", "vv"]]'
```

### 🔍 Change Detection

```bash
python eval_bands_cd.py --model_config './configs/terrafm.json' --dataset_config './configs/harvey.json' \
  --checkpoint_path "/your/path/to/checkpoint" \
  --size 224 --bands '[["B2", "B3", "B4"], ["B5","B3","B4"], ["B6", "B5", "B4"], ["B8A", "B11", "B12"], ["vh", "vv"]]'
```

