<h1 align="center">
  <b>Finetuning Models for Remote Sensing Tasks</b><br>
</h1>


### 🌱 How to use <a name="use"></a>

Evaluation of remote sensing foundation models on various tasks such as change detection, multi-label classification.



### 🔭 Models <a name="models"></a>


#### Architectures <a name="architectures"></a>

- [x] UPerNet [[paper](https://arxiv.org/abs/1807.10221)]


#### Encoders <a name="encoders"></a>

- [x]  iBot* [[paper](https://arxiv.org/abs/2111.07832)]
- [x]  SatlasPretrain [[paper](https://arxiv.org/abs/2211.15660)]
- [x]  GFM [[paper](https://arxiv.org/abs/1807.10221)]

* we train iBot on MillionAid dataset

### Change Detection <a name="cd"></a>

For full finetuning 
run

```bash
  torchrun --nnodes=1 --nproc_per_node=1 --rdzv-endpoint=localhost:29501 local_test.py --backbone 'ibot-B' --encoder_weights "million_aid" --experiment_name 'levir_ibot' --dataset_name 'Levir_CD' --dataset_path '/path/to/data' --batch_size 32 --max_epochs 200 --lr_sched 'warmup_cosine' --img_suffix '.png' --warmup_steps 10 --weight_decay 0.05 --sub_dir_1 'A' --sub_dir_2 'B' --annot_dir 'OUT'
```
for frozen encoder pass `--freeze_encoder`




Available backbone types `Swin-B` `ibot-B`
Available encoder_weights for `Swin-B` are 
- [x] [[geopile](https://github.com/mmendiet/GFM/tree/main)]
- [x] [[satlas (sentinel-2) satlas_rs(Aerial) Swin-v2-Base single image](https://github.com/allenai/satlaspretrain_models/)]



### Classification <a name="cl"></a>

Look for dataset splits [[here](https://github.com/google-research/google-research/blob/master/remote_sensing_representations/README.md)]


For full finetuning 
run

```bash
  python train_classifier.py --experiment_name "ibot_resisc" --dataset_name "resisc45" --root "/root/path/to/datasets" --base_dir "NWPU-RESISC45" --num_classes "45" --in_features "768" --backbone_name "ibot-B" --encoder_weights "million_aid_fa" --lr 1e-4
```

For linear probing
```bash
  python train_classifier.py --experiment_name "ibot_resisc_head" --dataset_name "resisc45" --root "/root/path/to/datasets" --base_dir "NWPU-RESISC45" --num_classes "45" --in_features "768" --backbone_name "ibot-B" --encoder_weights "million_aid_fa" --only_head
```

### Inference <a name="infer"></a>

Change detection model evaluation on scales

run 

```bash
  python eval_scale_cd.py --model_config './configs/ibot-B.json' --dataset_config './configs/levir.json' --checkpoint_path 'path/to/finetuned/model.pth'
```

Classification

```bash
  python eval_scale_cls.py --model_config './configs/ibot-B.json' --dataset_config './configs/ucm.json' --checkpoint_path 'path/to/finetuned/model.pth'
```

### Inference MultiBand <a name="infer"></a>

Change detection model evaluation on bands

run 

```bash
  python eval_bands_cd.py --model_config './configs/ibot-B.json' --dataset_config './configs/oscd.json' --checkpoint_path 'path/to/finetuned/model.pth'
```

Classification

```bash
  python eval_bands_cls.py --model_config './configs/ibot-B.json' --dataset_config './configs/ben.json' --checkpoint_path 'path/to/finetuned/model.pth'
```

For evaluating on SAR bands pass `--sar`