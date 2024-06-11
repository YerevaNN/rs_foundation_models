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

run `python local_test.py --backbone Swin-B --encoder_weights geopile  --dataset_name Levir-CD --fusion diff`

Available backbone types `Swin-B` `ibot-B`
Available encoder_weights for `Swin-B` are 
- [x] [[geopile](https://github.com/mmendiet/GFM/tree/main)]
- [x] [[satlas (sentinel-2) satlas_rs(Aerial) Swin-v2-Base single image](https://github.com/allenai/satlaspretrain_models/)]



### Classification <a name="cl"></a>

Look for dataset splits [[here](https://github.com/google-research/google-research/blob/master/remote_sensing_representations/README.md)]

run `python train_classifier.py --dataset_name uc_merced --num_classes 21 --in_features 1024 --backbone_name "Swin-B" --encoder_weights satlas`

### Inference <a name="infer"></a>

Change detection model evaluation `inference-change.ipynb`

Classification model evaluation `inference-classifier.ipynb`
