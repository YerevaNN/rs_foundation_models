import os
import json 
import torch
import numpy as np
import torch.distributed as dist
import change_detection_pytorch as cdp

from argparse import ArgumentParser
from sklearn import metrics
from tqdm import tqdm
from change_detection_pytorch.base.modules import Activation
from change_detection_pytorch.utils import base
from change_detection_pytorch.utils import functional as F
from change_detection_pytorch.datasets import LEVIR_CD_Dataset, ChangeDetectionDataModule
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


def init_dist(master_port):
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = master_port

    dist.init_process_group(backend='nccl', init_method='env://')

def load_model(checkpoint_path='',encoder_depth=12, backbone='Swin-B', encoder_weights='geopile', upernet_width=256,
                fusion='diff', load_decoder=False, in_channels = 3, channels=[0, 1, 2], upsampling=4, out_size=224):
    model = cdp.UPerNet(
        encoder_depth = encoder_depth,
        encoder_name = backbone, # choose encoder, e.g. 'ibot-B', 
        encoder_weights = encoder_weights, # pre-trained weights for encoder initialization `imagenet`, `million_aid`
        in_channels = in_channels, # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes = 2, # model output channels (number of classes in your datasets)
        siam_encoder = True, # whether to use a siamese encoder
        fusion_form = fusion, # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
        pretrained = load_decoder,
        channels=channels,
        upsampling=upsampling,
        out_size=out_size,
        decoder_psp_channels=upernet_width * 2,
        decoder_pyramid_channels=upernet_width,
        decoder_segmentation_channels=upernet_width,

    )
    model.to('cuda:{}'.format(dist.get_rank()))
    model = DDP(model)
    
    model = torch.load(checkpoint_path, map_location='cuda')
    
    # model.load_state_dict(checkpoint.state_dict())
    model.load_state_dict(checkpoint)

    return model

def f1_bitwise(y_true, y_pred):
    TP = np.bitwise_and(y_true, y_pred).sum()
    FP = np.bitwise_and(y_pred, np.logical_not(y_true)).sum()
    FN = np.bitwise_and(np.logical_not(y_pred), y_true).sum()

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return F1

class CustomMetric(base.Metric):
    __name__ = 'custom'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, tile_size=192, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.data = {
                'p': np.empty((0, tile_size, tile_size), dtype='uint8'),
                't': np.empty((0, tile_size, tile_size), dtype='uint8'),
                'f': []
            }

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        self.data['p'] = np.concatenate([self.data['p'], y_pr.cpu().numpy().astype('uint8')])
        self.data['t'] = np.concatenate([self.data['t'], y_gt.cpu().numpy().astype('uint8')])
        
        fscores = torch.tensor([F.f_score(p, g) for p, g in zip(y_pr, y_gt)])
        return fscores.mean()


def main(args):
    results = {}

    with open(args.model_config) as config:
        cfg = json.load(config)
    
    init_dist(args.master_port)
    model = load_model(args.checkpoint_path, encoder_depth=cfg['encoder_depth'], backbone=cfg['backbone'], encoder_weights=cfg['encoder_weights'],
                   fusion=cfg['fusion'], load_decoder=cfg['load_decoder'], upsampling=args.upsampling)
    
    with open(args.dataset_config) as config:
        data_cfg = json.load(config)
    

    dataset_name = data_cfg['dataset_name']
    dataset_path = data_cfg['dataset_path']
    # tile_size = data_cfg['tile_size']
    sub_dir_1 = data_cfg['sub_dir_1']
    sub_dir_2 = data_cfg['sub_dir_2']
    ann_dir = data_cfg['ann_dir']
    img_suffix = data_cfg['img_suffix']
    batch_size = data_cfg['batch_size']

    tile_size = args.tile_size

    loss = cdp.utils.losses.CrossEntropyLoss()
    if args.use_dice_bce_loss:
        loss = cdp.utils.losses.dice_bce_loss()
    custom_metric =  CustomMetric(activation='argmax2d', tile_size=tile_size)
    our_metrics = [
        cdp.utils.metrics.Fscore(activation='argmax2d'),
        cdp.utils.metrics.Precision(activation='argmax2d'),
        cdp.utils.metrics.Recall(activation='argmax2d'),
        custom_metric
    ]

    DEVICE = 'cuda:{}'.format(dist.get_rank()) if torch.cuda.is_available() else 'cpu'
    results[args.checkpoint_path] = {}


    for scale in args.scales:
        custom_metric =  CustomMetric(activation='argmax2d', tile_size=tile_size)
        our_metrics = [
            cdp.utils.metrics.Fscore(activation='argmax2d'),
            cdp.utils.metrics.Precision(activation='argmax2d'),
            cdp.utils.metrics.Recall(activation='argmax2d'),
            custom_metric
        ]

        if 'oscd' in dataset_name.lower():
            if scale != '1x':
                mode = 'vanilla' if scale == '1x' else 'wo_train_aug' 
                
            datamodule = ChangeDetectionDataModule(dataset_path, patch_size=tile_size, mode=mode, scale=scale, batch_size=batch_size)
            datamodule.setup()
                
            valid_loader = datamodule.val_dataloader()
                
        else:
            valid_dataset = LEVIR_CD_Dataset(f'{dataset_path}/test',
                                                sub_dir_1=sub_dir_1,
                                                sub_dir_2=sub_dir_2 if scale == '1x' else f'{sub_dir_2}_{scale}',
                                                img_suffix=img_suffix,
                                                ann_dir=f'{dataset_path}/test/{ann_dir}',
                                                debug=False,
                                                seg_map_suffix=img_suffix,
                                                size=args.crop_size,
                                                test_mode=True)
            
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        valid_epoch = cdp.utils.train.ValidEpoch(
                model,
                loss=loss,
                metrics=our_metrics,
                device=DEVICE,
                verbose=True,
            )
            
        valid_logs = valid_epoch.run(valid_loader)

        data = custom_metric.data

        if 'oscd' in dataset_name.lower():
            data['f'] = [y for x in valid_logs['filenames'] for y in x]
            cities = []
            coords = []
            for name in data['f']:
                name = name.split('/')[-1]
                _parts = name.split('_')
                city = '_'.join(_parts[:-1])
                coord = [int(t) for t in _parts[-1][1:-1].split(', ')]
                cities.append(city)
                coords.append(coord)
            unique_cities = set(cities)
            maps = {city: {
                't': np.zeros((1000, 1000)),
                'p': np.zeros((1000, 1000)),
            } for city in unique_cities}
            for city, coord, p, t in zip(cities, coords, data['p'], data['t']):
                x1,y1,x2,y2 = coord
                maps[city]['t'][y1:y2,x1:x2] = t
                maps[city]['p'][y1:y2,x1:x2] = p
            for city in tqdm(maps.keys()):
                maps[city]['fscore'] = metrics.f1_score(maps[city]['t'].flatten(), maps[city]['p'].flatten())
            micro_f1 = metrics.f1_score(
                np.concatenate([maps[city]['t'].flatten() for city in maps]),
                np.concatenate([maps[city]['p'].flatten() for city in maps]), 
            )
            macro_f1 = np.mean([maps[city]['fscore'] for city in maps]) 

        else:
            fscores = []
            maps_t = []
            maps_p = []
            for p, t in tqdm(zip(data['p'], data['t'])):
                if p.sum() + t.sum() == 0:
                    fscores.append(0)
                else:
                    f1_real = metrics.f1_score(t.flatten(), p.flatten())
                    # f1_ours = f1_bitwise(t.flatten(), p.flatten())
                    fscores.append(f1_real)
                maps_t.append(t)
                maps_p.append(p)
                    
            macro_f1 = np.mean(fscores)
            maps_t = np.vstack(maps_t)
            maps_p = np.vstack(maps_p)
        
            micro_f1 = f1_bitwise(maps_t, maps_p)
            maps = {'t':maps_t, 'p':maps_p}
                
            print(scale, micro_f1)
        
            results[args.checkpoint_path][scale] = {
                'maps': maps,
                'micro_f1': micro_f1,
                'macro_f1': macro_f1
            }
    
    save_directory = f'./eval_outs/{args.checkpoint_path.split("/")[-2]}'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    savefile = f'{save_directory}/results.npy'
    np.save(savefile, results)

    for scale in scales:
        print(f"{scale} micro-F1 = {results[args.checkpoint_path][scale]['micro_f1']:.3f}")


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model_config', type=str, default='')
    parser.add_argument('--dataset_config', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--master_port', type=str, default="12345")
    parser.add_argument('--upsampling', type=float, default=4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--tile_size', type=int, default=256)
    parser.add_argument('--upsampling', type=float, default=4)
    parser.add_argument('--use_dice_bce_loss', action="store_true")
    parser.add_argument("--scales", nargs="+", type=str, default=['1x'])

    args = parser.parse_args()

    main(args)
