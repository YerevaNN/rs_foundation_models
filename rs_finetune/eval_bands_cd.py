from eval_scale_cd import CustomMetric, load_model, init_dist

import os
import torch
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from itertools import product
import json

from PIL import Image
from argparse import ArgumentParser

from sklearn import metrics
from glob import glob

import change_detection_pytorch as cdp
import torch.distributed as dist

import rasterio
from tqdm import tqdm
from change_detection_pytorch.datasets import ChangeDetectionDataModule, normalize_channel, RGB_BANDS, STATS

SAR_STATS = {
    'mean': {'VV': -9.152486082800158, 'VH': -16.23374164784503},
    'std': {'VV': 5.41078882186851, 'VH': 5.419913471274721}
} 

def get_image_array(path, return_rgb=False):
    channels = []
  
    if return_rgb:
        root = path.split('/')[:-2]
        root = os.path.join(*root)
        root = '/' + root
        band_files = os.listdir(root)
        for band_file in band_files:
            for b in RGB_BANDS:
                if b in band_file:
                    ch = rasterio.open(os.path.join(root, band_file)).read(1)
                    ch = normalize_channel(ch, mean=STATS['mean'][b], std=STATS['std'][b])
                    channels.append(ch)

    else:
        img = gdal.Open(path, gdal.GA_ReadOnly).ReadAsArray()
        
        vv_intensity = img[0]
        vh_intensity = img[1]
            
        vv = normalize_channel(vv_intensity, mean=SAR_STATS['mean']['VV'], std=SAR_STATS['std']['VV'])
        vh = normalize_channel(vh_intensity, mean=SAR_STATS['mean']['VH'], std=SAR_STATS['std']['VH'])

        channels.append(vv)    
        channels.append(vh)
        
    img = np.dstack(channels)
    img = Image.fromarray(img)
        
    return img

def eval_on_sar(args):
    test_cities = '/nfs/ap/mnt/sxtn/aerial/change/OSCD/test.txt'
    with open(test_cities) as f:
        test_set = f.readline()
    test_set = test_set[:-1].split(',')
    save_directory = f'./eval_outs/{args.checkpoint_path.split("/")[-2]}'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with open(args.model_config) as config:
        cfg = json.load(config)
    
    channels = [10,11,12,13] if 'cvit' in cfg['backbone'].lower() else [0, 1, 2]
    model = load_model(args.checkpoint_path, encoder_depth=cfg['encoder_depth'], backbone=cfg['backbone'], 
                       encoder_weights=cfg['encoder_weights'], fusion=cfg['fusion'], 
                       load_decoder=cfg['load_decoder'], channels=channels, in_channels=cfg['in_channels'])
    model.eval()
    fscore = cdp.utils.metrics.Fscore(activation='argmax2d')


    samples = 0
    fscores = 0
    for place in tqdm(glob("/nfs/ap/mnt/frtn/rs-multiband/oscd/multisensor_fusion_CD/S1/*")):
        city_name = place.split('/')[-1]
        if city_name in test_set:
            path1 = glob(f"{place}/imgs_1/transformed/*")[0]
            img1 = get_image_array(path1, return_rgb=True)
    
            path2 = glob(f"{place}/imgs_2/transformed/*")[0]
            img2 = get_image_array(path2)
    
            cm_path = os.path.join('/nfs/ap/mnt/sxtn/aerial/change/OSCD/', city_name, 'cm/cm.png')    
            cm = Image.open(cm_path).convert('L')

            limits = product(range(0, img1.width, args.size), range(0, img1.height, args.size))
            for l in limits:
                limit = (l[0], l[1], l[0] + args.size, l[1] + args.size)
                sample1 = np.array(img1.crop(limit))
                sample2 = np.array(img2.crop(limit))
                mask = np.array(cm.crop(limit)) / 255

                if 'cvit' not in cfg['backbone'].lower() and 'prithvi' not in cfg['backbone'].lower():
                    zero_image = np.zeros((192, 192, 3))
                    zero_image[:,:, 0] = sample1[:,:, 0]
                    zero_image[:,:, 1] = sample1[:,:, 1]
                    sample1 = zero_image
                    
                    zero_image = np.zeros((192, 192, 3))
                    zero_image[:,:, 0] = sample2[:,:, 0]
                    zero_image[:,:, 1] = sample2[:,:, 1]
                    sample2 = zero_image
                    
                    
                if 'satlas' in cfg['encoder_weights'].lower():
                    zero_image = np.zeros((192, 192, 9))
                    zero_image[:,:, 0] = sample1[:,:, 0]
                    zero_image[:,:, 1] = sample1[:,:, 1]
                    sample1 = zero_image
                    
                    zero_image = np.zeros((192, 192, 9))
                    zero_image[:,:, 0] = sample2[:,:, 0]
                    zero_image[:,:, 1] = sample2[:,:, 1]
                    sample2 = zero_image
    
                if 'prithvi' in cfg['backbone'].lower():
                    zero_image = np.zeros((224, 224, 6))
                    zero_image[:,:, 0] = sample1[:,:, 0]
                    zero_image[:,:, 1] = sample1[:,:, 1]
                    sample1 = zero_image
                    
                    zero_image = np.zeros((224, 224, 6))
                    zero_image[:,:, 0] = sample2[:,:, 0]
                    zero_image[:,:, 1] = sample2[:,:, 1]
                    sample2 = zero_image

                if 'cvit' in cfg['backbone'].lower():
                    zero_image = np.zeros((192, 192, 4))
                    zero_image[:,:, 0] = sample1[:,:, 0]
                    zero_image[:,:, 1] = sample1[:,:, 0]
                    zero_image[:,:, 2] = sample1[:,:, 1]
                    zero_image[:,:, 3] = sample1[:,:, 1]
                    sample1 = zero_image
    
                    zero_image = np.zeros((192, 192, 4))
                    zero_image[:,:, 0] = sample2[:,:, 0]
                    zero_image[:,:, 1] = sample2[:,:, 0]
                    zero_image[:,:, 2] = sample2[:,:, 1]
                    zero_image[:,:, 3] = sample2[:,:, 1]
                    sample2 = zero_image
                name = city_name + '-' + '-'.join(map(str, limit))
                savefile = f'{save_directory}/{name}_sample1.npy'
                np.save(savefile, sample1)
                savefile = f'{save_directory}/{name}_sample2.npy'
                np.save(savefile, sample2)
                savefile = f'{save_directory}/{name}_mask.npy'
                np.save(savefile, mask)


                sample1 = torch.tensor(sample1.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0).cuda()
                sample2 = torch.tensor(sample2.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0).cuda()
                mask = torch.tensor(mask.astype(np.float32)).unsqueeze(0).cuda()
                with torch.no_grad():
                    out = model(sample1, sample2)
                savefile = f'{save_directory}/{name}_out.npy'
                np.save(savefile, out.detach().cpu())
                samples += 1
                fs = fscore(out, mask)
                fscores += fs
               
    print(samples)
    results = {}
    results[args.checkpoint_path] = {}
    results[args.checkpoint_path]['VVVH'] = {
                'micro_f1': fscores/samples}
        
 
    savefile = f'{save_directory}/results_sar.npy'
    np.save(savefile, results)

    print(args.checkpoint_path, (fscores/samples)*100)

def main(args):
    init_dist(args.master_port)
    # model = load_model(args.checkpoint_path, encoder_depth=cfg['encoder_depth'], backbone=cfg['backbone'], encoder_weights=cfg['encoder_weights'],
    #                fusion=cfg['fusion'], load_decoder=cfg['load_decoder'])
    
    # with open(args.dataset_config) as config:
    #     data_cfg = json.load(config)

    if args.sar:
        eval_on_sar(args)
    else:
        results = {}
        with open(args.model_config) as config:
            cfg = json.load(config)
        
        with open(args.dataset_config) as config:
            data_cfg = json.load(config)

        model = load_model(args.checkpoint_path, encoder_depth=cfg['encoder_depth'], backbone=cfg['backbone'], 
                       encoder_weights=cfg['encoder_weights'], fusion=cfg['fusion'], 
                       load_decoder=cfg['load_decoder'], in_channels=cfg['in_channels'])
        
        dataset_path = data_cfg['dataset_path']
        metadata_dir = data_cfg['metadata_dir']
        # tile_size = data_cfg['tile_size']
        batch_size = data_cfg['batch_size']
        fill_zeros = cfg['fill_zeros']

        tile_size = args.size

        loss = cdp.utils.losses.CrossEntropyLoss()
        DEVICE = 'cuda:{}'.format(dist.get_rank()) if torch.cuda.is_available() else 'cpu'
        results[args.checkpoint_path] = {}

        for band in bands :            
            custom_metric =  CustomMetric(activation='argmax2d', tile_size=tile_size)
            our_metrics = [
                cdp.utils.metrics.Fscore(activation='argmax2d'),
                cdp.utils.metrics.Precision(activation='argmax2d'),
                cdp.utils.metrics.Recall(activation='argmax2d'),
                custom_metric
            ]

            if 'cvit' in model.module.encoder_name.lower():
                print('band1: ', band)
                get_indicies = []
                for b in band:
                    if '_' in b:
                        first_band, second_band = b.split('_')
                        get_indicies.append(channel_vit_order.index(first_band))
                        band[band.index(b)] = second_band
                    else:
                        get_indicies.append(channel_vit_order.index(b))

                print('band2: ', band)

                model.module.channels = get_indicies
            
            datamodule = ChangeDetectionDataModule(dataset_path, metadata_dir, patch_size=tile_size, bands=band, 
                                                   fill_zeros=fill_zeros, batch_size=batch_size)
            datamodule.setup()
                
            valid_loader = datamodule.val_dataloader()

            valid_epoch = cdp.utils.train.ValidEpoch(
                model,
                loss=loss,
                metrics=our_metrics,
                device=DEVICE,
                verbose=True,
            )
            
            valid_logs = valid_epoch.run(valid_loader)

            data = custom_metric.data

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
        
            results[args.checkpoint_path][''.join(band)] = {
                'micro_f1': micro_f1,
                'macro_f1': macro_f1
            }
        
        save_directory = f'./eval_outs/{args.checkpoint_path.split("/")[-2]}'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        savefile = f'{save_directory}/results.npy'
        np.save(savefile, results)

        for b in bands:
            print(f"{b} micro-F1 = {results[args.checkpoint_path][''.join(b)]['micro_f1']:.3f}")
            
if __name__== '__main__':

    bands = [['B04', 'B03', 'B02'], ['B04', 'B03', 'B05'], ['B04', 'B05', 'B06'], ['B8A', 'B11', 'B12']]
            #  ['B04', 'B03', 'B02_B05'], ['B04', 'B03_B05', 'B02_B06'], ['B04_B8A', 'B03_B11', 'B02_B12']]
    
    channel_vit_order = ['B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A',  'B11', 'B12'] #VVr VVi VHr VHi
    all_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A','B11', 'B12','vv', 'vh']

    parser = ArgumentParser()
    parser.add_argument('--model_config', type=str, default='')
    parser.add_argument('--dataset_config', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--sar', action="store_true")
    parser.add_argument('--size', type=int, default=192)
    parser.add_argument('--master_port', type=str, default="12345")

    args = parser.parse_args()

    main(args)
