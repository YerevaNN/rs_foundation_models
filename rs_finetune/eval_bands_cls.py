import torch

from argparse import ArgumentParser
from torchmetrics import AveragePrecision
import rasterio
import os
from change_detection_pytorch.datasets import BigearthnetDataModule
from torchvision import transforms

from tqdm import tqdm
import numpy as np

from change_detection_pytorch.datasets.BEN import NEW_LABELS, GROUP_LABELS, normalize_stats

import train_classifier as tr_cls
import json

SAR_STATS = {
    'mean': {'VH': -19.29836, 'VV': -12.623948},
    'std': {'VH': 5.4643545, 'VV':  5.1194134 }
}
def get_multihot_new(labels):

    target = np.zeros((len(NEW_LABELS),), dtype=np.float32)
    for label in labels:
        if label in GROUP_LABELS:
            target[NEW_LABELS.index(GROUP_LABELS[label])] = 1
        elif label not in set(NEW_LABELS):
            continue
        else:
            target[NEW_LABELS.index(label)] = 1
    return target
def eval_sar(args):

    cvit_channels = [10,11,12,13]
    results = {}
    test_samples = np.load('/nfs/ap/mnt/frtn/rs-multiband/BigEarthNet/s2_s1_mapping_test.npy', allow_pickle=True).item()
    root_path = '/nfs/ap/mnt/frtn/rs-multiband/'
    results[args.checkpoint_path] = {}
    with open(args.model_config) as config:
        cfg = json.load(config)
    
    with open(args.dataset_config) as config:
        data_cfg = json.load(config)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    prefix='encoder' 
    model = tr_cls.Classifier(backbone_name=cfg['backbone'], backbone_weights=cfg['encoder_weights'], 
                                  in_features=cfg['in_features'], num_classes=data_cfg['num_classes'],
                              lr=0.0, sched='', checkpoint_path=args.checkpoint_path, only_head='',
                            warmup_steps = '', eta_min = '', warmup_start_lr='', weight_decay= '', 
                            prefix=prefix, mixup=False)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    model = model.to(device)
    
    test_accuracy = AveragePrecision(num_classes=data_cfg['num_classes'], average='micro', task='binary')
    preds = []
    gts = []
    
    for _, s1_path in tqdm(test_samples.items()):
        data = os.listdir(os.path.join(root_path, s1_path))
        for d in data:
            suffix = d.split('_')[-1]
            if 'vv' in suffix.lower():
                vv = d
            elif 'vh' in suffix.lower():
                vh = d
            else:
                labels = d
                
        # labels, vv, vh = data
        channels = []
    
        vv_path = os.path.join(root_path, s1_path, vv )
        vv = rasterio.open(vv_path).read(1)
        vv = normalize_stats(vv, mean=SAR_STATS['mean']['VV'], std=SAR_STATS['std']['VV'])
        vv = transforms.functional.resize(torch.from_numpy(vv).unsqueeze(0), 128, 
                            interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        channels.append(vv)
        if 'cvit' in cfg['backbone'].lower():
            channels.append(vv)
        
        vh_path = os.path.join(root_path, s1_path, vh )

        vh = rasterio.open(vh_path).read(1)
        vh = normalize_stats(vh, mean=SAR_STATS['mean']['VH'], std=SAR_STATS['std']['VH'])
        vh = transforms.functional.resize(torch.from_numpy(vh).unsqueeze(0), 128, 
                            interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        channels.append(vh)
        if 'cvit' in cfg['backbone'].lower():
            channels.append(vh)
        if 'cvit' not in cfg['backbone'].lower():
            zero_channel = torch.zeros(128, 128).unsqueeze(0)
            channels.append(zero_channel)
            
        if 'satlas' in cfg['encoder_weights'].lower():
            for i in range(6):
                zero_channel = torch.zeros(128, 128).unsqueeze(0)
                channels.append(zero_channel)
            
        img = torch.cat(channels, dim=0)
        img = img.float().div(255)
        img = img.unsqueeze(0).to(device)
    
        labels_path = os.path.join(root_path, s1_path, labels )
        
        with open(labels_path, 'r') as f:
            labels = json.load(f)['labels']
        target = get_multihot_new(labels)
        target = torch.from_numpy(target)
        target = target.unsqueeze(0)
        gts.append(target.int())
        if 'cvit' in cfg['backbone'].lower():
            logits = model(img, channels = cvit_channels)
        else:
            logits = model(img)
        preds.append(logits.cpu().detach())
        
    accuracy = test_accuracy(torch.tensor(np.array(preds)), torch.tensor(np.array(gts))).to(device).detach()
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    results[args.checkpoint_path]['vvvh'] = accuracy * 100
            
    # save_directory = f'./eval_outs/{args.checkpoint_path.split('/')[-2]}'
    checkpoint_split = args.checkpoint_path.split('/')
    checkpoint_part = checkpoint_split[-2]
    save_directory = f'./eval_outs/{checkpoint_part}'

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    savefile = f'{save_directory}/results_sar.npy'
    np.save(savefile, results)

    print(results)

def main(args):
    if args.sar:
        eval_sar(args)
    else:
        results = {}
        results[args.checkpoint_path] = {}

        with open(args.model_config) as config:
            cfg = json.load(config)
        
        with open(args.dataset_config) as config:
            data_cfg = json.load(config)

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(args.checkpoint_path, cfg)
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        prefix='encoder'
        model = tr_cls.Classifier(backbone_name=cfg['backbone'], backbone_weights=cfg['encoder_weights'], 
                                    in_features=cfg['in_features'], num_classes=data_cfg['num_classes'],
                                lr=0.0, sched='', checkpoint_path=args.checkpoint_path, only_head='',
                                warmup_steps = '', eta_min = '', warmup_start_lr='', weight_decay= '', 
                                prefix=prefix, mixup=False)
        model.load_state_dict(checkpoint['state_dict'])
        
        model.eval()
        model = model.to(device)
        
        test_accuracy = AveragePrecision(num_classes=data_cfg['num_classes'], average='micro', task='binary')

        results[args.checkpoint_path] = {}
        for band in bands :
            get_indicies = []

            print('band1: ', band)

            for b in band:
                if b == 'B04_B05':
                    get_indicies.append(channel_vit_order.index('B04'))
                    band = ['B05', 'B03', 'B02']
                else:
                    get_indicies.append(channel_vit_order.index(b))

            print('band2: ', band)

            datamodule = BigearthnetDataModule(data_dir=data_cfg['base_dir'], batch_size=data_cfg['batch_size'],
                                                num_workers=24,
                                            bands=band, splits_dir=data_cfg['splits_dir'], fill_zeros=cfg['fill_zeros'])
            datamodule.setup()
            test_dataloader = datamodule.test_dataloader()

            with torch.no_grad():
                correct_predictions = 0
                total_samples = 0
                for batch in tqdm(test_dataloader):
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    if 'cvit' in cfg['backbone'].lower():
                        logits = model(x, channels = get_indicies)
                    else:
                        logits = model(x)
                    batch_accuracy = test_accuracy(logits, y.int()).to(device)
                    correct_predictions += batch_accuracy.item() * len(y)
                    total_samples += len(y)
            
                overall_test_accuracy = correct_predictions / total_samples
            print(args.checkpoint_path)
            print(f'Test Accuracy: {overall_test_accuracy * 100:.2f}%')
            results[args.checkpoint_path][''.join(band)] = overall_test_accuracy * 100
            
        # save_directory = f'./eval_outs/{args.checkpoint_path.split('/')[-2]}'
        checkpoint_split = args.checkpoint_path.split('/')
        checkpoint_part = checkpoint_split[-2]
        save_directory = f'./eval_outs/{checkpoint_part}'

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        savefile = f'{save_directory}/results.npy'
        np.save(savefile, results)

        print(results)

if __name__ == '__main__':

    # bands = [['B04', 'B03', 'B02'], ['B05', 'B03', 'B02'], ['B06', 'B05', 'B02'], ['B8A', 'B11', 'B12']]
    bands = [['B04', 'B03', 'B02'], ['B04_B05', 'B03', 'B02'], ['B05', 'B03', 'B02'], ['B06', 'B05', 'B02'], ['B8A', 'B11', 'B12']]

    channel_vit_order = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A',  'B11', 'B12'] #VVr VVi VHr VHi
    all_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A','B11', 'B12','vv', 'vh']

    parser = ArgumentParser()
    parser.add_argument('--model_config', type=str, default='')
    parser.add_argument('--dataset_config', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--sar', action="store_true")

    args = parser.parse_args()

    main(args)
