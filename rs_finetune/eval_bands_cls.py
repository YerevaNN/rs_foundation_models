import os
import torch
import rasterio
import json
import numpy as np
import train_classifier as tr_cls

from tqdm import tqdm
from argparse import ArgumentParser
from torchmetrics import AveragePrecision
from change_detection_pytorch.datasets import BigearthnetDataModule
from torchvision import transforms
from change_detection_pytorch.datasets.BEN import NEW_LABELS, GROUP_LABELS, normalize_stats


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
    if args.replace_rgb_with_others:
        cvit_channels = [0, 1]

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
    
    for k, s1_path in tqdm(test_samples.items()):
        data = os.listdir(os.path.join(root_path, s1_path))
        for d in data:
            suffix = d.split('_')[-1]
            if 'vv' in suffix.lower():
                vv = d
            elif 'vh' in suffix.lower():
                vh = d
            else:
                labels = d
        with open(f'/nfs/h100/raid/rs/metadata_ben_clay/{k}.json', 'r') as f:
            metadata = json.load(f)
            metadata.update({'waves': [3.5, 4.0, 0]})
            if args.replace_rgb_with_others:
                metadata.update({'waves': [0.665, 0.56, 0]})
                
        # labels, vv, vh = data
        channels = []
    
        vv_path = os.path.join(root_path, s1_path, vv )
        vv = rasterio.open(vv_path).read(1)
        vv = normalize_stats(vv, mean=SAR_STATS['mean']['VV'], std=SAR_STATS['std']['VV'])
        vv = transforms.functional.resize(torch.from_numpy(vv).unsqueeze(0), args.img_size, 
                            interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        channels.append(vv)
        if 'cvit' in cfg['backbone'].lower() and not args.replace_rgb_with_others:
            channels.append(vv)
        
        vh_path = os.path.join(root_path, s1_path, vh )

        vh = rasterio.open(vh_path).read(1)
        vh = normalize_stats(vh, mean=SAR_STATS['mean']['VH'], std=SAR_STATS['std']['VH'])
        vh = transforms.functional.resize(torch.from_numpy(vh).unsqueeze(0), args.img_size, 
                            interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        channels.append(vh)
        if 'cvit' in cfg['backbone'].lower() and not args.replace_rgb_with_others:
            channels.append(vh)
        if 'cvit' not in cfg['backbone'].lower():
            zero_channel = torch.zeros(args.img_size, args.img_size).unsqueeze(0)
            channels.append(zero_channel)
            
        if 'satlas' in cfg['encoder_weights'].lower():
            for i in range(6):
                zero_channel = torch.zeros(args.img_size, args.img_size).unsqueeze(0)
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
        elif 'clay' in cfg['backbone'].lower():
            logits = model(img, [metadata])
        else:
            logits = model(img)
        preds.append(logits.cpu().detach())
        
    accuracy = test_accuracy(torch.tensor(np.array(preds)), torch.tensor(np.array(gts))).to(device).detach()
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    results[args.checkpoint_path]['vvvh'] = accuracy * 100
            
    save_directory = f'./eval_outs/{args.checkpoint_path.split("/")[-2]}'

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    savefile = f'{save_directory}/results_sar.npy'
    np.save(savefile, results)

    print(results)

def main(args):
    bands = [['B04', 'B03', 'B02'], ['B04', 'B03', 'B05'], ['B04', 'B05', 'B06'], ['B8A', 'B11', 'B12']]

    if args.replace_rgb_with_others:
        bands = [['B04', 'B03', 'B02_B05'], ['B04', 'B03_B05', 'B02_B06'], ['B04_B8A', 'B03_B11', 'B02_B12']]

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
                if '_' in b:
                    first_band, second_band = b.split('_')
                    get_indicies.append(channel_vit_order.index(first_band))
                    band[band.index(b)] = second_band
                else:
                    get_indicies.append(channel_vit_order.index(b))

            print('band2: ', band)

            datamodule = BigearthnetDataModule(data_dir=data_cfg['base_dir'], batch_size=data_cfg['batch_size'],
                                    num_workers=24, img_size=args.img_size, replace_rgb_with_others=args.replace_rgb_with_others, 
                                    bands=band, splits_dir=data_cfg['splits_dir'], fill_zeros=cfg['fill_zeros'])
            datamodule.setup()
            test_dataloader = datamodule.test_dataloader()

            with torch.no_grad():
                correct_predictions = 0
                total_samples = 0
                for batch in tqdm(test_dataloader):
                    if 'ben' in data_cfg['dataset_name']:
                        x, y, metadata = batch
                    else:
                        x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    if 'cvit' in cfg['backbone'].lower():
                        logits = model(x, channels = get_indicies)
                    elif 'clay' in cfg['backbone'].lower():
                        logits = model(x, metadata)
                    else:
                        logits = model(x)
                    batch_accuracy = test_accuracy(logits, y.int()).to(device)
                    correct_predictions += batch_accuracy.item() * len(y)
                    total_samples += len(y)
            
                overall_test_accuracy = correct_predictions / total_samples
            print(args.checkpoint_path)
            print(f'Test Accuracy: {overall_test_accuracy * 100:.2f}%')
            results[args.checkpoint_path][''.join(band)] = overall_test_accuracy * 100
            
        save_directory = f'./eval_outs/{args.checkpoint_path.split("/")[-2]}'

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        savefile = f'{save_directory}/results.npy'
        np.save(savefile, results)

        print(results)

if __name__ == '__main__':

    # bands = [['B04', 'B03', 'B02'], ['B04', 'B03', 'B05'], ['B04', 'B05', 'B06'], ['B8A', 'B11', 'B12']]

    channel_vit_order = ['B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A',  'B11', 'B12'] #VVr VVi VHr VHi
    all_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A','B11', 'B12','vv', 'vh']

    parser = ArgumentParser()
    parser.add_argument('--model_config', type=str, default='')
    parser.add_argument('--dataset_config', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--sar', action="store_true")
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--replace_rgb_with_others', action="store_true")
    args = parser.parse_args()

    main(args)
