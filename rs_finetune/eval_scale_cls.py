import os
import json
import torch
import numpy as np
import train_classifier as tr_cls

from tqdm import tqdm
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from change_detection_pytorch.datasets import UCMerced, build_transform


def main(args):
    results = {}

    with open(args.model_config) as config:
        cfg = json.load(config)
    
    with open(args.dataset_config) as config:
        data_cfg = json.load(config)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    if 'satlas' in cfg['encoder_weights']:
        model = tr_cls.Classifier(backbone_name=cfg['backbone'], backbone_weights=cfg['encoder_weights'], 
                                  in_features=cfg['in_features'], num_classes=data_cfg['num_classes'], 
                                  lr=0.0, sched='', checkpoint_path=args.checkpoint_path, only_head='',
                                warmup_steps = '', eta_min = '', warmup_start_lr='', weight_decay= '',
                                  prefix='encoder', mixup=False)
    else:
        model = tr_cls.Classifier(backbone_name=cfg['backbone'], backbone_weights=cfg['encoder_weights'], 
                                  in_features=cfg['in_features'], num_classes=data_cfg['num_classes'], 
                                  lr=0.0, sched='', checkpoint_path='', only_head='',
                                     warmup_steps = '', eta_min = '', warmup_start_lr='', weight_decay= '', mixup=False)

        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    
    model.eval()
    model = model.to(device)
    image_size = 252 if 'dino' in cfg['backbone'] else data_cfg['image_size'] 

    results[args.checkpoint_path] = {}
    for scale in args.scales:
        base_dir = data_cfg['base_dir']

        if scale != '1x':
            base_dir = base_dir[:-1] + '_' + scale + base_dir[-1]
            
        test_transform = build_transform(split='test', image_size = image_size)
        test_dataset = UCMerced(root=data_cfg['root'], base_dir=base_dir, split='test', 
                                transform=test_transform, dataset_name=data_cfg['dataset_name'], image_size=data_cfg['image_size'])
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=data_cfg['batch_size'], shuffle=True, num_workers=args.num_workers)

        test_accuracy = Accuracy(task="multiclass", num_classes=data_cfg['num_classes']).to(device)
        with torch.no_grad():
            correct_predictions = 0
            total_samples = 0
            for batch in tqdm(test_dataloader):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                batch_accuracy = test_accuracy(torch.argmax(logits, dim=1), y)
                correct_predictions += batch_accuracy.item() * len(y)
                total_samples += len(y)
        
            overall_test_accuracy = correct_predictions / total_samples

        print(args.checkpoint_path, scale, f'Test Accuracy: {overall_test_accuracy * 100:.2f}%')
        results[args.checkpoint_path][scale] = overall_test_accuracy * 100

    save_directory = f'./eval_outs/{args.eval_epoch}/{args.checkpoint_path.split('/')[-2]}'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    savefile = f'{save_directory}/results.npy'
    np.save(savefile, results)

    print(results)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model_config', type=str, default='')
    parser.add_argument('--dataset_config', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--eval_epoch', type=str, default='last')
    parser.add_argument("--scales", nargs="+", type=str, default=['1x'])

    args = parser.parse_args()

    main(args)
