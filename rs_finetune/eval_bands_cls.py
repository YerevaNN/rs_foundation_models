import os
import torch
import rasterio
import json
import numpy as np
import train_classifier as tr_cls

from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchmetrics import AveragePrecision, Accuracy, F1Score
from change_detection_pytorch.datasets import BigearthnetDataModule, mEurosat, So2SatDataset, mBigearthnet, BrickKiln
from torchvision import transforms
from utils import get_band_indices, get_band_orders, create_collate_fn
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
    results = {}
    test_samples = np.load('/nfs/ap/mnt/frtn/rs-multiband/BigEarthNet/s2_s1_mapping_test.npy', allow_pickle=True).item()
    root_path = '/nfs/ap/mnt/frtn/rs-multiband/'
    results[args.checkpoint_path] = {}
    with open(args.model_config) as config:
        cfg = json.load(config)
    
    with open(args.dataset_config) as config:
        data_cfg = json.load(config)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if 'cvit' in cfg['backbone'].lower():
        cvit_channels = [10,11,12,13]
        bands = ['VH', 'VH', 'VV', 'VV']
    elif 'cvit-pretrained' in cfg['backbone'].lower():
        cvit_channels = [10,11]
        bands = ['VV', 'VH']
    elif 'anysat' in cfg['backbone'].lower():
        bands = ['VV', 'VH', '']
    else:
        bands = ['VH', 'VV']

    # if args.replace_rgb_with_others:
    #     cvit_channels = [0, 1]
    #     bands = ['VV', 'VH']


    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    prefix='encoder' 
    model = tr_cls.Classifier(backbone_name=cfg['backbone'], backbone_weights=cfg['encoder_weights'], 
                              in_features=cfg['in_features'], num_classes=data_cfg['num_classes'],
                              lr=0.0, scheduler='', checkpoint_path=args.checkpoint_path, only_head='',
                              warmup_steps = '', eta_min = '', warmup_start_lr='', weight_decay= '', 
                              prefix=prefix, mixup=False, bands=bands, 
                              enable_multiband_input=args.enable_multiband_input,
                              multiband_channel_count=args.multiband_channel_count,
                              shared_proj=args.shared_proj, add_ch_embed=args.add_ch_embed)
    model.load_state_dict(checkpoint['state_dict'])


    if args.preserve_rgb_weights:
        from classifier_utils import adapt_encoder_for_multiband_eval
        
        adapt_encoder_for_multiband_eval(
            encoder=model.encoder, 
            multiband_channel_count=args.multiband_channel_count
        )
        
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
                metadata.update({'waves': [0.493, 0.56, 0]})
                
        # labels, vv, vh = data
        channels = []
    
        vv_path = os.path.join(root_path, s1_path, vv )
        vv = rasterio.open(vv_path).read(1)
        vv = normalize_stats(vv, mean=SAR_STATS['mean']['VV'], std=SAR_STATS['std']['VV'])
        vv = transforms.functional.resize(torch.from_numpy(vv).unsqueeze(0), data_cfg['image_size'], 
                            interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        # channels.append(vv)
        # if 'cvit' in cfg['backbone'].lower() and not args.replace_rgb_with_others:
        #     channels.append(vv)
        
        vh_path = os.path.join(root_path, s1_path, vh )

        vh = rasterio.open(vh_path).read(1)
        vh = normalize_stats(vh, mean=SAR_STATS['mean']['VH'], std=SAR_STATS['std']['VH'])
        vh = transforms.functional.resize(torch.from_numpy(vh).unsqueeze(0), data_cfg['image_size'] , 
                            interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        channels.append(vh)

        if 'cvit' in cfg['backbone'].lower() and not args.replace_rgb_with_others:
            channels.append(vh)

        channels.append(vv)
        if 'cvit' in cfg['backbone'].lower() and not args.replace_rgb_with_others:
            channels.append(vv)


        if 'cvit' not in cfg['backbone'].lower():
            zero_channel = torch.zeros(data_cfg['image_size'] , data_cfg['image_size'] ).unsqueeze(0)
            if args.vh_vv_mean:
                channels.append(((vh.float() + vv.float()) / 2).to(torch.uint8))
            else:
                channels.append(zero_channel)
            
        if 'satlas' in cfg['encoder_weights'].lower():
            for i in range(6):
                zero_channel = torch.zeros(data_cfg['image_size'] , data_cfg['image_size'] ).unsqueeze(0)
                channels.append(zero_channel)
    
        img = torch.cat(channels, dim=0)
        img = img.float().div(255)
        img = img.unsqueeze(0).to(device)
    
        labels_path = os.path.join(root_path, s1_path, labels)
        
        with open(labels_path, 'r') as f:
            labels = json.load(f)['labels']
        target = get_multihot_new(labels)
        target = torch.from_numpy(target)
        target = target.unsqueeze(0)
        gts.append(target.int())
        if 'cvit-pretrained' in cfg['backbone'].lower():
            logits = model(img)
        elif 'cvit' in cfg['backbone'].lower():
            model.channels = cvit_channels
            logits = model(img)
        elif 'clay' in cfg['backbone'].lower():
            logits = model(img, [metadata])
        elif 'dofa' in cfg['backbone'].lower():
            logits = model(img, metadata['waves'])
        else:
            logits = model(img)
        preds.append(logits.cpu().detach())
        
    accuracy = test_accuracy(torch.tensor(np.array(preds)), torch.tensor(np.array(gts))).to(device).detach()
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    with open(f"{args.filename}.txt", "a") as log_file:
        log_file.write(f"{args.checkpoint_path}" +"\n" + f"{accuracy * 100:.2f}" + "\n")

    results[args.checkpoint_path]['vvvh'] = accuracy * 100
            
    save_directory = f'./eval_outs/{args.checkpoint_path.split("/")[-2]}'

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    savefile = f'{save_directory}/results_sar.npy'
    np.save(savefile, results)

    print(results)

def main(args):
    with open(f"{args.filename}.txt", "a") as log_file:
        log_file.write(f"{args.checkpoint_path}" + "\n")

    # if 'brick' in args.dataset_config and args.save_encoder_features:
    #     checkpoint_name = args.checkpoint_path.split('/')[-1].replace('.ckpt', '').replace('.pth', '')
    #     os.makedirs(f'./brickkiln_features_train_{checkpoint_name}', exist_ok=True)

    if 'eurosat' in args.dataset_config and args.save_encoder_features:
        os.makedirs('./eurosat_features', exist_ok=True)

    if 'so2sat' in args.dataset_config and args.save_encoder_features:
        print(f"Creating SO2Sat directory: /nfs/dgx/raid/rs/rs/so2sat_UB_S2/{args.so2sat_folder_name}")
        os.makedirs(f'/nfs/dgx/raid/rs/rs/so2sat_UB_S2/{args.so2sat_folder_name}', exist_ok=True)
        print(f"Directory created successfully")

    bands = json.loads(args.bands)
    
    if args.sar:
        eval_sar(args)
    else:
        results = {}
        results[args.checkpoint_path] = {}

        with open(args.model_config) as config:
            cfg = json.load(config)
        
        with open(args.dataset_config) as config:
            data_cfg = json.load(config)

        print(f"Dataset config loaded: {data_cfg}")
        print(f"Dataset name: {data_cfg.get('dataset_name', 'NOT_FOUND')}")
        print(f"Save encoder features flag: {args.save_encoder_features}")


        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(args.checkpoint_path, cfg)
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        prefix='encoder'
        if 'ben' in data_cfg['dataset_name']:
            multilabel = True
        else:
            multilabel = False
        model = tr_cls.Classifier(backbone_name=cfg['backbone'], backbone_weights=cfg['encoder_weights'], 
                                    in_features=cfg['in_features'], num_classes=data_cfg['num_classes'],
                                lr=0.0, scheduler='', checkpoint_path=args.checkpoint_path, only_head='',
                                warmup_steps = '', eta_min = '', warmup_start_lr='', weight_decay= '', 
                                prefix=prefix, mixup=False, multilabel=multilabel,
                                enable_multiband_input=args.enable_multiband_input,
                                multiband_channel_count=args.multiband_channel_count,
                                shared_proj=args.shared_proj, add_ch_embed=args.add_ch_embed) #, channels=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13])
        model.load_state_dict(checkpoint['state_dict'])
        
        if args.preserve_rgb_weights:
            from classifier_utils import adapt_encoder_for_multiband_eval
            
            # Adapt the encoder to handle multiband input while preserving existing band weights
            print("Adapting encoder for multiband input while preserving existing weights...")
            print(f"Current encoder input channels: {getattr(model.encoder, 'in_chans', 'unknown')}")
            print(f"Target multiband channels: {args.multiband_channel_count}")
            
            success = adapt_encoder_for_multiband_eval(
                encoder=model.encoder, 
                multiband_channel_count=args.multiband_channel_count
            )
            cfg['in_channels'] = args.multiband_channel_count
            if not success:
                print("Warning: Failed to adapt encoder for multiband input")
            else:
                print("Successfully adapted encoder for multiband input")
                print(f"Updated encoder input channels: {getattr(model.encoder, 'in_chans', 'unknown')}")
        
        model.eval()
        model = model.to(device)
        
        if multilabel:
            test_map = AveragePrecision(num_classes=data_cfg['num_classes'], average='micro', task='binary')
            test_f1 =  F1Score(task='multilabel', num_labels=data_cfg['num_classes'], threshold=0.5, average='micro').to(device)
            test_accuracy = Accuracy(num_labels=data_cfg['num_classes'], task='multilabel').to(device) 
        else:
            test_accuracy = Accuracy(num_classes=data_cfg['num_classes'], task='multiclass').to(device)

        results[args.checkpoint_path] = {}
        for band in bands :
            model.bands = band
            get_indicies = []

            print('band1: ', band)

            for b in band:
                # if '_' in b:
                #     first_band, second_band = b.split('_')
                #     get_indicies.append(channel_vit_order.index(first_band))
                #     band[band.index(b)] = second_band
                # else:
                get_indicies.append(channel_vit_order.index(b))

            if args.replace_rgb_with_others:
                get_indicies = [0, 1, 2]
            
            if args.band_mean_repeat_count != 0:
                get_indicies = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]

            print('band2: ', band)
            print("get_indicies: ", get_indicies)

            bands_order = get_band_orders(model_name=cfg['backbone'])
            rgb_bands = get_band_orders(model_name=cfg['backbone'], rgb=True)
            custom_collate_fn = create_collate_fn('classification')

            if 'eurosat' in data_cfg['dataset_name']:
                # ms_dir = data_cfg['base_dir']
                # sar_dir = data_cfg['base_dir'].replace('-MS', "-SAR")
                # split_path = data_cfg['splits_dir']

                # datamodule = EuroSATCombinedDataset(ms_dir, sar_dir, band, split_path, 
                #                                     img_size=data_cfg['image_size'], split='test')
                
                # test_dataloader = DataLoader(datamodule, batch_size=data_cfg['batch_size'], 
                #                              shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
                datamodule = mEurosat(split='test', bands=band, img_size=args.img_size)
                test_dataloader = DataLoader(datamodule, batch_size=data_cfg['batch_size'], collate_fn=custom_collate_fn)

            elif 'so2sat' in data_cfg['dataset_name']:
                datamodule = So2SatDataset(split='test', bands=band, img_size=args.img_size)
                test_dataloader = DataLoader(datamodule, batch_size=data_cfg['batch_size'], collate_fn=custom_collate_fn)
            elif 'm_ben' in data_cfg['dataset_name']:
                datamodule = mBigearthnet(split='test', bands=band, img_size=args.img_size)
                test_dataloader = DataLoader(datamodule, batch_size=data_cfg['batch_size'], collate_fn=custom_collate_fn)
            elif 'brick' in data_cfg['dataset_name']:
                datamodule = BrickKiln(split='test', bands=band, img_size=args.img_size)
                test_dataloader = DataLoader(datamodule, batch_size=data_cfg['batch_size'], collate_fn=custom_collate_fn)
            # else:
            #     datamodule = BigearthnetDataModule(data_dir=data_cfg['base_dir'], batch_size=data_cfg['batch_size'],
            #                             num_workers=24, img_size=data_cfg['image_size'] , replace_rgb_with_others=args.replace_rgb_with_others, 
            #                             bands=band, splits_dir=data_cfg['splits_dir'], fill_zeros=cfg['fill_zeros'], 
            #                             weighted_input= args.weighted_input, weight=args.weight,
            #                             band_mean_repeat_count=args.band_mean_repeat_count,
            #                             bands_order=bands_order, rgb_bands=rgb_bands)
            

                # datamodule.setup()
                # test_dataloader = datamodule.test_dataloader()

            with torch.no_grad():
                correct_predictions = 0
                correct_predictions_map = 0
                correct_predictions_f1 = 0
                total_samples = 0
                for batch in tqdm(test_dataloader):
                    # if 'ben' in data_cfg['dataset_name'] or 'eurosat' in data_cfg['dataset_name']:
                    x, y, metadata = batch
                    if 'anysat' in cfg['backbone'].lower() and len(model.bands) == 2:
                        zero_ch = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
                        x = torch.cat([x, zero_ch], dim=1)
                    if 'satlas' in cfg['backbone'].lower():
                        zero_channels = torch.zeros(x.shape[0], 12 - x.shape[1], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
                        x = torch.cat([x, zero_channels], dim=1)
                    if 'prithvi' in cfg['backbone'].lower():
                        zero_channels = torch.zeros(x.shape[0], 12 - x.shape[1], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
                        x = torch.cat([x, zero_channels], dim=1)
                    if args.band_mean_repeat_count != 0:
                        for item in metadata:
                            wave_values = item['waves']
                            mean_val = sum(wave_values) / len(wave_values)
                            item['waves'].extend([mean_val] * args.band_mean_repeat_count)
                    # else:
                    #     x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    if 'cvit' in cfg['backbone'].lower():
                        model.channels = get_indicies
                        logits = model(x)
                    elif 'clay' in cfg['backbone'].lower():
                        logits = model(x, metadata)
                    elif 'dofa' in cfg['backbone'].lower():
                        logits = model(x, metadata[0]['waves'])
                    else:
                        if x.shape[1] == 2:
                            zero_channel = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
                            x = torch.cat([x, zero_channel], dim=1)
                        if args.enable_multiband_input:
                            zero_channels = torch.zeros(x.shape[0], 
                                    args.multiband_channel_count-x.shape[1], 
                                    x.shape[2], x.shape[3]).to(device)
                            x = torch.cat([x, zero_channels], dim=1)
                        
                    # if 'brick' in data_cfg['dataset_name'] and args.save_encoder_features:
                    #     checkpoint_name = args.checkpoint_path.split('/')[-2]
                    #     os.makedirs(f'./brickkiln_features_test_{checkpoint_name}', exist_ok=True)
                    #     with torch.no_grad():
                    #         if 'anysat' in cfg['backbone'].lower():
                    #             zero_channel = torch.zeros(x.shape[0], 3 - x.shape[1], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
                    #             x = torch.cat([x, zero_channel], dim=1)
                    #             feats = model.encoder({'_rgb': x}, patch_size=10, output='tile')
                    #         else:
                    #             feats = model.encoder(x)
                    #         for idx, (feat, true_class) in enumerate(zip(feats, y)):
                    #             image_id = f"brick_{total_samples + idx}"
                    #             feat_np = feat.cpu().numpy()
                    #             class_np = true_class.cpu().numpy()
                    #             np.save(f'./brickkiln_features_test_{checkpoint_name}/{image_id}_{"".join(band)}.npy', feat_np)
                    #             np.save(f'./brickkiln_features_test_{checkpoint_name}/{image_id}_{"".join(band)}_class.npy', class_np)
                        
                    if args.save_encoder_features:
                        print(f"SO2Sat feature extraction triggered for band: {band}")
                        with torch.no_grad():      
                            print("cfg['backbone'].lower(): ", cfg['backbone'].lower())                      
                            if 'anysat' in cfg['backbone'].lower():
                                zero_channel = torch.zeros(x.shape[0], 10 - x.shape[1], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
                                x = torch.cat([x, zero_channel], dim=1)
                                feats = model.encoder({'_s2': x}, patch_size=10, output='tile')
                            elif 'swin-b' in cfg['backbone'].lower():
                                zero_channels = torch.zeros(x.shape[0], 12 - x.shape[1], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
                                x = torch.cat([x, zero_channels], dim=1)
                                feats = model.encoder(x)[-1]
                                norm_layer = torch.nn.LayerNorm([1024, 4, 4], device=x.device) 
                                global_average_pooling = torch.nn.AdaptiveAvgPool2d(1)
                                feats = norm_layer(feats)
                                feats = global_average_pooling(feats)
                                feats = torch.flatten(feats, 1)
                            elif 'cvit-pretrained' in cfg['backbone'].lower():
                                model.channels = get_indicies
                                feats = model.encoder(x, channel_idxs=get_indicies)
                            elif "dinov3" in cfg['backbone'].lower():
                                feats = model.encoder(x).last_hidden_state[:, 0]
                            elif 'dofa' in cfg['backbone'].lower():
                                feats = model(x, metadata[0]['waves'])
                            elif 'prithvi' in cfg['backbone'].lower():
                                zero_channels = torch.zeros(x.shape[0], 12 - x.shape[1], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
                                x = torch.cat([x, zero_channels], dim=1)
                                feats = model.encoder(x)
                            else:
                                feats = model.encoder(x)
                            print(f"Extracted features shape: {feats.shape}")
                            for idx, (feat, true_class) in enumerate(zip(feats, y)):
                                image_id = f"so2sat_{total_samples + idx}"
                                feat_np = feat.cpu().numpy()
                                class_np = true_class.cpu().numpy()
                                feat_path = f'/nfs/dgx/raid/rs/rs/so2sat_UB_S2/{args.so2sat_folder_name}/{image_id}_{"".join(band)}.npy'
                                class_path = f'/nfs/dgx/raid/rs/rs/so2sat_UB_S2/{args.so2sat_folder_name}/{image_id}_{"".join(band)}_class.npy'
                                print(f"Saving feature to: {feat_path}")
                                np.save(feat_path, feat_np)
                                np.save(class_path, class_np)
                            print(f"Saved {len(feats)} features for band {band}")

                        
                    logits = model(x)
                    
                    if multilabel:
                        batch_map = test_map(logits, y.int()).to(device)
                        batch_f1 = test_f1(torch.sigmoid(logits), y.int()).to(device)
                        batch_accuracy = test_accuracy(torch.sigmoid(logits), y).to(device)
                        
                    else:
                        batch_accuracy = test_accuracy(torch.argmax(logits, dim=1), y).to(device)
                    correct_predictions += batch_accuracy.item() * len(y)
                    if multilabel:
                        correct_predictions_map += batch_map.item() * len(y)
                        correct_predictions_f1 += batch_f1.item() * len(y)
                    total_samples += len(y)
                    # print(correct_predictions / total_samples)

                overall_test_accuracy = correct_predictions / total_samples
                overall_test_map = correct_predictions_map / total_samples
                overall_test_f1 = correct_predictions_f1 / total_samples
            print(args.checkpoint_path)
            print(f'Test Accuracy: {overall_test_accuracy * 100:.2f}%')
            with open(f"{args.filename}.txt", "a") as log_file:
                log_file.write(f"{band}" + "  " + f"{overall_test_accuracy * 100:.2f}" + \
                                "  " + f"{overall_test_map * 100:.2f}" + " " +  f"{overall_test_f1 * 100:.2f}" +"\n")
            results[args.checkpoint_path][''.join(band)] = overall_test_accuracy * 100
            
        save_directory = f'./eval_outs/{args.checkpoint_path.split("/")[-2]}'

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        savefile = f'{save_directory}/results.npy'
        np.save(savefile, results)

        print(results)

if __name__ == '__main__':
    
    channel_vit_order = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A',  'B11', 'B12', 'VV', 'VH'] #VVr VVi VHr VHi
    all_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A','B11', 'B12','vv', 'vh']

    parser = ArgumentParser()
    parser.add_argument('--model_config', type=str, default='')
    parser.add_argument('--dataset_config', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--sar', action="store_true")
    parser.add_argument('--filename', type=str, default='eval_bands_cls_log')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--replace_rgb_with_others', action="store_true")
    # parser.add_argument("--bands", type=str, default=json.dumps([['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'], ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', 'VV', 'VH'], ['VV', 'VH']]))
    # parser.add_argument("--bands", type=str, default=json.dumps([['B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'], ['B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', 'VV', 'VH'], ['VV', 'VH']]))
    # parser.add_argument("--bands", type=str, default=json.dumps([['VV', 'VH']]))

    parser.add_argument("--bands", type=str, default=json.dumps([['B04', 'B03', 'B02'], ['B04','B03','B05'], ['B04', 'B05', 'B06'], ['B8A', 'B11', 'B12'], ['VV', 'VH']]))
    # parser.add_argument("--bands", type=str, default=json.dumps([['B02', 'B03', 'B04'], ['B05','B03','B04'], ['B06', 'B05', 'B04'], ['B8A', 'B11', 'B12'], ['VV', 'VH']]))
    parser.add_argument('--weighted_input', action="store_true") 
    parser.add_argument('--shared_proj', action='store_true')
    parser.add_argument('--add_ch_embed', action='store_true')  
    parser.add_argument('--weight', type=float, default=1) 
    parser.add_argument('--vh_vv_mean', action="store_true") 
    parser.add_argument('--repeat_values', action="store_true")
    parser.add_argument('--band_mean_repeat_count', type=int, default=0)

    parser.add_argument('--multiband_channel_count', type=int, default=12)
    parser.add_argument('--enable_multiband_input', action='store_true')
    parser.add_argument('--preserve_rgb_weights', action='store_true')
    parser.add_argument('--save_encoder_features', action='store_true', help='Save DINO feature vectors for analysis')
    parser.add_argument('--so2sat_folder_name', type=str, default='so2sat_features', help='Folder name for saving SO2Sat features')

    args = parser.parse_args()
    main(args)

