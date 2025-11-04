import os
import json
import torch
import rasterio
import numpy as np
import torch.distributed as dist
import change_detection_pytorch as cdp

from PIL import Image
from osgeo import gdal
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from eval_scale_cd import CustomMetric, load_model, init_dist
from torch.nn.parallel import DistributedDataParallel as DDP
from change_detection_pytorch.datasets import BuildingDataset, Sen1Floods11, mCashewPlantation, mSAcrop
from change_detection_pytorch.datasets import normalize_channel, RGB_BANDS, STATS
from evaluator import SegEvaluator
from utils import create_collate_fn


def main(args):
    init_dist(args.master_port)
    
    bands = json.loads(args.bands)
    results = {}
    with open(args.model_config) as config:
        cfg = json.load(config)
    
    with open(args.dataset_config) as config:
        data_cfg = json.load(config)

    dataset_path = data_cfg['dataset_path']
    metadata_dir = data_cfg['metadata_dir']
    dataset_name = data_cfg['dataset_name']
    # tile_size = data_cfg['tile_size']
    batch_size = data_cfg['batch_size']
    fill_zeros = cfg['fill_zeros']
    tile_size = args.size

    if not args.enable_multiband_input and args.multiband_channel_count > 3:
        initial_channels = 3
    else:
        initial_channels = args.multiband_channel_count

    model = cdp.UPerNetSeg(
                encoder_depth=cfg['encoder_depth'],
                encoder_name=cfg['backbone'], # choose encoder, e.g. overlap_ibot-B, mobilenet_v2 or efficientnet-b7
                encoder_weights=cfg['encoder_weights'], # use `imagenet` pre-trained weights for encoder initialization
                in_channels=cfg['in_channels'], # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                decoder_psp_channels=args.upernet_width * 2,
                decoder_pyramid_channels=args.upernet_width,
                decoder_segmentation_channels=args.upernet_width,
                decoder_merge_policy="add",
                classes=args.classes, # model output channels (number of classes in your datasets)
                activation=None,
                freeze_encoder=False,
                pretrained = False,
                upsampling=args.upsampling,
                out_size=args.size,
                enable_multiband_input=args.enable_multiband_input,
                multiband_channel_count=initial_channels
                # channels=args.channels  #[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
            )
    model.to(args.device)
    # model = DDP(model)
    finetuned_model = torch.load(args.checkpoint_path, map_location=args.device)
    msg = model.load_state_dict(finetuned_model)

    if args.preserve_rgb_weights:
        from classifier_utils import adapt_encoder_for_multiband_eval
        
        adapt_encoder_for_multiband_eval(
            encoder=model.encoder, 
            multiband_channel_count=args.multiband_channel_count
        )

    loss = cdp.utils.losses.CrossEntropyLoss()
    if args.use_dice_bce_loss:
        loss = cdp.utils.losses.dice_bce_loss()

    # DEVICE = 'cuda:{}'.format(dist.get_rank()) if torch.cuda.is_available() else 'cpu'
    results[args.checkpoint_path] = {}

    for band in bands :            

        if 'cvit' in model.encoder_name.lower():
            print('band1: ', band)
            get_indicies = []
            for b in band:
                get_indicies.append(channel_vit_order.index(b))
            
            if args.fill_mean:
                get_indicies = [0, 1, 2, 3, 4 ,5, 6, 7, 8, 9, 10]
            elif args.fill_zeros:
                for _ in range(args.band_repeat_count):
                    get_indicies.append(0)

            if args.replace_rgb_with_others:
                get_indicies = [0, 1, 2]

            print('band2: ', band)

            model.channels = get_indicies

        if 'clay' in model.encoder_name.lower():
            for b in band:
                if '_' in b:
                    first_band, second_band = b.split('_')
                    band[band.index(b)] = second_band

        
        if 'sen1floods11' in dataset_name:
            test_dataset = Sen1Floods11(bands=band, split = 'test', img_size=tile_size, fill_zeros=args.fill_zeros)
        elif 'harvey' in dataset_name:
            test_dataset = BuildingDataset(split_list=f"{dataset_path}/test.txt", 
                                            img_size=args.size,
                                            fill_zeros=args.fill_zeros,
                                            fill_mean=args.fill_mean,
                                            band_repeat_count=args.band_repeat_count,
                                            weighted_input=args.weighted_input,
                                            weight=args.weight,
                                            replace_rgb_with_others=args.replace_rgb_with_others,
                                            bands=band)
                
        elif 'cashew' in dataset_name:
            test_dataset = mCashewPlantation(split='test',
                                        bands=band,
                                        img_size=args.size,
                                        fill_zeros=args.fill_zeros,
                                        )
        elif 'crop' in dataset_name:
            test_dataset = mSAcrop(split='test',
                                bands=band,
                                img_size=args.size,
                                fill_zeros=args.fill_zeros,
                                )


        custom_collate_fn = create_collate_fn('segmentation')
        
        test_loader=DataLoader(test_dataset, drop_last=False, collate_fn=custom_collate_fn)
        
        # valid_epoch = cdp.utils.train.ValidEpoch(
        #         model,
        #         loss=loss,
        #         metrics=metrics,
        #         device='cuda:{}'.format(dist.get_rank()),
        #         verbose=True,
        #     )

        # test_logs = valid_epoch.run_seg(test_loader)
        # results[args.checkpoint_path][''.join(band)] = {
        #     'iou_score': test_logs['IoU'],
        # }

        evaluator = SegEvaluator(
                    val_loader=test_loader,
                    exp_dir='',
                    device=args.device,
                    inference_mode="whole",  # or "whole", as needed
                    sliding_inference_batch=batch_size,  # if using sliding mode
                )
        
        metrics, used_time = evaluator(model, model_name="seg_model")
        print("Evaluation Metrics from checkpoint:", metrics)
        
        if 'cashew' in dataset_name or 'crop' in dataset_name:
            metric = metrics['mIoU']
        else:
            metric = metrics['IoU'][1]

        with open(f"{args.filename}.txt", "a") as log_file:
            log_file.write(f'{args.checkpoint_path}' + "\n")
            log_file.write(f'{band} : {metric}' + "\n")

    # with open(f"{args.filename}.txt", "a") as log_file:
    #     log_file.write(f'{args.checkpoint_path}' + "\n")
    #     for b in bands:
    #         log_file.write(f"{b}" + "  " + 'IoU:  ')
    #         message = f"{results[args.checkpoint_path][''.join(b)]['iou_score'] * 100:.2f}"
    #         print(message)
    #         log_file.write(message + "\n")






if __name__== '__main__':
    
    channel_vit_order = ['B4', 'B3', 'B2', 'B5', 'B6', 'B7', 'B8', 'B8A',  'B11', 'B12', 'vv', 'vh'] #VVr VVi VHr VHi
    # channel_vit_order = ['B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A',  'B11', 'B12', 'VV', 'VH'] #VVr VVi VHr VHi

    parser = ArgumentParser()
    # parser.add_argument("--bands", type=str, default=json.dumps([['B02', 'B03', 'B04'], [ 'B05','B03','B04'], ['B06', 'B05', 'B04'], ['B8A', 'B11', 'B12'], ['VV', 'VH', 'VH']]))
    # parser.add_argument("--bands", type=str, default=json.dumps([['VV', 'VH']]))

    parser.add_argument("--bands", type=str, default=json.dumps([['B02', 'B03', 'B04'], [ 'B05','B03','B04'], ['B06', 'B05', 'B04'], ['B8A', 'B11', 'B12'], ['VV', 'VH']]))
    # parser.add_argument("--bands", type=str, default=json.dumps([['B2', 'B3', 'B4'], [ 'B5','B3','B4'], ['B6', 'B5', 'B4'], ['B8A', 'B11', 'B12'], ['vh', 'vv']]))
    # parser.add_argument("--bands", type=str, default=json.dumps([[ 'B4','B3','B2'], ['B4','B3','B5'], ['B4', 'B5', 'B6'], ['B8A', 'B11', 'B12']]))
    # parser.add_argument("--bands", type=str, default=json.dumps([['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'], ['B2', 'B3', 'B4' ], [ 'B5','B3','B4'], ['B6', 'B5', 'B4'], ['B8A', 'B11', 'B12'], ['vh', 'vv']]))
    parser.add_argument('--model_config', type=str, default='')
    parser.add_argument('--dataset_config', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--replace_rgb_with_others', action="store_true")
    parser.add_argument('--upsampling', type=float, default=4)
    parser.add_argument('--master_port', type=str, default="12345")
    parser.add_argument('--filename', type=str, default='eval_bands_seg_log')
    parser.add_argument('--use_dice_bce_loss', action="store_true")
    parser.add_argument('--size', type=int, default=96)
    parser.add_argument('--fill_mean', action="store_true")
    parser.add_argument('--fill_zeros', action="store_true")
    parser.add_argument('--band_repeat_count', type=int, default=0)
    parser.add_argument('--weighted_input', action="store_true") 
    parser.add_argument('--weight', type=float, default=1) 
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--upernet_width', type=int, default=64)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument('--enable_multiband_input', action='store_true')
    parser.add_argument('--multiband_channel_count', type=int, default=3)
    parser.add_argument('--preserve_rgb_weights', action='store_true')



    args = parser.parse_args()
    main(args)


