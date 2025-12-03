import os
import json
import torch
import rasterio
import numpy as np
import torch.distributed as dist
import change_detection_pytorch as cdp
from typing import Optional

from PIL import Image
# from osgeo import gdal
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
from eval_scale_cd import CustomMetric, load_model, init_dist
from utils_terramind_adapt import (
    replace_terramind_projection_layer,
    adapt_terramind_state_dict,
    adapt_terramind_s2_to_s2s1,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from change_detection_pytorch.datasets import BuildingDataset, Sen1Floods11, mCashewPlantation, mSAcrop
from change_detection_pytorch.datasets import normalize_channel, RGB_BANDS, STATS
from evaluator import SegEvaluator
from utils import create_collate_fn
from normalize_bands import normalize_band_names


def compute_binary_iou(pred_masks, gt_masks):
    """
    Compute binary IoU (bIoU) between predictions and ground truth masks.
    Formula: bIoU = TP / (TP + FP + FN)
    
    Args:
        pred_masks: list of numpy arrays of shape (H, W) with predicted binary masks (0 or 1)
        gt_masks: list of numpy arrays of shape (H, W) with ground truth binary masks (0 or 1)
        
    Returns:
        binary_iou: scalar binary IoU value (as percentage)
    """
    if len(pred_masks) == 0:
        return 0.0
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        # True Positives: predicted=1, ground truth=1
        tp = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
        # False Positives: predicted=1, ground truth=0
        fp = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
        # False Negatives: predicted=0, ground truth=1
        fn = np.logical_and(pred_mask == 0, gt_mask == 1).sum()
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Compute binary IoU: TP / (TP + FP + FN)
    denominator = total_tp + total_fp + total_fn
    if denominator > 0:
        biou = total_tp / denominator
    else:
        biou = 0.0
    
    return biou * 100  # Return as percentage


def get_terramind_bands(ms=False):
    """Always return RGB bands for TerraMind (B02, B03, B04)"""
    if ms:
        print('returning ms bands')
        return ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    else:
        return ['B02', 'B03', 'B04']


def parse_encoder_bands_arg(bands_arg: Optional[str], backbone: Optional[str] = None):
    # If encoder_bands is provided, respect it for all backbones (including TerraMind)
    if not bands_arg:
        # Default behavior: for TerraMind, use RGB or MS bands based on multiband flag
        if backbone and 'terramind' in backbone.lower():
            # This will be handled in main() where args is available
            return None
        return None
    
    try:
        parsed = json.loads(bands_arg)
    except json.JSONDecodeError:
        parsed = bands_arg.split()
    if isinstance(parsed, dict):
        first_value = next(iter(parsed.values()), [])
        parsed = first_value
    # Handle list of lists (e.g., [["B02", "B03", ...]]) - extract first inner list
    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
        parsed = parsed[0]
    if isinstance(parsed, str):
        parsed = [parsed]
    normalized = normalize_band_names(parsed)
    if isinstance(normalized, str):
        return [normalized]
    return normalized


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

    # Parse encoder bands - respect --encoder_bands if provided, otherwise use defaults
    encoder_bands = parse_encoder_bands_arg(args.encoder_bands, backbone=cfg.get('backbone', ''))
    
    # Default behavior for TerraMind if encoder_bands not provided
    backbone_name = cfg.get('backbone', '').lower()
    if encoder_bands is None and 'terramind' in backbone_name:
        encoder_bands = get_terramind_bands(ms=args.enable_multiband_input)

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
                multiband_channel_count=initial_channels,
                bands=encoder_bands,
                # channels=args.channels  #[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
            )
    model.to(args.device)
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    finetuned_model = torch.load(args.checkpoint_path, map_location=args.device)
    msg = model.load_state_dict(finetuned_model)
    print(f"Model loaded. Checkpoint path key: {args.checkpoint_path}")

    # Optional TerraMind RGB -> multiband adaptation for segmentation
    backbone_name = cfg.get('backbone', '').lower()
    is_terramind = 'terramind' in backbone_name
    if (
        is_terramind
        and args.enable_multiband_input
        and args.multiband_channel_count > 3
        and args.adapt_terramind_rgb_to_multiband
    ):
        print("Adapting TerraMind encoder for multiband input in segmentation (RGB -> multiband)...")
        # Save current encoder weights (assumed RGB-trained)
        rgb_enc_sd = model.encoder.state_dict()

        # Resize projection for desired number of bands
        model.encoder = replace_terramind_projection_layer(
            model.encoder,
            num_bands=args.multiband_channel_count,
            patch_size=16,
            embed_dim=getattr(model.encoder, "embed_dim", 768),
        )

        # Adapt weights from RGB to multiband (extra bands = mean of RGB)
        adapted_sd = adapt_terramind_state_dict(
            rgb_enc_sd,
            model.encoder,
            patch_size=16,
        )
        model.encoder.load_state_dict(adapted_sd, strict=False)
        print("TerraMind encoder adapted for multiband segmentation.")

    # Optional TerraMind S2-only -> S2+S1 adaptation for segmentation
    if (
        is_terramind
        and args.adapt_terramind_s2_to_s2s1
    ):
        print("Adapting TerraMind encoder from S2-only to S2+S1 for segmentation...")
        from change_detection_pytorch.encoders.terramind import TerraMindEncoder
        device = args.device if torch.cuda.is_available() else 'cpu'
        
        # Get current encoder parameters
        current_encoder = model.encoder
        if hasattr(current_encoder, 'model'):
            # TerraMindEncoder wraps the actual model
            s2_model = current_encoder.model
            
            # Add S1GRD bands to the bands dictionary
            s2s1_bands = current_encoder.bands.copy() if current_encoder.bands is not None else {}
            s2s1_bands['S1GRD'] = ['VV', 'VH']
            print(f"Updated bands for S2+S1: {s2s1_bands}")
            
            # Build new S2+S1 encoder with same parameters
            s2s1_encoder = TerraMindEncoder(
                model_name=current_encoder.model_name,
                pretrained=False,
                modalities=['S2L2A', 'S1GRD'],
                img_size=current_encoder.img_size,
                patch_size=current_encoder.patch_size,
                bands=s2s1_bands,
                for_cls=False,
                out_idx=current_encoder.out_idx,
            )
            s2s1_encoder = s2s1_encoder.to(device)
            
            # Adapt weights from S2-only to S2+S1
            adapt_terramind_s2_to_s2s1(s2_model, s2s1_encoder.model)
            
            # Replace encoder in model
            model.encoder = s2s1_encoder
            print("TerraMind encoder adapted from S2-only to S2+S1 for segmentation.")
        else:
            print("Warning: Could not find TerraMind model structure for S2->S2+S1 adaptation")

    if args.preserve_rgb_weights and not is_terramind:
        # Legacy path for non-TerraMind backbones
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
    print(f"Initializing results for checkpoint: {args.checkpoint_path}")

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

        # Compute bIoU for Harvey dataset
        biou = None
        if 'harvey' in dataset_name:
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for data in tqdm(test_loader, desc="Computing bIoU"):
                    image, target, _, metadata = data
                    image = image.to(args.device)
                    target = target.to(args.device)
                    
                    logits = model(image, metadata=metadata)
                    if logits.shape[1] == 1:
                        pred = (torch.sigmoid(logits) > 0.5).type(torch.int64).squeeze(dim=1)
                    else:
                        pred = torch.argmax(logits, dim=1)
                    
                    # Store predictions and targets for class 1 (building class)
                    for p, t in zip(pred.cpu().numpy(), target.cpu().numpy()):
                        pred_class1 = (p == 1).astype(np.uint8)
                        gt_class1 = (t == 1).astype(np.uint8)
                        all_preds.append(pred_class1)
                        all_targets.append(gt_class1)
            
            if len(all_preds) > 0:
                biou = compute_binary_iou(all_preds, all_targets)
                print(f'bIoU: {biou:.2f}')

        result_dict = {
            'iou_score': metric
        }
        if biou is not None:
            result_dict['biou'] = biou
        results[args.checkpoint_path][''.join(band)] = result_dict

        with open(f"{args.filename}.txt", "a") as log_file:
            log_file.write(f'{args.checkpoint_path}' + "\n")
            log_file.write(f'{band} : {metric}' + "\n")
    
    save_directory = f'./eval_outs/{args.checkpoint_path.split("/")[-2]}'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    savefile = f'{save_directory}/results.npy'
    print(f"Saving results to: {savefile}")
    print(f"Results keys before save: {list(results.keys())}")
    # Load existing results if file exists and merge
    if os.path.exists(savefile):
        existing_results = np.load(savefile, allow_pickle=True).item()
        print(f"Existing results keys: {list(existing_results.keys())}")
        # Merge existing results with new results
        for key in results:
            if key in existing_results:
                print(f"Warning: Checkpoint path {key} already exists in results. Merging band results.")
                # Merge band results
                existing_results[key].update(results[key])
            else:
                existing_results[key] = results[key]
        results = existing_results
    print(f"Final results keys: {list(results.keys())}")
    print(f"Results for current checkpoint: {results.get(args.checkpoint_path, 'NOT FOUND')}")
    np.save(savefile, results)






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
    parser.add_argument('--encoder_bands', type=str, default=None, help="Bands to pass to TerraMind encoders (JSON list or space-separated).")
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
    parser.add_argument(
        '--adapt_terramind_rgb_to_multiband',
        action='store_true',
        help='Adapt TerraMind RGB encoder to multiband (e.g., RGBN) for segmentation eval.',
    )
    parser.add_argument(
        '--adapt_terramind_s2_to_s2s1',
        action='store_true',
        help='Adapt TerraMind S2-only encoder to S2+S1 (multimodal) for segmentation eval.',
    )



    args = parser.parse_args()
    main(args)


