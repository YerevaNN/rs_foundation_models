import os
import torch
import rasterio
import json
import numpy as np
import change_detection_pytorch as cdp
from typing import Optional
# import torch.distributed as dist

from tqdm import tqdm
# from osgeo import gdal  # Replaced with rasterio to avoid GDAL C extension issues
from itertools import product
from PIL import Image
from argparse import ArgumentParser
from sklearn import metrics
from glob import glob
from tqdm import tqdm
from eval_scale_cd import CustomMetric, load_model, init_dist
from change_detection_pytorch.datasets import ChangeDetectionDataModule, FloodDataset, normalize_channel
from torch.utils.data import DataLoader
from evaluator_change import SegEvaluator
from utils import get_band_orders, create_collate_fn
from normalize_bands import normalize_band_names
from utils_terramind_adapt import (
    replace_terramind_projection_layer,
    adapt_terramind_state_dict,
    adapt_terramind_s2_to_s2s1,
)


RGB_BANDS = ['B02', 'B03', 'B04']
STATS = {
    'mean': {
        'B02': 1422.4117861742477,
        'B03': 1359.4422181552754,
        'B04': 1414.6326650140888,
        'B05': 1557.91209397433,
        'B06': 1986.5225593959844,
        'B07': 2211.038518780755,
        'B08': 2119.168043369016,
        'B8A': 2345.3866026353567,
        'B11': 2133.990133983443,
        'B12': 1584.1727764661696,
        'VV': -9.152486082800158, 
        'VH': -16.23374164784503
        },
    'std' :  {
        'B02': 456.1716680330627,
        'B03': 590.0730894364552,
        'B04': 849.3395398520846,
        'B05': 811.3614662999139,
        'B06': 813.441067258119,
        'B07': 891.792623998175,
        'B08': 901.4549041572363,
        'B8A': 954.7424298485422,
        'B11': 1116.63101989494,
        'B12': 985.2980824905794,
        'VV': 5.41078882186851, 
        'VH': 5.419913471274721}

}


SAR_STATS = {
    'mean': {'VV': -9.152486082800158, 'VH': -16.23374164784503},
    'std': {'VV': 5.41078882186851, 'VH': 5.419913471274721}
} 


def get_terramind_rgb_bands(ms=False):
    """Always return RGB bands for TerraMind (B02, B03, B04)"""
    if ms:
        print('returning ms bands')
        return ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    else:
        return ['B02', 'B03', 'B04']


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


def parse_encoder_bands_arg(bands_arg: Optional[str], backbone: Optional[str] = None):
    # Always use RGB bands for TerraMind, regardless of input
    # if backbone and 'terramind' in backbone.lower():
    #     return get_terramind_rgb_bands(ms=args.enable_multiband_input)
    
    if not bands_arg:
        return None
    try:
        parsed = json.loads(bands_arg)
    except json.JSONDecodeError:
        parsed = bands_arg.split()
    # If dict is provided, take the first modality definition
    print("parsed: ", parsed)
    if isinstance(parsed, dict):
        first_value = next(iter(parsed.values()), [])
        parsed = first_value
    if isinstance(parsed, str):
        parsed = [parsed]
    normalized = normalize_band_names(parsed)
    if isinstance(normalized, str):
        return [normalized]
    return normalized

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
        # Read SAR (e.g., VV, VH) channels with rasterio instead of GDAL
        with rasterio.open(path) as src:
            img = src.read()  # shape: (bands, H, W)

        vv_intensity = img[0]
        vh_intensity = img[1]

        vv = normalize_channel(
            vv_intensity,
            mean=SAR_STATS['mean']['VV'],
            std=SAR_STATS['std']['VV'],
        )
        vh = normalize_channel(
            vh_intensity,
            mean=SAR_STATS['mean']['VH'],
            std=SAR_STATS['std']['VH'],
        )

        channels.append(vv)
        channels.append(vh)
        
    img = np.dstack(channels)
    img_clipped = np.clip(img, 0.0, 1.0)
    img = (img_clipped * 255).astype(np.uint8)

    img = Image.fromarray(img)
        
    return img

def eval_on_sar(args, encoder_bands=None):
    test_cities = '/nfs/ap/mnt/frtn/rs-multiband/OSCD/test.txt'
    with open(test_cities) as f:
        test_set = f.readline()
    test_set = test_set[:-1].split(',')
    save_directory = f'./eval_outs/{args.checkpoint_path.split("/")[-2]}'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with open(args.model_config) as config:
        cfg = json.load(config)
    
    channels = [10,11,12,13] if 'cvit' in cfg['backbone'].lower() else [0, 1, 2]

    if args.replace_rgb_with_others and 'cvit' in cfg['backbone'].lower():
        channels = [0, 1]
    
    model = load_model(args.checkpoint_path, encoder_depth=cfg['encoder_depth'], backbone=cfg['backbone'], 
                       encoder_weights=cfg['encoder_weights'], fusion=cfg['fusion'], upsampling=args.upsampling, out_size=args.size,
                       load_decoder=cfg['load_decoder'], channels=args.cvit_channels, in_channels=cfg['in_channels'], upernet_width=args.upernet_width,
                       enable_multiband=args.enable_multiband_input, multiband_channel_count=args.multiband_channel_count,
                       bands_param=encoder_bands, strict_loading=args.strict_loading)
    
    # Optional TerraMind RGB -> multiband adaptation for SAR evaluation
    backbone_name = cfg.get('backbone', '').lower()
    is_terramind = 'terramind' in backbone_name
    if (
        is_terramind
        and args.enable_multiband_input
        and args.multiband_channel_count > 3
        and args.adapt_terramind_rgb_to_multiband
    ):
        print("Adapting TerraMind encoder for multiband input in SAR evaluation (RGB -> multiband)...")
        # load_model returns DDP(model); unwrap to access encoder
        enc_container = model.module if hasattr(model, 'module') else model
        
        # Adapt main encoder
        encoder = enc_container.encoder
        rgb_enc_sd = encoder.state_dict()
        encoder = replace_terramind_projection_layer(
            encoder,
            num_bands=args.multiband_channel_count,
            patch_size=16,
            embed_dim=getattr(encoder, "embed_dim", 768),
        )
        adapted_sd = adapt_terramind_state_dict(
            rgb_enc_sd,
            encoder,
            patch_size=16,
        )
        encoder.load_state_dict(adapted_sd, strict=False)
        enc_container.encoder = encoder
        
        # If non-siam encoder exists, adapt it too
        if hasattr(enc_container, 'encoder_non_siam'):
            encoder_non_siam = enc_container.encoder_non_siam
            rgb_enc_non_siam_sd = encoder_non_siam.state_dict()
            encoder_non_siam = replace_terramind_projection_layer(
                encoder_non_siam,
                num_bands=args.multiband_channel_count,
                patch_size=16,
                embed_dim=getattr(encoder_non_siam, "embed_dim", 768),
            )
            adapted_non_siam_sd = adapt_terramind_state_dict(
                rgb_enc_non_siam_sd,
                encoder_non_siam,
                patch_size=16,
            )
            encoder_non_siam.load_state_dict(adapted_non_siam_sd, strict=False)
            enc_container.encoder_non_siam = encoder_non_siam

        print("TerraMind encoder adapted for multiband SAR evaluation.")

    # Optional TerraMind S2-only -> S2+S1 adaptation for SAR evaluation
    if (
        is_terramind
        and args.adapt_terramind_s2_to_s2s1
    ):
        print("Adapting TerraMind encoder from S2-only to S2+S1 for SAR evaluation...")
        from change_detection_pytorch.encoders.terramind import TerraMindEncoder
        
        # load_model returns DDP(model); unwrap to access encoder
        enc_container = model.module if hasattr(model, 'module') else model
        encoder = enc_container.encoder_non_siam if hasattr(enc_container, 'encoder_non_siam') else enc_container.encoder
        
        if hasattr(encoder, 'model'):
            # TerraMindEncoder wraps the actual model
            s2_model = encoder.model
            
            # Add S1GRD bands to the bands dictionary
            s2s1_bands = encoder.bands.copy() if encoder.bands is not None else {}
            s2s1_bands['S1GRD'] = ['VV', 'VH']
            print(f"Updated bands for S2+S1: {s2s1_bands}")
            
            # Build new S2+S1 encoder with same parameters
            s2s1_encoder = TerraMindEncoder(
                model_name=encoder.model_name,
                pretrained=False,
                modalities=['S2L2A', 'S1GRD'],
                img_size=encoder.img_size,
                patch_size=encoder.patch_size,
                bands=s2s1_bands,
                for_cls=False,
                out_idx=encoder.out_idx,
            )
            s2s1_encoder = s2s1_encoder.to(args.device)
            
            # Adapt weights from S2-only to S2+S1
            adapt_terramind_s2_to_s2s1(s2_model, s2s1_encoder.model)
            
            # Replace encoder in model
            if hasattr(enc_container, 'encoder_non_siam'):
                enc_container.encoder_non_siam = s2s1_encoder
            else:
                enc_container.encoder = s2s1_encoder
            print("TerraMind encoder adapted from S2-only to S2+S1 for SAR evaluation.")
        else:
            print("Warning: Could not find TerraMind model structure for S2->S2+S1 adaptation")
    
    model.eval()
    model.to(args.device)
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
    
            cm_path = os.path.join('/nfs/ap/mnt/frtn/rs-multiband/OSCD/', city_name, 'cm/cm.png')    
            cm = Image.open(cm_path).convert('L')


            # if args.metadata_path:
            #     with open(f"{args.metadata_path}/{city_name}.json", 'r') as file:
            #         metadata = json.load(file)
            #         metadata.update({'waves': [3.5, 4.0, 0]})
            #         if args.replace_rgb_with_others:
            #             metadata.update({'waves': [0.49, 0.56, 0]})
            # else:
            #     metadata = None

            metadata = {}
            metadata.update({'waves': [3.5, 4.0, 0]})
            limits = product(range(0, img1.width, args.size), range(0, img1.height, args.size))
            for l in limits:
                limit = (l[0], l[1], l[0] + args.size, l[1] + args.size)
                sample1 = np.array(img1.crop(limit))
                sample2 = np.array(img2.crop(limit))
                mask = np.array(cm.crop(limit)) / 255


                # if ('cvit' not in cfg['backbone'].lower() and 
                #     'prithvi' not in cfg['backbone'].lower() and
                #     'dofa' not in cfg['backbone'].lower() and 
                #     'croma' not in cfg['backbone'].lower() and 
                #     'anysat' not in cfg['backbone'].lower() and 
                #     'satlas' not in cfg['backbone'].lower() and 
                #     'terrafm' not in cfg['backbone'].lower() and 
                #     'ibot' not in cfg['backbone'].lower() and 
                #     'resnet' not in cfg['backbone'].lower() and 
                #     'vit' not in cfg['backbone'].lower() and 
                #     'satlas' not in cfg['backbone'].lower() and 
                #     'dino' not in cfg['backbone'].lower()):
                #     zero_image = np.zeros((192, 192, 3))
                #     zero_image[:,:, 0] = sample1[:,:, 0]
                #     zero_image[:,:, 1] = sample1[:,:, 1]
                #     sample1 = zero_image
                    
                    
                    
                if 'satlas' in cfg['encoder_weights'].lower():
                    zero_image = np.zeros((224, 224, 9))
                    zero_image[:,:, 0] = sample1[:,:, 0]
                    zero_image[:,:, 1] = sample1[:,:, 1]
                    sample1 = zero_image
                    
                    zero_image = np.zeros((224, 224, 9))
                    zero_image[:,:, 0] = sample2[:,:, 0]
                    zero_image[:,:, 1] = sample2[:,:, 1]
                    sample2 = zero_image

                if 'anysat' in cfg['encoder_weights'].lower() or 'croma' in cfg['encoder_weights'].lower():
                    zero_image = np.zeros((120, 120, 3))
                    zero_image[:,:, 0] = sample1[:,:, 0]
                    zero_image[:,:, 1] = sample1[:,:, 1]
                    sample1 = zero_image
                    
                    zero_image = np.zeros((120, 120, 3))
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


                if 'dino' in cfg['backbone'].lower() or 'resnet' in cfg['backbone'].lower() or 'vit' in cfg['backbone'].lower():
                    zero_image = np.zeros((224, 224, 3))
                    zero_image[:,:, 0] = sample1[:,:, 0]
                    zero_image[:,:, 1] = sample1[:,:, 1]
                    sample1 = zero_image
    
                    zero_image = np.zeros((224, 224, 3))
                    zero_image[:,:, 0] = sample2[:,:, 0]
                    zero_image[:,:, 1] = sample2[:,:, 1]
                    sample2 = zero_image

                if 'dofa' in cfg['backbone'].lower() or 'terrafm' in cfg['backbone'].lower():
                    zero_image = np.zeros((224, 224, 12))
                    zero_image[:,:, 0] = sample1[:,:, 0]
                    zero_image[:,:, 1] = sample1[:,:, 1]
                    sample1 = zero_image
    
                    zero_image = np.zeros((224, 224, 12))
                    zero_image[:,:, 0] = sample2[:,:, 0]
                    zero_image[:,:, 1] = sample2[:,:, 1]
                    sample2 = zero_image

                if 'cvit' in cfg['backbone'].lower():
                    if args.replace_rgb_with_others:
                        zero_image = np.zeros((224, 224, 2))
                        zero_image[:,:, 0] = sample1[:,:, 0]
                        zero_image[:,:, 1] = sample1[:,:, 1]
                        sample1 = zero_image
        
                        zero_image = np.zeros((224, 224, 2))
                        zero_image[:,:, 0] = sample2[:,:, 0]
                        zero_image[:,:, 1] = sample2[:,:, 1]
                        sample2 = zero_image
                    
                    else:
                        zero_image = np.zeros((224, 224, 4))
                        zero_image[:,:, 0] = sample1[:,:, 0]
                        zero_image[:,:, 1] = sample1[:,:, 0]
                        zero_image[:,:, 2] = sample1[:,:, 1]
                        zero_image[:,:, 3] = sample1[:,:, 1]
                        sample1 = zero_image
        
                        zero_image = np.zeros((224, 224, 4))
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
                    out = model(sample1, sample2, [metadata])
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
    with open(f"{args.filename}.txt", "a") as log_file:
        log_file.write(f"{args.checkpoint_path}" +"\n" + f"{(fscores/samples)*100}" + "\n")

def eval_on_s2_sar(args, encoder_bands=None):
    test_cities = '/nfs/ap/mnt/frtn/rs-multiband/OSCD/test.txt'
    with open(test_cities) as f:
        test_set = f.readline()
    test_set = test_set[:-1].split(',')
    save_directory = f'./eval_outs/{args.checkpoint_path.split("/")[-2]}'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with open(args.model_config) as config:
        cfg = json.load(config)
    
    # Load model with the same configuration as training
    model = load_model(args.checkpoint_path, encoder_depth=cfg['encoder_depth'], backbone=cfg['backbone'], 
                       encoder_weights=cfg['encoder_weights'], fusion=cfg['fusion'], upsampling=args.upsampling, out_size=args.size,
                       load_decoder=cfg['load_decoder'], channels=args.cvit_channels, in_channels=cfg['in_channels'], 
                       upernet_width=args.upernet_width, enable_multiband=args.enable_multiband_input, 
                       multiband_channel_count=args.multiband_channel_count, bands_param=encoder_bands, strict_loading=args.strict_loading)
    
    # Optional TerraMind RGB -> multiband adaptation for S2+SAR evaluation
    backbone_name = cfg.get('backbone', '').lower()
    is_terramind = 'terramind' in backbone_name
    if (
        is_terramind
        and args.enable_multiband_input
        and args.multiband_channel_count > 3
        and args.adapt_terramind_rgb_to_multiband
    ):
        print("Adapting TerraMind encoder for multiband input in S2+SAR evaluation (RGB -> multiband)...")
        # load_model returns DDP(model); unwrap to access encoder
        enc_container = model.module if hasattr(model, 'module') else model
        
        # Adapt main encoder
        encoder = enc_container.encoder
        rgb_enc_sd = encoder.state_dict()
        encoder = replace_terramind_projection_layer(
            encoder,
            num_bands=args.multiband_channel_count,
            patch_size=16,
            embed_dim=getattr(encoder, "embed_dim", 768),
        )
        adapted_sd = adapt_terramind_state_dict(
            rgb_enc_sd,
            encoder,
            patch_size=16,
        )
        encoder.load_state_dict(adapted_sd, strict=False)
        enc_container.encoder = encoder
        
        # If non-siam encoder exists, adapt it too
        if hasattr(enc_container, 'encoder_non_siam'):
            encoder_non_siam = enc_container.encoder_non_siam
            rgb_enc_non_siam_sd = encoder_non_siam.state_dict()
            encoder_non_siam = replace_terramind_projection_layer(
                encoder_non_siam,
                num_bands=args.multiband_channel_count,
                patch_size=16,
                embed_dim=getattr(encoder_non_siam, "embed_dim", 768),
            )
            adapted_non_siam_sd = adapt_terramind_state_dict(
                rgb_enc_non_siam_sd,
                encoder_non_siam,
                patch_size=16,
            )
            encoder_non_siam.load_state_dict(adapted_non_siam_sd, strict=False)
            enc_container.encoder_non_siam = encoder_non_siam

        print("TerraMind encoder adapted for multiband S2+SAR evaluation.")

    # Optional TerraMind S2-only -> S2+S1 adaptation for S2+SAR evaluation
    if (
        is_terramind
        and args.adapt_terramind_s2_to_s2s1
    ):
        print("Adapting TerraMind encoder from S2-only to S2+S1 for S2+SAR evaluation...")
        from change_detection_pytorch.encoders.terramind import TerraMindEncoder
        
        # load_model returns DDP(model); unwrap to access encoder
        enc_container = model.module if hasattr(model, 'module') else model
        encoder = enc_container.encoder_non_siam if hasattr(enc_container, 'encoder_non_siam') else enc_container.encoder
        
        if hasattr(encoder, 'model'):
            # TerraMindEncoder wraps the actual model
            s2_model = encoder.model
            
            # Add S1GRD bands to the bands dictionary
            s2s1_bands = encoder.bands.copy() if encoder.bands is not None else {}
            s2s1_bands['S1GRD'] = ['VV', 'VH']
            print(f"Updated bands for S2+S1: {s2s1_bands}")
            
            # Build new S2+S1 encoder with same parameters
            s2s1_encoder = TerraMindEncoder(
                model_name=encoder.model_name,
                pretrained=False,
                modalities=['S2L2A', 'S1GRD'],
                img_size=encoder.img_size,
                patch_size=encoder.patch_size,
                bands=s2s1_bands,
                for_cls=False,
                out_idx=encoder.out_idx,
            )
            s2s1_encoder = s2s1_encoder.to(args.device)
            
            # Adapt weights from S2-only to S2+S1
            adapt_terramind_s2_to_s2s1(s2_model, s2s1_encoder.model)
            
            # Replace encoder in model
            if hasattr(enc_container, 'encoder_non_siam'):
                enc_container.encoder_non_siam = s2s1_encoder
            else:
                enc_container.encoder = s2s1_encoder
            print("TerraMind encoder adapted from S2-only to S2+S1 for S2+SAR evaluation.")
        else:
            print("Warning: Could not find TerraMind model structure for S2->S2+S1 adaptation")
    
    model.eval()
    model.to(args.device)
    fscore = cdp.utils.metrics.Fscore(activation='argmax2d')

    samples = 0
    fscores = 0
    for place in tqdm(glob("/nfs/ap/mnt/frtn/rs-multiband/oscd/multisensor_fusion_CD/S1/*")):
        city_name = place.split('/')[-1]
        if city_name in test_set:
            # Load S2 data (RGB bands)
            path1 = glob(f"{place}/imgs_1/transformed/*")[0]
            img1 = get_image_array(path1, return_rgb=True)  # RGB bands
    
            # Load SAR data (VV, VH bands)
            path2 = glob(f"{place}/imgs_2/transformed/*")[0]
            img2 = get_image_array(path2)  # VV, VH bands
    
            # Load change mask
            cm_path = os.path.join('/nfs/ap/mnt/frtn/rs-multiband/OSCD/', city_name, 'cm/cm.png')    
            cm = Image.open(cm_path).convert('L')

            # Load metadata if available
            # if args.metadata_path:
            #     with open(f"{args.metadata_path}/{city_name}.json", 'r') as file:
            #         metadata = json.load(file)
            #         metadata.update({'waves': [3.5, 4.0, 0]})
            #         if args.replace_rgb_with_others:
            #             metadata.update({'waves': [0.49, 0.56, 0]})
            # else:
            #     metadata = None
            metadata = {}
            metadata.update({'waves': [3.5, 4.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]})

            # Process the data the same way as training
            limits = product(range(0, img1.width, args.size), range(0, img1.height, args.size))
            for l in limits:
                limit = (l[0], l[1], l[0] + args.size, l[1] + args.size)
                sample1 = np.array(img1.crop(limit))  # RGB
                sample2 = np.array(img2.crop(limit))  # VV, VH
                mask = np.array(cm.crop(limit)) / 255

                # Pad both images to 12 channels (same as training)
                if sample1.shape[2] < 12:  # RGB has 3 channels
                    x1_zeros = torch.zeros((sample1.shape[0], sample1.shape[1], 12 - sample1.shape[2]), dtype=torch.float32)
                    sample1 = np.concatenate([sample1, x1_zeros.numpy()], axis=2)
                
                if sample2.shape[2] < 12:  # VV, VH has 2 channels
                    x2_zeros = torch.zeros((sample2.shape[0], sample2.shape[1], 12 - sample2.shape[2]), dtype=torch.float32)
                    sample2 = np.concatenate([sample2, x2_zeros.numpy()], axis=2)

                # Convert to tensors and move to GPU
                sample1 = torch.tensor(sample1.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0).cuda()
                sample2 = torch.tensor(sample2.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0).cuda()
                mask = torch.tensor(mask.astype(np.float32)).unsqueeze(0).cuda()

                # Run inference
                with torch.no_grad():
                    out = model(sample1, sample2, [metadata])
                
                # Calculate metrics
                samples += 1
                fs = fscore(out, mask)
                fscores += fs
               
    print(f"Total samples: {samples}")
    results = {}
    results[args.checkpoint_path] = {}
    results[args.checkpoint_path]['S2_SAR'] = {
                'micro_f1': fscores/samples}
        
    savefile = f'{save_directory}/results_s2_sar.npy'
    np.save(savefile, results)

    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"F1 Score: {(fscores/samples)*100:.2f}%")
    
    with open(f"{args.filename}.txt", "a") as log_file:
        log_file.write(f"{args.checkpoint_path}\n")
        log_file.write(f"S2+SAR F1: {(fscores/samples)*100:.2f}%\n")

def main(args):
    init_dist(args.master_port)
    
    bands = json.loads(args.bands)
    
    with open(args.model_config) as config:
        cfg = json.load(config)
    
    # Always use RGB bands for TerraMind
    encoder_bands = parse_encoder_bands_arg(args.encoder_bands, backbone=cfg.get('backbone', ''))

    if args.sar:
        eval_on_sar(args, encoder_bands)  # SAR only
    elif args.s2_sar:
        eval_on_s2_sar(args, encoder_bands)  # S2 + SAR
    else:
        results = {}
        
        with open(args.dataset_config) as config:
            data_cfg = json.load(config)

        print(f"Loading model from checkpoint: {args.checkpoint_path}")
        print("$$$$$$$$$$$$$$$$$$$$: ", args.adapt_terramind_s2_to_s2s1, cfg['backbone'])
        
        model = load_model(
            checkpoint_path=args.checkpoint_path,
            encoder_depth=cfg['encoder_depth'],
            backbone=cfg['backbone'],
            encoder_weights=cfg['encoder_weights'],
            fusion=cfg['fusion'],
            out_size=args.size,
            upernet_width=args.upernet_width,
            load_decoder=cfg['load_decoder'],
            in_channels=cfg['in_channels'],
            upsampling=args.upsampling,
            channels=args.cvit_channels,
            enable_multiband=args.enable_multiband_input,
            multiband_channel_count=args.multiband_channel_count,
            bands_param=encoder_bands,
            strict_loading=args.strict_loading,
        )

        # Optional TerraMind RGB -> multiband adaptation for change detection
        backbone_name = cfg.get('backbone', '').lower()
        is_terramind = 'terramind' in backbone_name
        print("$$$$$$$$$$$$$$$$$$$$: ", is_terramind, args.adapt_terramind_s2_to_s2s1)
        if (
            is_terramind
            and args.enable_multiband_input
            and args.multiband_channel_count > 3
            and args.adapt_terramind_rgb_to_multiband
        ):
            print("Adapting TerraMind encoder for multiband input in change detection (RGB -> multiband)...")
            # load_model returns DDP(model); unwrap to access encoder
            enc_container = model.module if hasattr(model, 'module') else model
            
            # Adapt main encoder
            encoder = enc_container.encoder
            rgb_enc_sd = encoder.state_dict()
            encoder = replace_terramind_projection_layer(
                encoder,
                num_bands=args.multiband_channel_count,
                patch_size=16,
                embed_dim=getattr(encoder, "embed_dim", 768),
            )
            adapted_sd = adapt_terramind_state_dict(
                rgb_enc_sd,
                encoder,
                patch_size=16,
            )
            encoder.load_state_dict(adapted_sd, strict=False)
            enc_container.encoder = encoder
            
            # If non-siam encoder exists, adapt it too
            if hasattr(enc_container, 'encoder_non_siam'):
                encoder_non_siam = enc_container.encoder_non_siam
                rgb_enc_non_siam_sd = encoder_non_siam.state_dict()
                encoder_non_siam = replace_terramind_projection_layer(
                    encoder_non_siam,
                    num_bands=args.multiband_channel_count,
                    patch_size=16,
                    embed_dim=getattr(encoder_non_siam, "embed_dim", 768),
                )
                adapted_non_siam_sd = adapt_terramind_state_dict(
                    rgb_enc_non_siam_sd,
                    encoder_non_siam,
                    patch_size=16,
                )
                encoder_non_siam.load_state_dict(adapted_non_siam_sd, strict=False)
                enc_container.encoder_non_siam = encoder_non_siam

            print("TerraMind encoder adapted for multiband change detection.")

        # Optional TerraMind S2-only -> S2+S1 adaptation for change detection
        if (
            is_terramind
            and args.adapt_terramind_s2_to_s2s1
        ):
            print("Adapting TerraMind encoder from S2-only to S2+S1 for change detection...")
            from change_detection_pytorch.encoders.terramind import TerraMindEncoder
            
            # load_model returns DDP(model); unwrap to access encoder
            enc_container = model.module if hasattr(model, 'module') else model
            encoder = enc_container.encoder_non_siam if hasattr(enc_container, 'encoder_non_siam') else enc_container.encoder
            
            if hasattr(encoder, 'model'):
                # TerraMindEncoder wraps the actual model
                s2_model = encoder.model
                
                # Add S1GRD bands to the bands dictionary
                s2s1_bands = encoder.bands.copy() if encoder.bands is not None else {}
                s2s1_bands['S1GRD'] = ['VV', 'VH']
                print(f"Updated bands for S2+S1: {s2s1_bands}")
                
                # Build new S2+S1 encoder with same parameters
                s2s1_encoder = TerraMindEncoder(
                    model_name=encoder.model_name,
                    pretrained=False,
                    modalities=['S2L2A', 'S1GRD'],
                    img_size=encoder.img_size,
                    patch_size=encoder.patch_size,
                    bands=s2s1_bands,
                    for_cls=False,
                    out_idx=encoder.out_idx,
                )
                s2s1_encoder = s2s1_encoder.to(args.device)
                
                # Adapt weights from S2-only to S2+S1
                adapt_terramind_s2_to_s2s1(s2_model, s2s1_encoder.model)
                
                # Replace encoder in model
                if hasattr(enc_container, 'encoder_non_siam'):
                    enc_container.encoder_non_siam = s2s1_encoder
                else:
                    enc_container.encoder = s2s1_encoder
                print("TerraMind encoder adapted from S2-only to S2+S1 for change detection.")
            else:
                print("Warning: Could not find TerraMind model structure for S2->S2+S1 adaptation")

        model.eval()
        model.to(args.device)
        print(f"Model loaded. Checkpoint path key: {args.checkpoint_path}")
        dataset_path = data_cfg['dataset_path']
        dataset_name = data_cfg['dataset_name']
        metadata_dir = data_cfg['metadata_dir']
        batch_size = data_cfg['batch_size']
        fill_zeros = cfg['fill_zeros']
        tile_size = args.size

        loss = cdp.utils.losses.CrossEntropyLoss()
        if args.use_dice_bce_loss:
            loss = cdp.utils.losses.dice_bce_loss()

        # DEVICE = 'cuda:{}'.format(dist.get_rank()) if torch.cuda.is_available() else 'cpu'
        results[args.checkpoint_path] = {}
        print(f"Initializing results for checkpoint: {args.checkpoint_path}")

        for band in bands :            
            if 'cvit' in model.module.encoder_name.lower():
                print('band1: ', band)
                get_indicies = []
                for b in band:
                    get_indicies.append(channel_vit_order.index(b))

                if args.replace_rgb_with_others:
                    get_indicies = [0, 1, 2]
                print('band2: ', band)

                model.module.channels = get_indicies
                
                
            if 'clay' in model.module.encoder_name.lower():
                for b in band:
                    if '_' in b:
                        first_band, second_band = b.split('_')
                        band[band.index(b)] = second_band

            if 'oscd' in dataset_name.lower():
                dataset_path = "/nfs/ap/mnt/frtn/rs-multiband/oscd/multisensor_fusion_CD/S1"
                datamodule = ChangeDetectionDataModule(dataset_path, metadata_dir, patch_size=tile_size, bands=band, 
                                                        fill_zeros=fill_zeros, batch_size=batch_size, 
                                                        replace_rgb_with_others=args.replace_rgb_with_others)
                datamodule.setup()
                valid_loader = datamodule.test_dataloader()

            elif 'harvey' in dataset_name.lower():
                print("band: ", band)
                rgb_bands = get_band_orders(model_name=cfg['backbone'], rgb=True)
                rgb_mapping = {'B02': 'B2', 'B03': 'B3', 'B04': 'B4'}
                rgb_bands = [rgb_mapping[b] for b in rgb_bands]

                test_dataset = FloodDataset(
                    # split_list=f"{dataset_path}/test.txt",
                    split_list=f"/nfs/h100/raid/rs/harvey_new_test.txt",
                    bands=band,
                    rgb_bands=rgb_bands,
                    fill_zeros=args.fill_zeros,
                    img_size=args.size)
                
                custom_collate_fn = create_collate_fn('change_detection')
                
                valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

            evaluator = SegEvaluator(
                # val_loader=test_loader,
                val_loader=valid_loader,
                exp_dir='',
                device=args.device,
                inference_mode="whole",  # or "whole", as needed
                sliding_inference_batch=batch_size,  # if using sliding mode
            )

            metrics, _ = evaluator(model, model_name="seg_model")
            if 'oscd' in dataset_name.lower():
                metric = metrics['F1_change']
            else:
                metric = metrics['IoU'][1]
   
            print(f'metrics: {metrics}')
            
            # Compute bIoU for Harvey dataset
            biou = None
            if 'harvey' in dataset_name.lower():
                model.eval()
                all_preds = []
                all_targets = []
                
                with torch.no_grad():
                    for data in tqdm(valid_loader, desc="Computing bIoU"):
                        image1, image2, target, _, metadata = data
                        image1 = image1.to(args.device)
                        image2 = image2.to(args.device)
                        target = target.to(args.device)
                        
                        logits = model(image1, image2, metadata=metadata)
                        if logits.shape[1] == 1:
                            pred = (torch.sigmoid(logits) > 0.5).type(torch.int64).squeeze(dim=1)
                        else:
                            pred = torch.argmax(logits, dim=1)
                        
                        # Store predictions and targets for class 1 (change/building class)
                        for p, t in zip(pred.cpu().numpy(), target.cpu().numpy()):
                            pred_class1 = (p == 1).astype(np.uint8)
                            gt_class1 = (t == 1).astype(np.uint8)
                            all_preds.append(pred_class1)
                            all_targets.append(gt_class1)
                
                if len(all_preds) > 0:
                    biou = compute_binary_iou(all_preds, all_targets)
                    print(f'bIoU: {biou:.2f}')
            
            if 'oscd' in dataset_name.lower():
                print(''.join(band), band, '########################')
                results[args.checkpoint_path][''.join(band)] = {
                    'f1_change': metric
                }
            else:
                result_dict = {
                    'iou_score': metric
                }
                if biou is not None:
                    result_dict['biou'] = biou
                results[args.checkpoint_path][''.join(band)] = result_dict
            with open(f"{args.filename}.txt", "a") as log_file:
                log_file.write(args.checkpoint_path)
                log_file.write(f"{band}" + "  " + f"{metric :.2f}" + "\n")
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

    # bands = [['B04', 'B03', 'B02'], ['B04', 'B03', 'B05'], ['B04', 'B05', 'B06'], ['B8A', 'B11', 'B12']]
    channel_vit_order = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', 'VV', 'VH'] #VVr VVi VHr VHi 
    # channel_vit_order = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A',  'B11', 'B12', 'vh', 'vv'] #VVr VVi VHr VHi
    all_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A','B11', 'B12','vv', 'vh']

    parser = ArgumentParser()
    parser.add_argument('--model_config', type=str, default='')
    parser.add_argument('--dataset_config', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--metadata_path', type=str, default='')
    parser.add_argument('--cvit_channels', nargs='+', type=int, default= [0, 1, 2])
    parser.add_argument('--sar', action="store_true")
    parser.add_argument('--s2_sar', action="store_true")  # Add this new argument

    parser.add_argument('--replace_rgb_with_others', action="store_true")
    parser.add_argument('--size', type=int, default=96)
    parser.add_argument('--upsampling', type=float, default=4)
    parser.add_argument('--master_port', type=str, default="12345")
    parser.add_argument('--use_dice_bce_loss', action="store_true")
    # parser.add_argument("--bands", type=str, default=json.dumps([['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'vh', 'vv'], ['B2', 'B3', 'B4' ], ['B5', 'B3','B4'], ['B5', 'B6', 'B4'], ['B8A', 'B11', 'B12']]))

    # parser.add_argument("--bands", type=str, default=json.dumps([['B2', 'B3', 'B4'], [ 'B5','B3','B4'], ['B6', 'B5', 'B4'], ['B8A', 'B11', 'B12'], ['vh', 'vv']]))
    parser.add_argument("--bands", type=str, default=json.dumps([['B02', 'B03', 'B04' ], ['B05', 'B03','B04'], ['B05', 'B06', 'B04'], ['B8A', 'B11', 'B12']]))
    parser.add_argument('--encoder_bands', type=str, default=None, help="Bands to pass to TerraMind encoders (JSON list or space-separated).")
    parser.add_argument('--filename', type=str, default='eval_bands_cd_log')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--upernet_width', type=int, default=256)
    parser.add_argument('--fill_zeros', action="store_true")
    parser.add_argument('--enable_multiband_input', action="store_true")
    parser.add_argument('--multiband_channel_count', type=int, default=12)
    parser.add_argument('--strict_loading', action='store_true', help='Use strict=True when loading checkpoint (fail on mismatches)')
    parser.add_argument(
        '--adapt_terramind_rgb_to_multiband',
        action='store_true',
        help='Adapt TerraMind RGB encoder to multiband (e.g., RGBN) for change detection eval.',
    )
    parser.add_argument(
        '--adapt_terramind_s2_to_s2s1',
        action='store_true',
        help='Adapt TerraMind S2-only encoder to S2+S1 (multimodal) for change detection eval.',
    )


    args = parser.parse_args()

    main(args)
