import os
import torch
import rasterio
import json
import numpy as np
import change_detection_pytorch as cdp
# import torch.distributed as dist

from tqdm import tqdm
from osgeo import gdal
from itertools import product
from PIL import Image
from argparse import ArgumentParser
from sklearn import metrics
from glob import glob
from tqdm import tqdm
from eval_scale_cd import CustomMetric, load_model, init_dist
from change_detection_pytorch.datasets import ChangeDetectionDataModule, FloodDataset, normalize_channel, RGB_BANDS, STATS
from torch.utils.data import DataLoader
from evaluator_change import SegEvaluator



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

    if args.replace_rgb_with_others and 'cvit' in cfg['backbone'].lower():
        channels = [0, 1]

    model = load_model(args.checkpoint_path, encoder_depth=cfg['encoder_depth'], backbone=cfg['backbone'], 
                       encoder_weights=cfg['encoder_weights'], fusion=cfg['fusion'], upsampling=args.upsampling,
                       load_decoder=cfg['load_decoder'], channels=channels, in_channels=cfg['in_channels'])    
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
    
            cm_path = os.path.join('/nfs/ap/mnt/sxtn/aerial/change/OSCD/', city_name, 'cm/cm.png')    
            cm = Image.open(cm_path).convert('L')


            if args.metadata_path:
                with open(f"{args.metadata_path}/{city_name}.json", 'r') as file:
                    metadata = json.load(file)
                    metadata.update({'waves': [3.5, 4.0, 0]})
                    if args.replace_rgb_with_others:
                        metadata.update({'waves': [0.49, 0.56, 0]})
            else:
                metadata = None

            limits = product(range(0, img1.width, args.size), range(0, img1.height, args.size))
            for l in limits:
                limit = (l[0], l[1], l[0] + args.size, l[1] + args.size)
                sample1 = np.array(img1.crop(limit))
                sample2 = np.array(img2.crop(limit))
                mask = np.array(cm.crop(limit)) / 255


                if ('cvit' not in cfg['backbone'].lower() and 
                    'prithvi' not in cfg['backbone'].lower() and
                    'dofa' not in cfg['backbone'].lower() and 
                    'satlas' not in cfg['backbone'].lower() and 
                    'dino' not in cfg['backbone'].lower()):
                    zero_image = np.zeros((192, 192, 3))
                    zero_image[:,:, 0] = sample1[:,:, 0]
                    zero_image[:,:, 1] = sample1[:,:, 1]
                    sample1 = zero_image
                    
                    
                    
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


                if 'dino' in cfg['backbone'].lower():
                    zero_image = np.zeros((196, 196, 3))
                    zero_image[:,:, 0] = sample1[:,:, 0]
                    zero_image[:,:, 1] = sample1[:,:, 1]
                    sample1 = zero_image
    
                    zero_image = np.zeros((196, 196, 3))
                    zero_image[:,:, 0] = sample2[:,:, 0]
                    zero_image[:,:, 1] = sample2[:,:, 1]
                    sample2 = zero_image

                if 'dofa' in cfg['backbone'].lower():
                    zero_image = np.zeros((224, 224, 3))
                    zero_image[:,:, 0] = sample1[:,:, 0]
                    zero_image[:,:, 1] = sample1[:,:, 1]
                    sample1 = zero_image
    
                    zero_image = np.zeros((224, 224, 3))
                    zero_image[:,:, 0] = sample2[:,:, 0]
                    zero_image[:,:, 1] = sample2[:,:, 1]
                    sample2 = zero_image

                if 'cvit' in cfg['backbone'].lower():
                    if args.replace_rgb_with_others:
                        zero_image = np.zeros((192, 192, 2))
                        zero_image[:,:, 0] = sample1[:,:, 0]
                        zero_image[:,:, 1] = sample1[:,:, 1]
                        sample1 = zero_image
        
                        zero_image = np.zeros((192, 192, 2))
                        zero_image[:,:, 0] = sample2[:,:, 0]
                        zero_image[:,:, 1] = sample2[:,:, 1]
                        sample2 = zero_image
                    
                    else:
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

def main(args):
    init_dist(args.master_port)
    
    bands = json.loads(args.bands)

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
                       load_decoder=cfg['load_decoder'], in_channels=cfg['in_channels'], upsampling=args.upsampling)
        model.eval()
        model.to(args.device)
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
                datamodule = ChangeDetectionDataModule(dataset_path, metadata_dir, patch_size=tile_size, bands=band, 
                                                        fill_zeros=fill_zeros, batch_size=batch_size, 
                                                        replace_rgb_with_others=args.replace_rgb_with_others)
                datamodule.setup()
                valid_loader = datamodule.val_dataloader()

            elif 'harvey' in dataset_name.lower():
                print("band: ", band)
                test_dataset = FloodDataset(
                    split_list=f"{dataset_path}/test.txt",
                    bands=band,
                    img_size=args.size)
                
                def custom_collate_fn(batch):
                    images1, images2, labels, filename, metadata_list = zip(*batch)

                    images1 = torch.stack(images1) 
                    images2 = torch.stack(images2) 

                    labels = torch.tensor(np.array(labels))
                    metadata = list(metadata_list)

                    return images1,  images2, labels, filename, metadata
                
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
            print(f'metrics: {metrics}')
            with open(f"{args.filename}.txt", "a") as log_file:
                log_file.write(args.checkpoint_path)
                log_file.write(f"{band}" + "  " + f"{metrics['mF1'] * 100:.2f}" + "\n")
        save_directory = f'./eval_outs/{args.checkpoint_path.split("/")[-2]}'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        savefile = f'{save_directory}/results.npy'
        np.save(savefile, results)
  
            
if __name__== '__main__':

    # bands = [['B04', 'B03', 'B02'], ['B04', 'B03', 'B05'], ['B04', 'B05', 'B06'], ['B8A', 'B11', 'B12']]
    
    channel_vit_order = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A',  'B11', 'B12', 'vh', 'vv'] #VVr VVi VHr VHi
    all_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A','B11', 'B12','vv', 'vh']

    parser = ArgumentParser()
    parser.add_argument('--model_config', type=str, default='')
    parser.add_argument('--dataset_config', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--metadata_path', type=str, default='')
    parser.add_argument('--sar', action="store_true")

    parser.add_argument('--replace_rgb_with_others', action="store_true")
    parser.add_argument('--size', type=int, default=96)
    parser.add_argument('--upsampling', type=float, default=4)
    parser.add_argument('--master_port', type=str, default="12345")
    parser.add_argument('--use_dice_bce_loss', action="store_true")
    parser.add_argument("--bands", type=str, default=json.dumps([['B02', 'B03', 'B04' ], ['B05', 'B03','B04'], ['B05', 'B06', 'B04'], ['B8A', 'B11', 'B12']]))
    parser.add_argument('--filename', type=str, default='eval_bands_cd_log')
    parser.add_argument('--device', type=str, default='cuda:0')


    args = parser.parse_args()

    main(args)
