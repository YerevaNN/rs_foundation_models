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
from change_detection_pytorch.datasets import ChangeDetectionDataModule, FloodDataset, normalize_channel
from torch.utils.data import DataLoader
from evaluator_change import SegEvaluator
from utils import get_band_orders


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
    img_clipped = np.clip(img, 0.0, 1.0)
    img = (img_clipped * 255).astype(np.uint8)

    img = Image.fromarray(img)
        
    return img

def eval_on_sar(args):
    test_cities = '/nfs/ap/mnt/frtn/OSCD/test.txt'
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
                       encoder_weights=cfg['encoder_weights'], fusion=cfg['fusion'], upsampling=args.upsampling, out_size=120,
                       load_decoder=cfg['load_decoder'], channels=channels, in_channels=cfg['in_channels'], upernet_width=args.upernet_width)    
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
    
            cm_path = os.path.join('/nfs/ap/mnt/frtn/OSCD/', city_name, 'cm/cm.png')    
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
                    'croma' not in cfg['backbone'].lower() and 
                    'anysat' not in cfg['backbone'].lower() and 
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
                       encoder_weights=cfg['encoder_weights'], fusion=cfg['fusion'], out_size=args.size, upernet_width=args.upernet_width,
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
            if 'oscd' in dataset_name.lower():
                metric = metrics['F1_change']
            else:
                metric = metrics['IoU'][1]
   
            print(f'metrics: {metrics}')
            with open(f"{args.filename}.txt", "a") as log_file:
                log_file.write(args.checkpoint_path)
                log_file.write(f"{band}" + "  " + f"{metric :.2f}" + "\n")
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
    # parser.add_argument("--bands", type=str, default=json.dumps([['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'vh', 'vv'], ['B2', 'B3', 'B4' ], ['B5', 'B3','B4'], ['B5', 'B6', 'B4'], ['B8A', 'B11', 'B12']]))

    # parser.add_argument("--bands", type=str, default=json.dumps([['B2', 'B3', 'B4'], [ 'B5','B3','B4'], ['B6', 'B5', 'B4'], ['B8A', 'B11', 'B12'], ['vh', 'vv']]))
    parser.add_argument("--bands", type=str, default=json.dumps([['B02', 'B03', 'B04' ], ['B05', 'B03','B04'], ['B05', 'B06', 'B04'], ['B8A', 'B11', 'B12']]))
    parser.add_argument('--filename', type=str, default='eval_bands_cd_log')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--upernet_width', type=int, default=256)
    parser.add_argument('--fill_zeros', action="store_true")


    args = parser.parse_args()

    main(args)
