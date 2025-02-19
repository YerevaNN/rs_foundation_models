import torch
import torch.distributed as dist
import wandb
import os

import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import LEVIR_CD_Dataset, FloodDataset
from torch.utils.data import DataLoader

from change_detection_pytorch.datasets import ChangeDetectionDataModule
from argparse import ArgumentParser
torch.set_float32_matmul_precision('medium')

import random
import numpy as np

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    checkpoints_dir = f'/nfs/h100/raid/rs/checkpoints_anna/checkpoints/OSCD/{args.experiment_name}'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)


    wandb.init(
        project="change_detection",
        name=args.experiment_name,
        config=vars(args)
    )
    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    print('running on', DEVICE)

    model = cdp.UPerNet(
        encoder_depth=args.encoder_depth,
        encoder_name=args.backbone, # choose encoder, e.g. overlap_ibot-B, mobilenet_v2 or efficientnet-b7
        encoder_weights=args.encoder_weights, # use `imagenet` pre-trained weights for encoder initialization
        in_channels=args.in_channels, # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        decoder_psp_channels=512,
        decoder_pyramid_channels=256,
        decoder_segmentation_channels=256,
        decoder_merge_policy="add",
        fusion_form=args.fusion, # Must be concat for Overlap
        classes=2, # model output channels (number of classes in your datasets)
        activation=None,
        siam_encoder=True, # whether to use a siamese encoder
        freeze_encoder=args.freeze_encoder,
        pretrained = args.load_decoder,
        upsampling=args.upsampling,
        channels=args.cvit_channels
    )
    if args.load_decoder:

        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(DEVICE))
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['decoder'].items()}
        unexpected_keys = {}
        for k, _ in state_dict.items():
            if 'segmentation_head' in k:
                unexpected_keys[k] = state_dict[k]
                # unexpected_keys.append(k)

        for key, _ in unexpected_keys.items():
            if key in state_dict:
                del state_dict[key]
        # unexpected_keys = {k.replace("segmentation_head.", ""): v for k, v in unexpected_keys.items()}

        # model.segmentation_head.load_state_dict(unexpected_keys)
        msg = model.decoder.load_state_dict(state_dict) 

        print('Decoder load with message', msg)

    if args.load_from_checkpoint:
        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(DEVICE))
        msg = model.load_state_dict(checkpoint.state_dict())
        print('Model load with message', msg)

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    if 'harvey' in args.dataset_name.lower():
        
        def custom_collate_fn(batch):
            images1, images2, labels, filename, metadata_list = zip(*batch)

            images1 = torch.stack(images1) 
            images2 = torch.stack(images2) 

            labels = torch.tensor(np.array(labels))
            metadata = list(metadata_list)

            return images1,  images2, labels, filename, metadata

        train_dataset = FloodDataset(
            split_list=f"{args.dataset_path}/train.txt",
            bands=args.bands,
            img_size=args.tile_size,
            is_train=True)

        valid_dataset = FloodDataset(
            split_list=f"{args.dataset_path}/val.txt",
            img_size=args.tile_size,
            bands=args.bands)

        # Initialize dataloader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn,)
        valid_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn,)

    elif 'oscd' in args.dataset_name.lower():
        datamodule = ChangeDetectionDataModule(args.dataset_path, args.metadata_path, patch_size=args.tile_size,
                                                bands=args.bands, mode=args.mode, batch_size=args.batch_size, 
                                                scale=None, fill_zeros=args.fill_zeros)
        datamodule.setup()

        train_loader = datamodule.train_dataloader()
        valid_loader = datamodule.val_dataloader()
        print('train_loader', len(train_loader), 'val_loader', len(valid_loader))
    else:
        if 'cdd' in args.dataset_name.lower():
            train_folder = f'{args.dataset_path}/train_large' if args.train_type == 'aug' else f'{args.dataset_path}/train'
        else:
            train_folder = f'{args.dataset_path}/train'
        train_dataset = LEVIR_CD_Dataset(train_folder,
                                        sub_dir_1=args.sub_dir_1,
                                        sub_dir_2=args.sub_dir_2,
                                        img_suffix=args.img_suffix,
                                        ann_dir=f'{train_folder}/{args.annot_dir}',
                                        debug=False,
                                        seg_map_suffix=args.img_suffix,
                                        size=args.crop_size,
                                        train_type=args.train_type)

        valid_dataset = LEVIR_CD_Dataset(f'{args.dataset_path}/val',
                                        sub_dir_1='A',
                                        sub_dir_2='B',
                                        img_suffix=args.img_suffix,
                                        ann_dir=f'{args.dataset_path}/val/OUT',
                                        debug=False,
                                        seg_map_suffix=args.img_suffix,
                                        size=args.crop_size)
        
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler)

        valid_sampler = torch.utils.data.DistributedSampler(valid_dataset, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=valid_sampler)
        
    loss = cdp.utils.losses.CrossEntropyLoss()
    loss_name = 'cross_entropy_loss'
    if args.use_dice_bce_loss:
        loss = cdp.utils.losses.dice_bce_loss()
        loss_name = 'dice_bce_loss'
    metrics = [
        cdp.utils.metrics.Fscore(activation='argmax2d'),
        cdp.utils.metrics.Precision(activation='argmax2d'),
        cdp.utils.metrics.Recall(activation='argmax2d'),
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {args.optimizer}")

    if args.lr_sched == 'exponential':
        scheduler_steplr = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    elif args.lr_sched == 'constant':
        scheduler_steplr = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.max_epochs)
    elif args.lr_sched == 'multistep':
        scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*args.max_epochs), int(0.8*args.max_epochs)])
    # elif args.lr_sched == 'multistep':
    #     scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, ], gamma=0.1)
    elif args.lr_sched == 'poly':
        scheduler_steplr = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.max_epochs, power=1)

    elif args.lr_sched == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler_steplr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif args.lr_sched == 'warmup_cosine':
        def lr_lambda(current_step, warmup_steps, warmup_lr, end_lr):
            if current_step < warmup_steps:
                return warmup_lr + (1.0 - warmup_lr) * float((current_step + 1) / warmup_steps)
            else:
                return end_lr
            
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, args.warmup_steps, args.warmup_lr, args.lr))

        scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs - args.warmup_steps)
    
    elif args.lr_sched == 'poly_warmup':
        def lr_lambda_poly(current_step, warmup_steps, warmup_lr, lr, power=1.0):

            if current_step < warmup_steps:
                return warmup_lr + (lr - warmup_lr) * float(current_step) / warmup_steps
            else:
                return max((1 - (current_step - warmup_steps) / (args.max_epochs - warmup_steps)) ** power, args.min_lr)

        scheduler_steplr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda_poly(step, args.warmup_steps, args.warmup_lr, args.lr))

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = cdp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
        grad_accum=args.grad_accum
    )

    valid_epoch = cdp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    # train model for 60 epochs

    max_score = 0
    MAX_EPOCH = args.max_epochs

    for i in range(MAX_EPOCH):
        print('\nEpoch: {}'.format(i))
        # train_loader.sampler.set_epoch(i)
        train_logs = train_epoch.run(train_loader)
        wandb.log({"fscore_train": train_logs['Fscore'], 'loss_train': train_logs[loss_name],
                    "precision_train": train_logs['Precision'], 'recall_train': train_logs['Recall'], 
                    "lr": optimizer.param_groups[0]['lr']})

        valid_logs = valid_epoch.run(valid_loader)
        wandb.log({"fscore_val": valid_logs['Fscore'], 'loss_val': valid_logs[loss_name]})

        wandb.log({"precision_val": valid_logs['Precision'], 'recall_val': valid_logs['Recall']})
        if args.warmup_steps!=0 and (i+1) < args.warmup_steps and args.lr_sched == 'warmup_cosine':
            warmup_scheduler.step()
        else:
            scheduler_steplr.step()

        if max_score < valid_logs['Fscore']:
            max_score = valid_logs['Fscore']
            print('max_score', max_score)
            torch.save(model, f'{checkpoints_dir}/best_model.pth')
            print('Model saved!')

    # save results (change maps)
    """
    Note: if you use sliding window inference, set: 
        from change_detection_pytorch.datasets.transforms.albu import (
            ChunkImage, ToTensorTest)
        
        test_transform = A.Compose([
            A.Normalize(),
            ChunkImage({window_size}}),
            ToTensorTest(),
        ], additional_targets={'image_2': 'image'})

    """
    valid_epoch.infer_vis(valid_loader, save=False, slide=False, save_dir='./res')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--backbone', type=str, default='')
    parser.add_argument('--encoder_weights', type=str, default='')
    parser.add_argument('--encoder_depth', type=int, default=12)

    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--metadata_path', type=str, default='')
    parser.add_argument('--mode', type=str, default='vanilla')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=70)
    parser.add_argument('--tile_size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr_sched', type=str, default='poly')
    parser.add_argument('--fusion', type=str, default='diff')
    parser.add_argument('--train_type', type=str, default='')
    parser.add_argument('--img_suffix', type=str, default='.jpg')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--load_from_checkpoint', action="store_true")
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--warmup_lr', type=float, default=1e-6)
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--sub_dir_1', type=str, default='A')
    parser.add_argument('--sub_dir_2', type=str, default='B')
    parser.add_argument('--annot_dir', type=str, default='OUT')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--freeze_encoder', action="store_true")
    parser.add_argument('--load_decoder', action="store_true")
    parser.add_argument('--fill_zeros', action="store_true")
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--upsampling', type=float, default=4)
    parser.add_argument('--use_dice_bce_loss', action="store_true")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--cvit_channels", nargs='+', type=int, default= [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13])
    parser.add_argument("--bands", nargs='+', type=str, default= ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'VH', 'VH','VV', 'VV'])


    args = parser.parse_args()
    seed_torch(seed=args.seed)


    main(args)
