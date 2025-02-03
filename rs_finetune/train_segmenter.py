import torch
import torch.distributed as dist
import wandb
import os

import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import BuildingDataset
from torch.utils.data import DataLoader

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
    checkpoints_dir = f'./checkpoints/{args.experiment_name}'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)


    wandb.init(
        project="segmentation",
        name=args.experiment_name,
        config=vars(args)
    )
    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    print('running on', DEVICE)
    if args.decoder == 'unet':
        model = cdp.UnetSeg(
            encoder_depth=args.encoder_depth,
            scales = [8, 4, 2, 1],
            encoder_name=args.backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=args.encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=args.in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,  # model output channels (number of classes in your datasets)
            decoder_channels =(768, 768, 768, 768)
        )
    else:
        model = cdp.UPerNetSeg(
            encoder_depth=args.encoder_depth,
            encoder_name=args.backbone, # choose encoder, e.g. overlap_ibot-B, mobilenet_v2 or efficientnet-b7
            encoder_weights=args.encoder_weights, # use `imagenet` pre-trained weights for encoder initialization
            in_channels=args.in_channels, # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            decoder_psp_channels=512,
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=256,
            decoder_merge_policy="add",
            classes=2, # model output channels (number of classes in your datasets)
            activation=None,
            freeze_encoder=args.freeze_encoder,
            pretrained = args.load_decoder,
            upsampling=args.upsampling,
            channels=args.cvit_channels
        )
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

    train_dataset = BuildingDataset(split_list=f"{args.dataset_path}/train.txt", bands=args.bands, img_size=args.img_size)
    valid_dataset = BuildingDataset(split_list=f"{args.dataset_path}/val.txt", bands=args.bands, img_size=args.img_size)


    # Initialize dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    if args.loss_type == 'bce':
        loss = torch.nn.BCEWithLogitsLoss()
    elif args.loss_type == 'ce':
        loss = torch.nn.CrossEntropyLoss()
    elif args.loss_type == 'dice':
        loss = cdp.utils.losses.DiceLoss()
    metrics = [
        cdp.utils.metrics.IoU(activation="identity"),
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    if args.lr_sched == 'warmup_cosine':
        def lr_lambda(current_step, warmup_steps, warmup_lr, end_lr):
            if current_step < warmup_steps:
                return warmup_lr + (1.0 - warmup_lr) * float((current_step + 1) / warmup_steps)
            else:
                return end_lr
            
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, args.warmup_steps, args.warmup_lr, args.lr))

        scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs - args.warmup_steps)
    
    elif args.lr_sched == 'multistep':
        scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*args.max_epochs), int(0.8*args.max_epochs)])
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
        train_logs = train_epoch.run_seg(train_loader)
        wandb.log({"IoU_train": train_logs['IoU'], 'loss_train': train_logs[type(loss).__name__], 
                    "lr": optimizer.param_groups[0]['lr']})

        valid_logs = valid_epoch.run_seg(valid_loader)
        wandb.log({"IoU_val": valid_logs['IoU'], 'loss_val': valid_logs[type(loss).__name__]})
        if args.lr_sched:
            if args.warmup_steps!=0 and (i+1) < args.warmup_steps and args.lr_sched == 'warmup_cosine':
                warmup_scheduler.step()
            else:
                scheduler_steplr.step()

        if max_score < valid_logs['IoU']:
            max_score = valid_logs['IoU']
            print('max_score', max_score)
            torch.save(model, f'{checkpoints_dir}/best_model.pth')
            print('Model saved!')
   
    torch.save(model, f'{checkpoints_dir}/last_model.pth')
            

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--backbone', type=str, default='')
    parser.add_argument('--encoder_weights', type=str, default='')
    parser.add_argument('--encoder_depth', type=int, default=12)

    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--metadata_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--load_from_checkpoint', action="store_true")
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--freeze_encoder', action="store_true")
    parser.add_argument('--load_decoder', action="store_true")
    parser.add_argument('--fill_zeros', action="store_true")
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--upsampling', type=float, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=96)
    parser.add_argument('--loss_type', type=str, default='bce')
    parser.add_argument('--lr_sched', type=str, default='')
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--warmup_lr', type=float, default=1e-6)
    parser.add_argument('--decoder', type=str, default='upernet')
    parser.add_argument("--cvit_channels", nargs='+', type=int, default= [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13])
    parser.add_argument("--bands", nargs='+', type=str, default= ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'VH', 'VH','VV', 'VV'])

    args = parser.parse_args()
    seed_torch(seed=args.seed)

    main(args)