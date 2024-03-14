import torch
import wandb
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import LEVIR_CD_Dataset
from torch.utils.data import DataLoader
# from change_detection_pytorch.utils.lr_scheduler import GradualWarmupScheduler

from change_detection_pytorch.datasets import ChangeDetectionDataModule
from argparse import ArgumentParser

def setup_distributed_environment():
    # Initialize the distributed environment.
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)

def prepare_model_for_ddp(model):
    # Prepare the model for DDP.
    model = model.to(args.device)
    model = DDP(model, device_ids=[args.rank], output_device=args.rank)
    return model

def main(args):
    checkpoints_dir = f'./checkpoints/{args.experiment_name}'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)


    wandb.init(
        project="change_detection",
        name=args.experiment_name,
        config=vars(args)
    )
    setup_distributed_environment()

    DEVICE = f'cuda:{args.rank}' if torch.cuda.is_available() else 'cpu'
    print('Running on', DEVICE)

    #DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    #print('running on', DEVICE)
    model = cdp.UPerNet(
        encoder_depth=12,
        encoder_name=args.backbone, # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=args.encoder_weights, # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3, # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2, # model output channels (number of classes in your datasets)
        siam_encoder=True, # whether to use a siamese encoder
        fusion_form=args.fusion, # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
    )

    model = prepare_model_for_ddp(model)
    
    if 'oscd' in args.dataset_name.lower():
        datamodule = ChangeDetectionDataModule(args.dataset_path, patch_size=args.tile_size, mode=args.mode, batch_size=args.batch_size, scale=None)
        datamodule.setup()

        train_loader = datamodule.train_dataloader()
        valid_loader = datamodule.val_dataloader()
        print('train_loader', len(train_loader), 'val_loader', len(valid_loader))
    else:
        train_folder = f'{args.dataset_path}/train_large' if args.train_type == 'aug' else f'{args.dataset_path}/train'
        train_dataset = LEVIR_CD_Dataset(train_folder,
                                        sub_dir_1='A',
                                        sub_dir_2='B',
                                        img_suffix=args.img_suffix,
                                        ann_dir=f'{train_folder}/OUT',
                                        debug=False,
                                        seg_map_suffix=args.img_suffix)

        valid_dataset = LEVIR_CD_Dataset(f'{args.dataset_path}/val',
                                        sub_dir_1='A',
                                        sub_dir_2='B',
                                        img_suffix=args.img_suffix,
                                        ann_dir=f'{args.dataset_path}/val/OUT',
                                        debug=False,
                                        seg_map_suffix=args.img_suffix)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    loss = cdp.utils.losses.CrossEntropyLoss()
    metrics = [
        # torchmetrics.Precision(num_classes=1, threshold=0.5, task='binary'),
        # torchmetrics.Recall(num_classes=1, threshold=0.5, task='binary'),
        # torchmetrics.F1Score(num_classes=1, threshold=0.5, task='binary')

        cdp.utils.metrics.Fscore(activation='argmax2d'),
        cdp.utils.metrics.Precision(activation='argmax2d'),
        cdp.utils.metrics.Recall(activation='argmax2d'),
    ]
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
    ])
    if args.lr_sched == 'exponential':
        scheduler_steplr = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    elif args.lr_sched == 'multistep':
        scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, ], gamma=0.1)
    elif args.lr_sched == 'poly':
        scheduler_steplr = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.max_epochs, power=1)

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = cdp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
        grad_accum=args.grad_accum
    )

    valid_epoch = cdp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # train model for 60 epochs

    max_score = 0
    MAX_EPOCH = args.max_epochs

    for i in range(MAX_EPOCH):
        print('\nEpoch: {}'.format(i))
        wandb.log({"lr": optimizer.param_groups[0]['lr']})

        train_logs = train_epoch.run(train_loader)
        # wandb.log({"fscore_train": train_logs['BinaryF1Score'], 'loss_train': train_logs['cross_entropy_loss']})
        # wandb.log({"precision_train": train_logs['BinaryPrecision'], 'recall_train': train_logs['BinaryRecall']})
        wandb.log({"fscore_train": train_logs['Fscore'], 'loss_train': train_logs['cross_entropy_loss']})
        wandb.log({"precision_train": train_logs['Precision'], 'recall_train': train_logs['Recall']})


        valid_logs = valid_epoch.run(valid_loader)
        # wandb.log({"fscore_val": valid_logs['BinaryF1Score'], 'loss_val': valid_logs['cross_entropy_loss']})
        # wandb.log({"precision_val": valid_logs['BinaryPrecision'], 'recall_val[]': valid_logs['BinaryRecall']})
        wandb.log({"fscore_val": valid_logs['Fscore'], 'loss_val': valid_logs['cross_entropy_loss']})
        wandb.log({"precision_val": valid_logs['Precision'], 'recall_val': valid_logs['Recall']})

        scheduler_steplr.step()

        # do something (save model, change lr, etc.)
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

    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--mode', type=str, default='vanilla')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=70)
    parser.add_argument('--tile_size', type=int, default=192)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_sched', type=str, default='exponential')
    parser.add_argument('--fusion', type=str, default='diff')
    parser.add_argument('--train_type', type=str, default='')
    parser.add_argument('--img_suffix', type=str, default='.jpg')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--grad_accum', type=int, default=1)


    args = parser.parse_args()

    main(args)
