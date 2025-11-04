# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import datetime
import time
import random
import math
import json
import numpy as np
import utils
import models
from collections import defaultdict
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from pathlib import Path
from torchvision import models as torchvision_models
from tensorboardX import SummaryWriter
from models.head import iBOTHead
from torch.profiler import profile, record_function, ProfilerActivity

from dataset.augmentation import MAIDAugmentation, MAIDAugmentationCO
from dataset.hdf5_augmentation import HDF5Augmentation
from dataset import MAIDDataset, MAIDDatasetCO, HDF5Dataset, HDF5DatasetCO
from dataset.loader import MyDistributedBatchSampler, ContDistBatchSampler, ContDistSyncBatchSampler
from dataset.satlas_datasets import NaipDataset, Sen1Dataset, Sen2Dataset
from dataset.lmdb_dataset import LMDBDataset

#from evaluation.unsupervised.unsup_cls import eval_pred
from models.decoder import UPerNet


def get_args_parser():
    parser = argparse.ArgumentParser('iBOT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        # choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'deit_tiny', 'deit_small',
        #          'swin_tiny','swin_small', 'swin_base', 'swin_large'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--patch_out_dim', default=8192, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--shared_head', default=False, type=utils.bool_flag, help="""Wether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, help="""See above.
        Only works for teacher model. (Defeault: True)""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
        help="Whether to use batch normalizations in projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--pred_ratio', default=0.3, type=float, nargs='+', help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
    parser.add_argument('--pred_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred_shape', default='block', type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for dino
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""")
    parser.add_argument('--lambda3', default=1.0, type=float, help="""loss weight overlap loss (Default: 1.0)""")
        
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_sample_iters', default=2e6, type=utils.sci_int,
        help='Number of warmup iterations for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--accumulation_steps', default=1, type=int, 
                        help='Number of batches to accumulate gradients over before performing an optimization step.')
    parser.add_argument('--sample_iters', default=100e6, type=utils.sci_int, help='Number of iterations of training datasets.')
    parser.add_argument('--freeze_last_layer', default=6e6, type=utils.sci_int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_sample_iters", default=1e6, type=utils.sci_int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--only_decay', default=False, type=utils.bool_flag, help="""Whether to decay in wsd scenario.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--drop_path', type=float, default=0.1, help="""Drop path rate for student network.""")
    parser.add_argument('--saving_freq', default=5e6, type=utils.sci_int, help="""Number of iterations to save the checkpoint.""")
    parser.add_argument('--saving_sample_iters', nargs='+', type=utils.sci_int, default=[10e6, 20e6, 30e6, 40e6], help="""Number of iterations to save the checkpoint for decaying.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--compile_loss', default=False, type=utils.bool_flag, help="""Attempt to compile
        the loss.""")
    parser.add_argument('--compile_decoder', default=False, type=utils.bool_flag, help="""Attempt to compile
        the Decoder.""")
    parser.add_argument('--decoder_compile_mode', default="default", type=str)
    parser.add_argument('--use_overlap', default=False, type=utils.bool_flag)
    parser.add_argument('--sampling_subset', default=False, type=utils.bool_flag)
    parser.add_argument('--add_ch_embed', default=True, type=utils.bool_flag)
    parser.add_argument('--shared_proj', default=True, type=utils.bool_flag)
    parser.add_argument('--sync_channels', default=False, type=utils.bool_flag, help="""sync datasets and number of sampling channels between GPUs""")

    return parser

compiler_options = {
    "triton.cudagraphs": True,
    # "precision": torch.float16,
    # "min_block_size": 1,
    # "require_full_compilation": True,
    # "trace.graph_diagram": True,
    # "debug": True,
    # "trace.enabled": True,
    "verbose_progress": True,
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_ibot(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)

    # Broadcast log_dir from rank 0 to all other processes
    object_list = [args.log_dir]
    dist.broadcast_object_list(object_list, src=0)
    args.log_dir = object_list[0]
    
    print(f"\n Logging into {args.log_dir}\n")
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    # if args.use_overlap:
    #     maid_data_class = MAIDDatasetCO
    #     maid_transform_class = MAIDAugmentationCO
    #     hdf5_data_class = HDF5DatasetCO 
    #     transform_class = HDF5AugmentationCO

    maid_data_class = MAIDDataset
    maid_transform_class = MAIDAugmentation
    hdf5_data_class = HDF5Dataset 
    transform_class = HDF5Augmentation
    
    maid_transform = maid_transform_class(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
    )
    transform = transform_class(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
    )
    so2sat_transform = transform_class(
        [0.42, 1.0],
        [0.20, 0.42],
        args.global_crops_number,
        args.local_crops_number,
    )
    
    pred_size = args.patch_size
    datasets = {}
    datasets['MillionAID'] = maid_data_class(
                "/nfs/h100/raid/rs/maid", 
                transform=maid_transform,
                patch_size=pred_size,
                pred_ratio=args.pred_ratio,
                pred_ratio_var=args.pred_ratio_var,
                pred_aspect_ratio=(0.3, 1/0.3),
                pred_shape=args.pred_shape,
                pred_start_epoch=args.pred_start_epoch)
    datasets['BEN'] = hdf5_data_class(
                "/nfs/h100/raid/rs/rs-multiband/BEN_complete.h5", 
                data_key="BEN",
                transform=transform,
                patch_size=pred_size,
                pred_ratio=args.pred_ratio,
                pred_ratio_var=args.pred_ratio_var,
                pred_aspect_ratio=(0.3, 1/0.3),
                pred_shape=args.pred_shape,
                pred_start_epoch=args.pred_start_epoch)
            # datasets['BEN'] = LMDBDataset(
            #             "/nfs/h100/raid/rs/rs-multiband/ben_lmdb", 
            #             transform=transform,
            #             data_shape=(120, 120, 12), 
            #             normalize=False,
            #             patch_size=pred_size,
            #             pred_ratio=args.pred_ratio,
            #             pred_ratio_var=args.pred_ratio_var,
            #             pred_aspect_ratio=(0.3, 1/0.3),
            #             pred_shape=args.pred_shape,
            #             pred_start_epoch=args.pred_start_epoch)
            # datasets['WHU'] = hdf5_data_class(
            #             "/nfs/h100/raid/rs/rs-multiband/whu_pretrain.h5", 
            #             data_key="train",
            #             transform=transform,
            #             patch_size=pred_size,
            #             pred_ratio=args.pred_ratio,
            #             pred_ratio_var=args.pred_ratio_var,
            #             pred_aspect_ratio=(0.3, 1/0.3),
            #             pred_shape=args.pred_shape,
            #             pred_start_epoch=args.pred_start_epoch)
    # datasets['So2Sat'] = hdf5_data_class(
    #             "/nfs/h100/raid/rs/rs-multiband/so2sat_pretrain.h5", 
    #             data_key="train",
    #             transform=so2sat_transform,
    #             patch_size=pred_size,
    #             pred_ratio=args.pred_ratio,
    #             pred_ratio_var=args.pred_ratio_var,
    #             pred_aspect_ratio=(0.3, 1/0.3),
    #             pred_shape=args.pred_shape,
    #             pred_start_epoch=args.pred_start_epoch)
    datasets['Intelinair'] = hdf5_data_class(
                "/nfs/h100/raid/rs/rs-multiband/intelinair.h5", 
                data_key="intelinair",
                transform=transform,
                patch_size=pred_size,
                pred_ratio=args.pred_ratio,
                pred_ratio_var=args.pred_ratio_var,
                pred_aspect_ratio=(0.3, 1/0.3),
                pred_shape=args.pred_shape,
                pred_start_epoch=args.pred_start_epoch)
            # datasets['Intelinair'] = LMDBDataset(
            #             "/nfs/h100/raid/rs/rs-multiband/intelinair_lmdb", 
            #             transform=transform,
            #             data_shape=(320, 320, 4), 
            #             normalize=False,
            #             patch_size=pred_size,
            #             pred_ratio=args.pred_ratio,
            #             pred_ratio_var=args.pred_ratio_var,
            #             pred_aspect_ratio=(0.3, 1/0.3),
            #             pred_shape=args.pred_shape,
            #             pred_start_epoch=args.pred_start_epoch)
    datasets['SEN12MS'] = hdf5_data_class(
                "/nfs/h100/raid/rs/rs-multiband/sen12ms.h5", 
                data_key="SEN12MS",
                transform=transform,
                patch_size=pred_size,
                pred_ratio=args.pred_ratio,
                pred_ratio_var=args.pred_ratio_var,
                pred_aspect_ratio=(0.3, 1/0.3),
                pred_shape=args.pred_shape,
                pred_start_epoch=args.pred_start_epoch)
            # datasets['SEN12MS'] = LMDBDataset(
            #             "/nfs/h100/raid/rs/rs-multiband/sen12ms_lmdb", 
            #             transform=transform,
            #             data_shape=(256, 256, 12), 
            #             normalize=False,
            #             patch_size=pred_size,
            #             pred_ratio=args.pred_ratio,
            #             pred_ratio_var=args.pred_ratio_var,
            #             pred_aspect_ratio=(0.3, 1/0.3),
            #             pred_shape=args.pred_shape,
            #             pred_start_epoch=args.pred_start_epoch)
    datasets['NAIP'] = NaipDataset(
                "/nfs/h100/raid/rs/satlas_dataset/naip",
                stats_dir="/nfs/h100/raid/rs/satlas_dataset/stats/naip_stats",
                transform=transform,
                patch_size=pred_size,
                pred_ratio=args.pred_ratio,
                pred_ratio_var=args.pred_ratio_var,
                pred_aspect_ratio=(0.3, 1/0.3),
                pred_shape=args.pred_shape,
                pred_start_epoch=args.pred_start_epoch)
    datasets['SEN1'] = Sen1Dataset(
                "/nfs/h100/raid/rs/satlas_dataset/sentinel1",
                stats_dir="/nfs/h100/raid/rs/satlas_dataset/stats/sentinel1_stats",
                transform=transform,
                patch_size=pred_size,
                pred_ratio=args.pred_ratio,
                pred_ratio_var=args.pred_ratio_var,
                pred_aspect_ratio=(0.3, 1/0.3),
                pred_shape=args.pred_shape,
                pred_start_epoch=args.pred_start_epoch)
    datasets['SEN2a'] = Sen2Dataset(
                "/nfs/h100/raid/rs/satlas_dataset/sentinel2a",
                stats_dir="/nfs/h100/raid/rs/satlas_dataset/stats/sen2a_stats",
                transform=transform,
                patch_size=pred_size,
                pred_ratio=args.pred_ratio,
                pred_ratio_var=args.pred_ratio_var,
                pred_aspect_ratio=(0.3, 1/0.3),
                pred_shape=args.pred_shape,
                pred_start_epoch=args.pred_start_epoch)
    datasets['SEN2b'] = Sen2Dataset(
                "/nfs/h100/raid/rs/satlas_dataset/sentinel2b",
                stats_dir="/nfs/h100/raid/rs/satlas_dataset/stats/sen2b_stats",
                transform=transform,
                patch_size=pred_size,
                pred_ratio=args.pred_ratio,
                pred_ratio_var=args.pred_ratio_var,
                pred_aspect_ratio=(0.3, 1/0.3),
                pred_shape=args.pred_shape,
                pred_start_epoch=args.pred_start_epoch)

    for name, dataset in datasets.items():
        print(f"{name} data loaded: there are {len(dataset)} images.")

    BatchSamplerClass = ContDistSyncBatchSampler if args.sync_channels else ContDistBatchSampler
    batch_sampler = BatchSamplerClass(datasets, 
                                    weights={   'MillionAID': 1,
                                                'BEN': 4,
                                                #'WHU': 0,
                                                # 'So2Sat': 4,
                                                'Intelinair': 4,
                                                'SEN12MS': 4,
                                                'NAIP': 1,
                                                'SEN1': 1,
                                                'SEN2a': 1,
                                                'SEN2b': 1
                                            }, 
                                    num_replicas=utils.get_world_size(),
                                    rank=utils.get_rank(),
                                    batch_size=args.batch_size_per_gpu,
                                    drop_last=True,
                                    shuffle=True)
    dataset = batch_sampler.final_dataset
    multi_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler = batch_sampler,
            # batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            # drop_last=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    epochs = int(np.ceil(args.sample_iters / len(dataset))) * 10
    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    if args.arch in models.__dict__.keys():
        student = models.__dict__[args.arch](
            patch_size=args.patch_size,
            in_chans=12,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
            return_feats=args.use_overlap,
            add_ch_embed=args.add_ch_embed,
            shared_proj=args.shared_proj,
        )
        teacher = models.__dict__[args.arch](
            patch_size=args.patch_size,
            in_chans=12,
            return_all_tokens=True,
            return_feats=args.use_overlap,
            add_ch_embed=args.add_ch_embed,
            shared_proj=args.shared_proj,
        )
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, iBOTHead(
        embed_dim,
        args.out_dim,
        patch_out_dim=args.patch_out_dim,
        norm=args.norm_in_head,
        act=args.act_in_head,
        norm_last_layer=args.norm_last_layer,
        shared_head=args.shared_head,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim, 
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )

    decoder = None
    if args.use_overlap:
        decoder = UPerNet(
                encoder_channels = (768, 768, 768, 768),
                encoder_depth = 12,
                psp_channels = 512,
                pyramid_channels = 256,
                segmentation_channels = 256,
                fusion_form = "concat",
                classes = 2,
                activation = None
            )
        decoder = decoder.cuda()
        if utils.has_batchnorms(decoder):
            decoder = nn.SyncBatchNorm.convert_sync_batchnorm(decoder)
            
        decoder = nn.parallel.DistributedDataParallel(decoder, device_ids=[args.gpu])#,  find_unused_parameters=True)
        if args.compile_decoder:
            print("Compiling the Decoder network.")
            decoder = torch.compile(decoder, mode=args.decoder_compile_mode)
            # decoder.compile(options=compiler_options)

    # print(count_parameters(decoder))

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")
    
    # ============ preparing loss ... ============
    niters = args.sample_iters // (args.batch_size_per_gpu * utils.get_world_size() * args.accumulation_steps)
    
    same_dim = args.shared_head or args.shared_head_teacher
    ibot_loss = iBOTLoss(
        out_dim=args.out_dim,
        patch_out_dim=args.out_dim if same_dim else args.patch_out_dim,
        ngcrops=args.global_crops_number,
        nlcrops=args.local_crops_number,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp2=args.warmup_teacher_patch_temp,
        teacher_temp2=args.teacher_patch_temp,
        warmup_teacher_temp_iters=args.warmup_teacher_temp_sample_iters // (args.batch_size_per_gpu * utils.get_world_size() * args.accumulation_steps),
        niters=niters,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3 if args.use_overlap else 0,
        mim_start_epoch=args.pred_start_epoch,
    ).cuda()

    if args.compile_loss:
        print("Compiling the Loss.")
        ibot_loss = torch.compile(ibot_loss)
        # ibot_loss.compile(options=compiler_options, dynamic=False)
    
    if utils.is_main_process(): # Tensorboard configuration
        local_runs = os.path.join(args.log_dir, 'tf_logs')
        writer = SummaryWriter(logdir=local_runs)
    else:
        writer = None
        
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.use_overlap:
        params_groups = params_groups + utils.get_params_groups(decoder)
    
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")
    
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.amp.GradScaler('cuda')

    # ============ init schedulers ... ============
    if args.only_decay:
        lr_schedule = utils.decay_scheduler(
            base_value=args.lr * (args.batch_size_per_gpu * utils.get_world_size() * args.accumulation_steps) / 256.,  # linear scaling rule
            final_value=args.min_lr,
            niters=niters,
        )
        wd_schedule = utils.decay_scheduler(
            base_value=args.weight_decay,
            final_value=args.weight_decay_end,
            niters=niters,
        )
        momentum_schedule = utils.decay_scheduler(
            base_value=args.momentum_teacher,
            final_value=1,
            niters=niters,
        )
    else:
        lr_schedule = utils.wsd_scheduler(
            base_value=args.lr * (args.batch_size_per_gpu * utils.get_world_size() * args.accumulation_steps) / 256.,  # linear scaling rule
            final_value=args.min_lr,
            niters=niters,
            warmup_iters=args.warmup_sample_iters // (args.batch_size_per_gpu * utils.get_world_size() * args.accumulation_steps)
        )
        wd_schedule = utils.wsd_scheduler(
            base_value=args.weight_decay,
            final_value=args.weight_decay_end,
            niters=niters,
            warmup_iters=0,
        )
        # momentum parameter is increased to 1. during training with a cosine schedule
        momentum_schedule = utils.wsd_scheduler(
            base_value=args.momentum_teacher, 
            final_value=1,
            niters=niters,
            warmup_iters=0,)
        
    # lr_schedule = utils.cosine_scheduler(
    #     args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
    #     args.min_lr,
    #     args.epochs, len(multi_dataloader),
    #     warmup_epochs=args.warmup_epochs,
    # )
    # wd_schedule = utils.cosine_scheduler(
    #     args.weight_decay,
    #     args.weight_decay_end,
    #     args.epochs, len(multi_dataloader),
    # )
    # # momentum parameter is increased to 1. during training with a cosine schedule
    # momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
    #                                         args.epochs, len(multi_dataloader))
                  
    print(f"Loss, optimizer and schedulers ready.")
    
    # ============ optionally resume training ... ============
    niter_per_epoch = len(multi_dataloader) // args.accumulation_steps
    to_restore = {"epoch": 0, 'it': 0}
    # to_restore = {"epoch": 0, 'it': 0, 'sub_batch_idx': 0}
    if args.load_from:
        utils.restart_from_checkpoint(
            args.load_from,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=ibot_loss,
        )
    start_epoch = to_restore["epoch"]
    start_iter = to_restore["it"]# % niter_per_epoch

    start_time = time.time()
    print("Starting iBOT training!", start_epoch, start_iter)
    for epoch in range(start_epoch, epochs):
        if epoch == start_epoch:
            shift_it = 0
            if args.load_from is not None and args.sample_iters >= 10e6:
                shift_it = 0
            sub_batch_idx = (start_iter + shift_it) * args.accumulation_steps
            multi_dataloader.batch_sampler.set_iteration(sub_batch_idx)
        else:
            multi_dataloader.batch_sampler.set_iteration(0)
        # multi_dataloader.batch_sampler.set_iteration(0)
        multi_dataloader.batch_sampler.set_epoch(epoch)
        # multi_dataloader.dataset.set_epoch(epoch)
        
        # ============ training one epoch of iBOT ... ============
        epoch_start_time = time.time()
        train_stats, start_iter = train_one_epoch(student, teacher, teacher_without_ddp, decoder, ibot_loss,
                                            multi_dataloader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                            epoch, fp16_scaler, writer, epochs, start_iter, niter_per_epoch, args)

        # Print profiling results
        epoch_total_time = time.time() - epoch_start_time
        epoch_total_time_str = str(datetime.timedelta(seconds=int(epoch_total_time)))
        print(f'Epoch {epoch} training time {epoch_total_time_str}')

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'decoder': decoder.state_dict() if args.use_overlap else None,
            'optimizer': optimizer.state_dict(),
            'it': start_iter,
            'epoch': epoch + 1,
            'args': args,
            'ibot_loss': ibot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        # utils.save_on_master(save_dict, os.path.join(args.log_dir, 'checkpoint.pth'))
        # if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
        #     utils.save_on_master(save_dict, os.path.join(args.log_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.log_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                for k, v in train_stats.items():
                    writer.add_scalar(f"{k}_epoch", v, epoch)
    
    save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'decoder': decoder.state_dict() if args.use_overlap else None,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'it': start_iter,
            'args': args,
            'ibot_loss': ibot_loss.state_dict(),
        } 
    utils.save_on_master(save_dict, os.path.join(args.log_dir, f'checkpoint_end.pth'))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, decoder, ibot_loss, 
                    data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, 
                    epoch, fp16_scaler, writer, epochs, start_iter, niter_per_epoch, args):

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_iter_logger = utils.MetricLogger()
    metric_iter_grad_logger = utils.MetricLogger()
    header = 'Epoch: [{}/{}]'.format(epoch, epochs)
    
    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]
    print('start iteration')
    stop_iteration = args.sample_iters // (args.batch_size_per_gpu * utils.get_world_size() * args.accumulation_steps)
    save_iters = [iter // (args.batch_size_per_gpu * utils.get_world_size() * args.accumulation_steps) for iter in args.saving_sample_iters]
    saving_freq = args.saving_freq // (args.batch_size_per_gpu * utils.get_world_size() * args.accumulation_steps)
    freeze_last_layer = args.freeze_last_layer // (args.batch_size_per_gpu * utils.get_world_size() * args.accumulation_steps)
    tensorboard_freq = 100
    
    optimizer.zero_grad()
    accum_loss_val = 0
    accum_all_loss_val = defaultdict(float)
    for sub_batch_idx, data in enumerate(metric_logger.log_every(data_loader, 100, header)):
    # data_loader_iterator = metric_logger.log_every(data_loader, 100, header)
    # first_data = next(data_loader_iterator)
    # for sub_batch_idx in range(stop_iteration * args.accumulation_steps):
    #     data = first_data
        # if sub_batch_idx >= (niter_per_epoch - start_iter) * args.accumulation_steps:
        #     break
        if decoder is None:
            images, masks, band_names, data_paths = data
            crop_overlap_label = None
        else:
            images, masks, band_names, crop_overlap_label, data_paths = data
        
        it = start_iter + sub_batch_idx // args.accumulation_steps # global training iteration
        if it >= stop_iteration: # end of training if iterations are over
            print(f"End of training: {it} >= {stop_iteration}")
            break
        
        # update weight decay and learning rate according to their schedule
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        band_names = [band_name[0] for band_name in band_names]
        band_indices = utils.get_band_indices(band_names)
        num_ch = len(band_indices)
        
        if args.sync_channels:
            # Synchronize global_num_ch across all ranks
            if utils.is_main_process():
                global_num_ch = random.randint(1, num_ch)
            else:
                global_num_ch = 0  # Temporary placeholder
            global_num_ch_tensor = torch.tensor(global_num_ch, dtype=torch.int, device='cuda')
            dist.broadcast(global_num_ch_tensor, src=0)
            global_num_ch = global_num_ch_tensor.item()
        else:
            global_num_ch = random.randint(1, num_ch)
        
        global_channels_idx = random.sample(range(num_ch), k=global_num_ch)
        global_channels = [band_indices[idx] for idx in global_channels_idx]
        
        if args.sync_channels:
            # Synchronize local_num_ch across all ranks
            if args.sampling_subset:
                if utils.is_main_process():
                    local_num_ch = random.randint(1, global_num_ch)
                else:
                    local_num_ch = 0  # Temporary placeholder
                local_num_ch_tensor = torch.tensor(local_num_ch, dtype=torch.int, device='cuda')
                dist.broadcast(local_num_ch_tensor, src=0)
                local_num_ch = local_num_ch_tensor.item()
                local_channels_idx = random.sample(global_channels_idx, k=local_num_ch)
            else:
                if utils.is_main_process():
                    local_num_ch = random.randint(1, num_ch)
                else:
                    local_num_ch = 0  # Temporary placeholder
                local_num_ch_tensor = torch.tensor(local_num_ch, dtype=torch.int, device='cuda')
                dist.broadcast(local_num_ch_tensor, src=0)
                local_num_ch = local_num_ch_tensor.item()
                local_channels_idx = random.sample(range(num_ch), k=local_num_ch)
        
        else:
            if args.sampling_subset:
                local_num_ch = random.randint(1, global_num_ch)
                local_channels_idx = random.sample(global_channels_idx, k=local_num_ch)   
            else:
                local_num_ch = random.randint(1, num_ch)
                local_channels_idx = random.sample(range(num_ch), k=local_num_ch)
            
        local_channels = [band_indices[idx] for idx in local_channels_idx]
        
        # move images to gpu
        global_images = [im[:, global_channels_idx, :, :].cuda(non_blocking=True) for im in images[:args.global_crops_number]]
        local_images = [im[:, local_channels_idx, :, :].cuda(non_blocking=True) for im in images[args.global_crops_number:]]
        
        masks = [msk.cuda(non_blocking=True) for msk in masks]
        if decoder is not None:
            crop_overlap_label = crop_overlap_label.cuda(non_blocking=True)
        
        # with torch.cuda.amp.autocast(fp16_scaler is not None):
        with torch.amp.autocast('cuda', enabled=fp16_scaler is not None):
                # get global views
                teacher_output, _ = teacher(global_images, global_channels)
                student_output, _ = student(global_images, global_channels, mask=masks[:args.global_crops_number])
                # except Exception as e:
                #     print("!!!! Dumping global_images and global_channels")
                #     torch.save(global_images, f'saved_files/global_images_{it}.pt')
                #     torch.save(global_channels, f'saved_files/global_channels_{it}.pt')
                #     raise e
                
                student.module.backbone.masked_im_modeling = False
                pred_overlap = None
                if decoder is not None:
                    _, feats1 = student(global_images[:1], global_channels, ret_feats=True)
                    _, feats2 = teacher(global_images[1:2], global_channels, ret_feats=True)
                    print("feats1.max:", [feat.max().item() for feat in feats1], 
                        " feats2.max:", [feat.max().item() for feat in feats2])
                    pred_overlap = decoder(feats1, feats2)
                    mm = pred_overlap.max()
                    print("psp_last_conv.1.running_mean:", decoder.module.psp_last_conv[1].running_mean.max().item(), 
                        " psp_last_conv.1.running_var:", decoder.module.psp_last_conv[1].running_var.max().item())

                    if torch.isnan(mm) or torch.isinf(mm):
                        print("Detected nan in pred_overlap", force=True)
                        print("!!!! Dumping feats1 and feats2 and decoder state_dict", force=True)
                        d = str(feats1[0].device)
                        if int(d.split(':')[-1]) == 0:
                            torch.save(feats1, f'saved_files/feats1_{it}_{d}.pt')
                            torch.save(feats2, f'saved_files/feats2_{it}_{d}.pt')
                            torch.save(decoder.state_dict(), f'saved_files/decoder_sd_{it}_{d}.pt')
                            torch.save(student.state_dict(), f'saved_files/student_sd_{it}_{d}.pt')
                            torch.save(images, f'saved_files/images_{it}_{d}.pt')
                            exit()
                        pred_overlap = torch.nan_to_num(pred_overlap, 0,0,0)
                    #print('pred_overlap.max', pred_overlap.max())
                
                student_local_cls = student(local_images, local_channels, ret_feats=False)[0][0] if len(images) > args.global_crops_number else None
                # except Exception as e:
                #     print("!!!! Dumping local_images and local_channels")
                #     torch.save(local_images, f'saved_files/local_images_{it}.pt')
                #     torch.save(local_channels, f'saved_files/local_channels_{it}.pt')
                #     raise e
                student.module.backbone.masked_im_modeling = args.use_masked_im_modeling
                all_loss = ibot_loss(student_output=student_output, teacher_output=teacher_output, student_local_cls=student_local_cls, student_mask=masks, 
                                    pred_ovelap=pred_overlap, crop_overlap_label=crop_overlap_label, it=it, num_ch=global_num_ch)
                loss = all_loss.pop('loss')

        if not math.isfinite(loss.item()):
            d = str(student_output[0].device)
            # if int(d.split(':')[-1]) < 4:
            print("!!!! Loss is inf or nan, dumping student_output, teacher_output, student_local_cls, masks", force=True)
            torch.save(student_output, f'saved_files/student_output_{it}_{d}.pt')
            torch.save(teacher_output, f'saved_files/teacher_output_{it}_{d}.pt')
            torch.save(student_local_cls, f'saved_files/student_local_cls_{it}_{d}.pt')
            torch.save(masks, f'saved_files/masks_{it}_{d}.pt')
            torch.save(student.state_dict(), f'saved_files/student_sd_{it}_{d}.pt')
            torch.save(teacher.state_dict(), f'saved_files/teacher_sd_{it}_{d}.pt')
            # torch.save(feats1, f'saved_files/feats1_{it}_{d}.pt')
            # torch.save(feats2, f'saved_files/feats2_{it}_{d}.pt')
            torch.save(images, f'saved_files/images_{it}_{d}.pt')
            torch.save(data_paths, f'saved_files/data_paths_{it}_{d}.pt')
            exit()

        loss = loss / args.accumulation_steps
        accum_loss_val += loss.item()
        for k, v in all_loss.items():
            accum_all_loss_val[k] += v.item() / args.accumulation_steps
        
        if fp16_scaler is None:
            loss.backward()
        else:
            fp16_scaler.scale(loss).backward()
            
         # Perform optimization step after accumulation_steps
        if (sub_batch_idx + 1) % args.accumulation_steps == 0:
            # Update schedules based on optimization steps
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[it]

            metric_iter_grad_logger.update(avg_grad_norm=utils.grad_norm(student))
            if (it + 1) % tensorboard_freq == 0:
                metric_iter_grad_logger.synchronize_between_processes()
                if utils.is_main_process(): 
                    for key, meter in metric_iter_grad_logger.meters.items():
                        print(f"{key}: {meter.global_avg}")
                        writer.add_scalar(key, meter.global_avg, it)
                metric_iter_grad_logger = utils.MetricLogger()
            
            if fp16_scaler is None:
                if args.clip_grad:
                    utils.clip_gradients_fast(student, args.clip_grad)
                    # for m in student.children():
                    #     torch.nn.utils.clip_grad_norm_(m.parameters(), args.clip_grad)
                    if decoder is not None:
                        # utils.clip_gradients_fast(decoder, args.clip_grad)
                        for m in decoder.children():
                            torch.nn.utils.clip_grad_norm_(m.parameters(), args.clip_grad)
                utils.cancel_gradients_last_layer(it, student, freeze_last_layer)
                optimizer.step()
            else:
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer)
                    utils.clip_gradients_fast(student, args.clip_grad)
                    # for m in student.children():
                    #     torch.nn.utils.clip_grad_norm_(m.parameters(), args.clip_grad)
                    if decoder is not None:
                        # utils.clip_gradients_fast(decoder, args.clip_grad)
                        for m in decoder.children():
                            torch.nn.utils.clip_grad_norm_(m.parameters(), args.clip_grad)
                utils.cancel_gradients_last_layer(it, student, freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[it]
                for param_q, param_k in zip(params_q, params_k):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            
            # log statistics
            probs1 = teacher_output[0].chunk(args.global_crops_number)
            probs2 = student_output[0].chunk(args.global_crops_number)
            pred1 = utils.concat_all_gather(probs1[0].max(dim=1)[1]) 
            pred2 = utils.concat_all_gather(probs2[1].max(dim=1)[1])
            acc = (pred1 == pred2).sum() / pred1.size(0)

            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss=accum_loss_val)
            metric_iter_logger.update(loss=accum_loss_val)
            for key, value in accum_all_loss_val.items():
                metric_logger.update(**{key: value})
                metric_iter_logger.update(**{key: value})
            metric_logger.update(acc=acc)
            metric_iter_logger.update(acc=acc)
            
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
            
            ### add part when one can save the intermediate checkpoints for fixed iterations
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'decoder': decoder.state_dict() if args.use_overlap else None,
                'optimizer': optimizer.state_dict(),
                'it': it + 1,
                'sub_batch_idx': sub_batch_idx + 1,
                'epoch': epoch,
                'args': args,
                'ibot_loss': ibot_loss.state_dict(),
            }
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            
            if (it + 1) in save_iters:
                print(f"Saving checkpoint (with save_iters) at iteration {it+1} ({(it+1) * args.batch_size_per_gpu * utils.get_world_size() * args.accumulation_steps} samples)")
                utils.save_on_master(save_dict, os.path.join(args.log_dir, f'checkpoint_decay_ep{epoch}_{it+1}.pth'))
            if (it + 1) % saving_freq == 0:
                print(f"Saving checkpoint (with saving_freq) at iteration {it+1} ({(it+1) * args.batch_size_per_gpu * utils.get_world_size() * args.accumulation_steps} samples)")
                utils.save_on_master(save_dict, os.path.join(args.log_dir, f'checkpoint.pth'))
            if (it + 1) % tensorboard_freq == 0:
                metric_iter_logger.synchronize_between_processes()
                if utils.is_main_process():
                    writer.add_scalar('acc', acc, it)
                    writer.add_scalar('loss', accum_loss_val, it)
                    for key, value in all_loss.items():
                        writer.add_scalar(key, value.item(), it)
                    writer.add_scalar('lr', optimizer.param_groups[0]["lr"], it)
                    writer.add_scalar('wd', optimizer.param_groups[0]["weight_decay"], it)
                    for key, meter in metric_iter_logger.meters.items():
                        writer.add_scalar(key, meter.global_avg, it)
                # Reset
                metric_iter_logger = utils.MetricLogger()
                
            # Reset accumulated loss and gradients
            accum_loss_val = 0
            accum_all_loss_val = defaultdict(float)
            optimizer.zero_grad()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return return_dict, it + 1


class iBOTLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_iters, niters, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0, lambda3=1.0, mim_start_epoch=0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.loss3 = nn.CrossEntropyLoss()
        # import pdb; pdb.set_trace()

        # we apply a warm-up for the teacher temperature because
        # a too high temperature makes the training unstable at the beginning
        teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_iters),
            np.ones(niters - warmup_teacher_temp_iters) * teacher_temp
        ))
        teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_iters),
            np.ones(niters - warmup_teacher_temp_iters) * teacher_temp2
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_iters),
            np.ones(niters - warmup_teacher_temp_iters - mim_start_epoch) * teacher_temp2
        ))

        self.register_buffer("teacher_temp_schedule", torch.from_numpy(teacher_temp_schedule))
        self.register_buffer("teacher_temp2_schedule", torch.from_numpy(teacher_temp2_schedule))

    def forward(self, student_output, teacher_output, student_local_cls, student_mask, pred_ovelap, crop_overlap_label, it, num_ch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # torch.compiler.cudagraph_mark_step_begin()
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output

        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)
        #
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[it]
        temp2 = self.teacher_temp2_schedule[it]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    mask = student_mask[v].flatten(-2, -1).repeat(1, num_ch)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1

        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        if self.lambda3 != 0:
            total_loss3 = self.loss3(pred_ovelap, crop_overlap_label.squeeze().long()) * self.lambda3
            total_loss = dict(cls=total_loss1, patch=total_loss2, overlap=total_loss3, loss=total_loss1 + total_loss2 + total_loss3)
        else:    
            total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2)
        self.update_center(teacher_cls, teacher_patch)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('iBOT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    current_month = datetime.datetime.now().strftime('%b_%Y')
    log_dir = os.path.join(args.output_dir, current_month, current_time)
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    
    # Save args as a JSON file in the log directory
    args_json_path = os.path.join(args.log_dir, 'args.json')
    with open(args_json_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    #torch.set_float32_matmul_precision("high")
    train_ibot(args)
