# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from .wave_dynamic_layer import Dynamic_MLP_OFA, Dynamic_MLP_Decoder
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
import numpy as np

import torch
import torch.nn as nn
import pdb
import math
from functools import reduce
import json

from timm.models.vision_transformer import PatchEmbed, Block

from copy import deepcopy
from .vision_transformer import MultiLevelNeck
from pretrainedmodels.models.torchvision_models import pretrained_settings


new_settings = {
    "Dofa": {
        "dofa": '/nfs/ap/mnt/sxtn/cd/dofa/DOFA_ViT_base_e100.pth'
    },
}

pretrained_settings = deepcopy(pretrained_settings)
for model_name, sources in new_settings.items():
    if model_name not in pretrained_settings:
        pretrained_settings[model_name] = {}

    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'num_classes': 45
        }


class OFAViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, drop_rate=0.,
                 embed_dim=1024, depth=24, num_heads=16, wv_planes=128, num_classes=45,
                 global_pool=True, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 for_cls=False, out_idx=None, out_channels=None):
        super().__init__()

        self.wv_planes = wv_planes
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        self.patch_embed = Dynamic_MLP_OFA(wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim)
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.for_cls = for_cls
        self.out_idx = out_idx
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        if not for_cls:
            self.neck = MultiLevelNeck(in_channels=[768, 768, 768, 768], out_channels=768, scales=[4, 2, 1, 0.5])

    def forward_features(self, x, wave_list):
        # embed patches
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist

        x, _ = self.patch_embed(x, self.waves)

        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # # apply Transformer blocks
        # for block in self.blocks:
        #     x = block(x)
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == len(self.blocks) - 1:
                x = self.norm(x)

            if i in self.out_idx:
                out = x[:, 1:]   # drop cls token

                # reshape
                img_side_length = int(np.sqrt(out.shape[1]))
                out = out.view(-1, img_side_length, img_side_length, self.embed_dim)

                # channels first
                out = out.permute(0, 3, 1, 2)

                outs.append(out)

        if self.for_cls:
            if self.global_pool:
                x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
                outcome = self.fc_norm(x)
            else:
                x = self.norm(x)
                outcome = x[:, 0]
            
            return outcome
        
        return self.neck(tuple(outs))

    def forward_head(self, x, pre_logits=False):
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, wave_list):
        x = self.forward_features(x, wave_list)
        # x = self.forward_head(x)
        return x


def vit_small_patch16(**kwargs):
    model = OFAViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = OFAViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = OFAViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = OFAViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


dofa_encoders = {
    "dofa": {
        "encoder": OFAViT,
        "pretrained_settings": pretrained_settings['Dofa'],
        "params": {
            # "ckpt_path": '/nfs/ap/mnt/sxtn/cd/dofa/DOFA_ViT_base_e100.pth',
            "depth": 12,
            "embed_dim": 768,
            "num_heads": 12,
            "patch_size": 16,
            "global_pool": False,
            "out_idx": (2, 5, 8, 11),
            "out_channels": (768, 768, 768, 768)
        }
    }
}