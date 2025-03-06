# Copyright (c) Insitro, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from functools import partial
from typing import List

import random
import torch
import torch.nn as nn

from .vision_transformer import Block
from .vision_transformer import trunc_normal_
from copy import deepcopy
from .vision_transformer import MultiLevelNeck
from pretrainedmodels.models.torchvision_models import pretrained_settings

new_settings = {
    "Cvit-B": {
        "ep20": "/nfs/dgx/raid/rs/rs/h100_channel_log_100ep_new/checkpoint0020.pth",
        "ep30": "/nfs/dgx/raid/rs/rs/h100_channel_log_100ep_new/checkpoint0030.pth",
        "ep40": "/nfs/dgx/raid/rs/rs/h100_channel_log_100ep_new/checkpoint0040.pth",
        "ep50": "/nfs/dgx/raid/rs/rs/h100_channel_log_100ep_new/checkpoint0050.pth",
        "ep60": "/nfs/dgx/raid/rs/rs/h100_channel_log_100ep_new/checkpoint0060.pth",
        "ep70": "/nfs/dgx/raid/rs/rs/h100_channel_log_100ep_new/checkpoint0070.pth",
        "ep80": "/nfs/dgx/raid/rs/rs/h100_channel_log_100ep_new/checkpoint0080.pth",
        "ep90": "/nfs/dgx/raid/rs/rs/h100_channel_log_100ep_new/checkpoint0090.pth",
        "ep100": "/nfs/dgx/raid/rs/rs/h100_channel_log_100ep_new/checkpoint.pth", # we don't need this one because of its high loss     
        "independent_sampled_5m": "/nfs/ap/mnt/frtn/rs-results/cvit_checkpoints_from_nebius/independent_sampling/checkpoint_5M.pth",
        "independent_sampled_10m": "/nfs/ap/mnt/frtn/rs-results/cvit_checkpoints_from_nebius/independent_sampling/checkpoint_10M.pth",
        "independent_sampled_20m": "/nfs/ap/mnt/frtn/rs-results/cvit_checkpoints_from_nebius/independent_sampling/checkpoint_20M.pth",
        "independent_sampled_40m": "/nfs/ap/mnt/frtn/rs-results/cvit_checkpoints_from_nebius/independent_sampling/checkpoint_40M.pth",
        "subset_sampled_5m": "/nfs/ap/mnt/frtn/rs-results/cvit_checkpoints_from_nebius/subset_sampling/checkpoint_5M.pth",
        "subset_sampled_10m": "/nfs/ap/mnt/frtn/rs-results/cvit_checkpoints_from_nebius/subset_sampling/checkpoint_10M.pth",
        "subset_sampled_20m": "/nfs/ap/mnt/frtn/rs-results/cvit_checkpoints_from_nebius/subset_sampling/checkpoint_20M.pth",
        "subset_sampled_40m": "/nfs/ap/mnt/frtn/rs-results/cvit_checkpoints_from_nebius/subset_sampling/checkpoint_40M.pth",


    }
}

pretrained_settings = deepcopy(pretrained_settings)
for model_name, sources in new_settings.items():
    if model_name not in pretrained_settings:
        pretrained_settings[model_name] = {}

    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }


class PatchEmbedPerChannel(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = False
    ):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )  # CHANGED

        self.channel_embed = nn.parameter.Parameter(
            torch.zeros(1, embed_dim, in_chans, 1, 1)
        )
        self.enable_sample = enable_sample
        print("enable_sample:", enable_sample)
        trunc_normal_(self.channel_embed, std=0.02)

    def forward(self, x, channel_idxs):
        B, Cin, H, W = x.shape
        # Note: The current number of channels (Cin) can be smaller or equal to in_chans
        if self.training and self.enable_sample:
            Cin_new = random.randint(1, Cin)

            # Randomly sample the selected channels
            channels = random.sample(range(Cin), k=Cin_new)
            Cin = Cin_new
            x = x[:, channels, :, :]
            # channel_idxs = channel_idxs[channels]
            channel_idxs = channels

        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1))  # B Cout Cin H W
        # channel specific offsets
        x += self.channel_embed[:, :, channel_idxs, :, :]  # B Cout Cin H W

        # # preparing the output sequence
        # x = x.flatten(2)  # B Cout CinHW
        # x = x.transpose(1, 2)  # B CinHW Cout
        return x


class SDChannelVisionTransformer(nn.Module):
    """Channel Vision Transformer"""

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        return_feats=False,
        enable_sample=False,
        **kwargs,
    ):
        super().__init__()
        self.return_feats = return_feats
        if return_feats:
            self.neck = MultiLevelNeck(in_channels=[embed_dim, embed_dim, embed_dim, embed_dim], 
                                    out_channels=embed_dim, 
                                    scales=[4, 2, 1, 0.5],
                                    norm_cfg=dict(type='BN')
                                    )
            self.out_channels = (768, 768, 768, 768)
            self.out_idx = (2, 5, 8, 11)
            self.feat_norms = nn.ModuleList([norm_layer(embed_dim) for _ in range(4)])
        self.num_features = self.embed_dim = self.out_dim = embed_dim
        self.in_chans = in_chans
        self.patch_embed = PatchEmbedPerChannel(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            enable_sample=enable_sample
        )
        num_patches = self.patch_embed.num_patches
        #self.neck = MultiLevelNeck(in_channels=[384, 384, 384, 384],out_channels=384, scales=[2, 1, 0.5, 0.25])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.num_extra_tokens = 1  # cls token

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, num_patches // self.in_chans + self.num_extra_tokens, embed_dim
            )
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h, c):
        # number of auxilary dimensions before the patches
        if not hasattr(self, "num_extra_tokens"):
            # backward compatibility
            num_extra_tokens = 1
        else:
            num_extra_tokens = self.num_extra_tokens

        npatch = x.shape[1] - num_extra_tokens
        N = self.pos_embed.shape[1] - num_extra_tokens

        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, :num_extra_tokens]
        patch_pos_embed = self.pos_embed[:, num_extra_tokens:]
        
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, 1, -1, dim)

        # create copies of the positional embeddings for each channel
        patch_pos_embed = patch_pos_embed.expand(1, c, -1, dim).reshape(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x, channel_idxs):
        B, nc, w, h = x.shape
        x = self.patch_embed(x, channel_idxs)  # B Cout Cin H W
        out_size = (x.shape[-2], x.shape[-1])
        Cin_new = x.shape[2]
        x = x.flatten(2).transpose(1, 2)
        
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional encoding to each token
        # x = x + self.interpolate_pos_encoding(x, w, h, nc)
        x = x + self.interpolate_pos_encoding(x, w, h, Cin_new)

        return self.pos_drop(x), out_size

    def forward(self, x, channel_idxs):
        feats = []
        x, hw_shape = self.prepare_tokens(x, channel_idxs)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.return_feats and (i in self.out_idx):
                norm_x = self.feat_norms[len(feats)](x)
                B, _, Cout = norm_x.shape
                feat = norm_x[:, 1:].reshape(B, -1, hw_shape[0], hw_shape[1],
                                             Cout).mean(dim=1).permute(0, 3, 1, 2).contiguous()
                feats.append(feat)

        x = self.norm(x)
                
        if self.return_feats:
            return self.neck(tuple(feats))

        return x[:, 0, :]

    def get_last_selfattention(self, x, channel_idxs):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, channel_idxs, n=1):
        x = self.prepare_tokens(x, channel_idxs)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def channelvit_tiny(patch_size=16, **kwargs):
    model = SDChannelVisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def channelvit_small(patch_size=16, **kwargs):
    model = SDChannelVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def channelvit_base(patch_size=16, **kwargs):
    model = SDChannelVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


sd_cvit_encoders = {
    "cvit-pretrained": {
        "encoder": SDChannelVisionTransformer,
        "pretrained_settings": pretrained_settings["Cvit-B"],
        "params": {
            "embed_dim": 768,
            "patch_size": 16,
            "in_chans": 12,
            # "enable_sample": True,
            "depth": 12, 
            "num_heads": 12, 
            "mlp_ratio": 4,
            "qkv_bias": True,
            "out_channels": (768, 768, 768, 768),
            "out_idx": (2, 5, 8, 11),
            }

        }
    }
