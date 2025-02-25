import math
import torch
import itertools
import copy
import numpy as np
import torch.nn as nn
import collections.abc
import rpe_index_cpp

from functools import partial
from itertools import repeat
from torch.jit import Final
from typing import Callable, Optional, Tuple, Union, List

class PatchMLP(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            scale: int = 1,
            resolution: float = 0.2,
            embed_dim: int = 768,
            patch_size: int = 10,
            bias: bool = True,
            mlp: List[int] = [],
            ):
        super().__init__()
        self.scale = scale
        self.res = int(10 / resolution)
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        layers = []
        for i in range(len(mlp) - 1):
            layers.extend(
                [
                    nn.Linear(mlp[i], mlp[i + 1]),
                    nn.LayerNorm(mlp[i + 1]),
                    nn.ReLU(),
                ]
            )
        self.mlp  = nn.Sequential(*layers)

    def forward(self, x):
        x = self.patch_embed(x)
        grid_size = max(self.res // self.patch_size, 1)
        x = x.unfold(2, grid_size, grid_size).unfold(3, grid_size, grid_size)
        x = x.flatten(4, 5)
        x = x.unfold(2, self.scale, self.scale).unfold(3, self.scale, self.scale)
        x = x.flatten(2, 3).permute(0, 1, 2, 4, 5, 3).flatten(3, 5)
        x = torch.permute(x,(0,2,3,1))
        x = x.flatten(0,1)
        x = self.mlp(x)
        return x

class PatchMLPMulti(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            resolution: float = 0.2,
            embed_dim: int = 768,
            patch_size: int = 10,
            bias: bool = True,
            mlp: List[int] = [],
            ):
        super().__init__()
        self.patch_size = patch_size
        self.res = int(10 / resolution)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        layers = []
        for i in range(len(mlp) - 1):
            layers.extend(
                [
                    nn.Linear(mlp[i], mlp[i + 1]),
                    nn.LayerNorm(mlp[i + 1]),
                    nn.ReLU(),
                ]
            )
        self.mlp  = nn.Sequential(*layers)

    def forward(self, x, scale):
        x = self.patch_embed(x)
        grid_size = (self.res // self.patch_size, self.res // self.patch_size)
        x = x.unfold(2, grid_size[0], grid_size[0]).unfold(3, grid_size[1], grid_size[1])
        x = x.flatten(4, 5)
        x = x.unfold(2, scale, scale).unfold(3, scale, scale)
        x = x.flatten(2, 3).permute(0, 1, 2, 4, 5, 3).flatten(3, 5)
        x = torch.permute(x,(0,2,3,1))
        x = x.flatten(0,1)
        x = self.mlp(x)
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table


class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        mlp_in = [10, 32, 128],
        T=1000,
        in_norm=True,
        return_att=False,
        return_att_full=False,
        positional_encoding=True,
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.return_att_full = return_att_full
        self.n_head = n_head

        if len(mlp_in) > 0:
            self.d_model = mlp_in[-1]
            mlp_in.insert(0, self.in_channels)
            layers = []
            for i in range(len(mlp_in) - 1):
                layers.extend(
                    [
                        nn.Linear(mlp_in[i], mlp_in[i + 1]),
                        #nn.BatchNorm1d(mlp_in[i + 1]),
                        nn.GroupNorm(4, mlp_in[i + 1]),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )

            self.inconv  = nn.Sequential(*layers)
        else:
            self.d_model = in_channels
            self.inconv = None

        self.mlp.insert(0, self.d_model)

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        
        if in_norm:
            self.in_norm = nn.GroupNorm(
                num_groups=n_head,
                num_channels=mlp_in[-1],
            )
        else:
            self.in_norm = nn.Identity()
        
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )
        
        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    #nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.GroupNorm(4, self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)

        if self.inconv is not None:
            out_reshape = out.view(-1, out.size(-1))
            out_reshape = self.inconv(out_reshape)
            out = out_reshape.view(out.shape[0], out.shape[1], -1)

        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = (
            out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        )  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w

        if self.return_att:
            return out, torch.mean(attn, dim=(0, 3, 4), keepdim=True).squeeze()
        elif self.return_att_full:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, d_k
        )  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn
  
    
class PatchLTAE(nn.Module):
    def __init__(
            self,
            in_channels=128,
            n_head=16,
            d_k=4,
            mlp=[256, 128],
            dropout=0.2,
            mlp_in = [10, 32, 128],
            T=1000,
            in_norm=True,
            return_att=False,
            positional_encoding=True,
            scale: int = 1,
            ):
        super().__init__()
        self.scale = scale
        self.patch_embed = LTAE2d(in_channels=in_channels, n_head=n_head, d_k=d_k, mlp=mlp, dropout=dropout, 
                                  mlp_in=mlp_in, T=T, in_norm=in_norm, return_att=return_att, 
                                  positional_encoding=positional_encoding)
        self.pad_parameter = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, x, dates, mask=None):
        if mask is not None:
            self.pad_parameter.requires_grad = True
            x = x.masked_fill(torch.logical_not(mask.bool()), self.pad_parameter)
        x = self.patch_embed(x, dates)
        B, E, _, _= x.shape
        x = x.unfold(2, self.scale, self.scale).unfold(3, self.scale, self.scale)
        x = x.flatten(2, 3).flatten(3, 4).permute(0, 2, 3, 1)
        x = x.flatten(0,1)
        return x
    
class PatchLTAEMulti(nn.Module):
    def __init__(
            self,
            in_channels=128,
            n_head=16,
            d_k=4,
            mlp=[256, 128],
            dropout=0.2,
            mlp_in = [10, 32, 128],
            T=1000,
            in_norm=True,
            return_att=False,
            positional_encoding=True,
            reduce_scale = 1,
            ):
        super().__init__()
        self.patch_embed = LTAE2d(in_channels=in_channels, n_head=n_head, d_k=d_k, mlp=mlp, dropout=dropout, 
                                  mlp_in=mlp_in, T=T, in_norm=in_norm, return_att=return_att, 
                                  positional_encoding=positional_encoding)
        self.reduce_scale = reduce_scale
        self.pad_parameter = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, x, dates, scale, mask=None):
        scale = scale // self.reduce_scale
        if scale == 0:
            scale = 1
        if mask is not None:
            self.pad_parameter.requires_grad = True
            x = x.masked_fill(torch.logical_not(mask.bool()), self.pad_parameter)
            self.pad_parameter.requires_grad = False
        x = self.patch_embed(x, dates)
        B, E, _, _= x.shape
        x = x.unfold(2, scale, scale).unfold(3, scale, scale)
        x = x.flatten(2, 3).flatten(3, 4).permute(0, 2, 3, 1)
        x = x.flatten(0,1)
        return x


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            use_flash_attn=True
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        if use_flash_attn:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
            except ImportError:
                raise ImportError("flash-attn is not installed. Please install it with `pip install flash-attn`")
        else:
            self.flash_attn_func = None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.flash_attn_func is not None:
            x = self.flash_attn_func(q, k, v, causal=False)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            flash_attn=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            use_flash_attn=flash_attn,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CrossRPEAttentionMulti(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches={}, modalities={}, scales={}, release=False,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.num_patches = num_patches
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_learned = nn.Parameter(torch.zeros(1, 1, dim))
        if not release: 
            self.datasets = list(modalities.keys())
            self.modis = {dataset: int("modis" in modalities[dataset]) for dataset in self.datasets}
            self.len_modalities = {}
            self.pos_embed = {}
            for dataset in self.datasets:
                self.len_modalities[dataset] = len(modalities[dataset]) - int("modis" in modalities[dataset])
                for scale in scales[dataset]:
                    num_p = num_patches[dataset] // (scale * scale)
                    self.pos_embed['_'.join([dataset, str(scale)])] = get_2d_sincos_pos_embed_with_scale(dim, int(num_p ** .5), 
                                                                                                scale, cls_token=True, modis=self.modis[dataset])

        # image relative position encoding
        rpe_config = get_rpe_config(
                ratio=1.9,
                method="euc",
                mode='ctx',
                shared_head=True,
                skip=1,
                rpe_on='k',
            )
        
        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config,
                      head_dim=head_dim,
                      num_heads=num_heads,
                      n_modalities=1)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None, dataset="", scale=1):
        B, N, C = x.shape
        pos_embed = self.pos_embed['_'.join([dataset, str(scale)])].to(x.device)
        modis = self.modis[dataset]
        # B1C -> B1H(C/H) -> BH1(C/H)
        if mask is None:
            num_patches = N // self.len_modalities[dataset] + int(N%self.len_modalities[dataset] > 0) + modis
            q_ = self.q_learned.expand(B, num_patches, -1) + pos_embed.expand(B, -1, -1)
        else:
            num_patches = mask.shape[-1] + 1 + modis
            mask_pos = mask.unsqueeze(-1).repeat(1, 1, pos_embed.shape[-1])
            pos_embed_e = pos_embed.expand(B, -1, -1)
            masked_pos_embed = torch.gather(pos_embed.expand(B, -1, -1)[:, 1:], dim=1, index=mask_pos)
            pos_embed = torch.cat([pos_embed_e[:, :(1 + modis)], masked_pos_embed], dim = 1)
            q_ = self.q_learned.expand(B, num_patches, -1) + pos_embed
        q = q_.reshape(B, num_patches, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N

        # image relative position on keys
        if self.rpe_k is not None:
            height = int((self.num_patches[dataset] ** 0.5) / scale)
            rpe = self.rpe_k(q, height=height, width=height, pos=mask, modis=modis)
            attn += torch.cat([rpe[:, :, :, :(1+ modis)], rpe[:, :, :, (1+ modis):].repeat(1, 1, 1, self.len_modalities[dataset])], dim=-1)
            
        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, num_patches, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = torch.cat([x[:, :1], x[:, (1 + modis):]], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def forward_release(self, x, mask=None, n_modalities=1, modis=False, scale=1):
        B, N, C = x.shape
        num_patches = N // n_modalities + int((N - int(modis)) % n_modalities > 0) + int(modis)
        pos_embed = get_2d_sincos_pos_embed_with_scale(C, int(num_patches ** .5), scale, cls_token=True, modis=modis).to(x.device)
        # B1C -> B1H(C/H) -> BH1(C/H)
        if mask is None:
            q_ = self.q_learned.expand(B, num_patches, -1) + pos_embed.expand(B, -1, -1)
        else:
            num_patches = mask.shape[-1] + 1 + modis
            mask_pos = mask.unsqueeze(-1).repeat(1, 1, pos_embed.shape[-1])
            pos_embed_e = pos_embed.expand(B, -1, -1)
            masked_pos_embed = torch.gather(pos_embed.expand(B, -1, -1)[:, 1:], dim=1, index=mask_pos)
            pos_embed = torch.cat([pos_embed_e[:, :(1 + modis)], masked_pos_embed], dim = 1)
            q_ = self.q_learned.expand(B, num_patches, -1) + pos_embed
        q = q_.reshape(B, num_patches, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N

        # image relative position on keys
        if self.rpe_k is not None:
            height = int((num_patches ** 0.5))
            rpe = self.rpe_k(q, height=height, width=height, pos=mask, modis=modis)
            attn += torch.cat([rpe[:, :, :, :(1+ modis)], rpe[:, :, :, (1+ modis):].repeat(1, 1, 1, n_modalities)], dim=-1)
            
        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, num_patches, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = torch.cat([x[:, :1], x[:, (1 + modis):]], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossBlockMulti(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., release=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patches={}, modalities={}, scales={}):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossRPEAttentionMulti(dim, num_heads=num_heads, qkv_bias=qkv_bias, modalities=modalities, release=release,
                        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, num_patches=num_patches, scales=scales)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, dataset="", scale=1):
        x = self.drop_path(self.attn(self.norm1(x), mask=mask, dataset=dataset, scale=scale))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    def forward_release(self, x, n_modalities=1, modis=False, scale=1):
        x = self.drop_path(self.attn.forward_release(self.norm1(x), n_modalities=n_modalities, modis=modis, scale=scale))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """
    return_indices: torch.jit.Final[bool]

    def __init__(
            self,
            prob: float = 0.5,
            num_prefix_tokens: int = 1,
            ordered: bool = False,
            return_indices: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered
        self.return_indices = return_indices

    def forward(self, x) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if not self.training or self.prob == 0.:
            if self.return_indices:
                return x, None
            return x

        if self.num_prefix_tokens:
            prefix_tokens, x = x[:, :self.num_prefix_tokens], x[:, self.num_prefix_tokens:]
        else:
            prefix_tokens = None

        B = x.shape[0]
        L = x.shape[1]
        num_keep = max(1, int(L * (1. - self.prob)))
        keep_indices = torch.argsort(torch.randn(B, L, device=x.device), dim=-1)[:, :num_keep]
        if self.ordered:
            # NOTE does not need to maintain patch order in typical transformer use,
            # but possibly useful for debug / visualization
            keep_indices = keep_indices.sort(dim=-1)[0]
        x = x.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))

        if prefix_tokens is not None:
            x = torch.cat((prefix_tokens, x), dim=1)

        if self.return_indices:
            return x, keep_indices
        return x


def get_2d_sincos_pos_embed_with_scale(
    embed_dim, grid_size, scale, cls_token=False, modis=False
):
    """
    grid_size: int of the grid height and width
    res: array of size n, representing the resolution of a pixel (say, in meters),
    return:
    pos_embed: dict of [n,grid_size*grid_size, embed_dim] or [n,1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(
        grid_w, grid_h, indexing="xy"
    )  # here h goes first,direction reversed for numpy
    grid = torch.stack(grid, dim=0)  # 2 x h x w

    grid = torch.einsum("chw,n->cnhw", grid, torch.tensor([scale])) 
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(
        embed_dim, grid
    )  #  # (nxH*W, D/2)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        pos_embed = torch.cat(
            [
                torch.zeros(
                    [n, 1, embed_dim], dtype=torch.float32, device=pos_embed.device
                ),
                pos_embed,
            ],
            dim=1,
        )
    if modis:
        pos_embed = torch.cat(
            [
                torch.zeros(
                    [n, 1, embed_dim], dtype=torch.float32, device=pos_embed.device
                ),
                pos_embed,
            ],
            dim=1,
        )
    return pos_embed

def get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    old_shape = pos
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class AnyModule(nn.Module):
    """
    Initializes AnySat encoding module.
    Args:
        spatial_encoder (nn.Module): Neural network module for spatial encoding
        projectors (dict): Dict of all possible projectors
        modalities (dict): Dict of modalities to use
        num_patches (dict): Dict of number of patches by observation for each modality
        embed_dim (int): Embed dimension of transformer blocks. Default: 768
        depth (int): Depth of transformer blocks. Default: 12
        num_heads (int): Number of heads of transformer blocks. Default: 12
        mlp_ratio (float): MLP ratio of transformer blocks. Default: 4.
        qkv_bias (bool): Whether to use bias in QKV projection. Default: True
        qk_scale: Scale factor for QK attention. Default: None
        class_token (bool): If True, add a class token. Default: True
        pre_norm (bool): Whether to apply normalization before transformer blocks. Default: False
        drop_rate (float): Dropout rate. Default: 0.
        patch_drop_rate (float): Patch dropout rate. Default: 0.
        drop_path_rate (float): Drop path rate for transformer blocks. Default: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        norm_layer (Optional[Callable]): Normalization layer. Default: None
        scales (dict): Dict of scales for each dataset
        keep_subpatch (bool): Whether to keep subpatch information. Default: False
        modality_keep (str): Which modality to keep subpatches for. Default: ""
        flash_attn (bool): Whether to use flash attention. Default: True
        release (bool): Whether to initialize hte model as the feature extractor. Default: False
    """
    def __init__(self,
                 spatial_encoder: nn.Module,
                 projectors: dict = {},
                 modalities: dict = {},
                 num_patches: dict = {},
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale = None,
                 class_token: bool = True,
                 pre_norm: bool = False,
                 drop_rate: float = 0.,
                 patch_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 norm_layer: Optional[Callable] = None,
                 scales: dict = {},
                 keep_subpatch: bool = False,
                 modality_keep: str = "",
                 flash_attn: bool = True,
                 release: bool = False,
                 ):
        
        super(AnyModule, self).__init__()
        self.modalities = modalities

        self.num_prefix_tokens = 1 if class_token else 0
        self.embed_dim = embed_dim
        self.keep_subpatch = keep_subpatch
        self.modality_keep = modality_keep

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        if not release:
            self.datasets = list(modalities.keys())
            self.pos_embed = {}
            for dataset in self.datasets:
                for scale in scales[dataset]:
                    num_p = num_patches[dataset] // (scale * scale)
                    self.pos_embed['_'.join([dataset, str(scale)])] = get_2d_sincos_pos_embed_with_scale(
                                                                        embed_dim, 
                                                                        int(num_p ** .5), 
                                                                        scale, 
                                                                        cls_token=class_token
                                                                    )
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()

        modalities_list = sorted(list(set(list(itertools.chain.from_iterable(modalities.values())))))
        for modality in modalities_list:
            if modality.split('-')[-1] == 'mono':
                m = '-'.join(modality.split('-')[:-1])
            else:
                m = modality
            setattr(self, '_'.join(['projector', modality]), projectors[m])

        self.spatial_encoder = spatial_encoder 

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth + 1)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop_rate, 
                drop_path=dpr[i], norm_layer=norm_layer, flash_attn=flash_attn) for i in range(depth)] + [CrossBlockMulti(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, modalities=modalities,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[-1], norm_layer=norm_layer, num_patches=num_patches,
                scales=scales, release=release)
                ])
        trunc_normal_(self.cls_token, std=.02)

    def forward_proj(self, x):
        """
        Forward function until masking used during pretraining
        """
        tokens = []
        masks = {}
        out = {}
        pos_embed = self.pos_embed['_'.join([x['dataset'], str(x['scale'])])].to(x['label'].device)
        _, N, _ = pos_embed.shape
        for modality in self.modalities[x['dataset']]:
            if modality == "aerial" or modality == "spot" or modality == "aerial-flair" or modality == "naip":
                token = getattr(self, '_'.join(['projector', modality]))(x[modality], x['scale'])
            else:
                if '_'.join([modality, "mask"]) in list(x.keys()):
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], 
                        x['_'.join([modality, "dates"])], x['scale'], x['_'.join([modality, "mask"])])
                    if modality != "modis":
                        out['_'.join(['masks', modality])] = get_mask(x['_'.join([modality, "mask"])], modality)
                else:
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], x['_'.join([modality, "dates"])], x['scale'])
            token = self.spatial_encoder(token, modality, x['dataset'], x['scale'])
            if modality == "modis":
                tokens.insert(0, token.unsqueeze(1))
            else:
                token = token.view(-1, N - 1, self.embed_dim)
                out['_'.join(['tokens', modality])] = token
                tokens.append(token + pos_embed[:, 1:, :])

        tokens = torch.cat(tokens, dim=1)
        return tokens, out
    
    def forward_transformer(self, x, mask, dataset, scale):
        """
        Forward function after masking used during pretraining
        """
        pos_embed = self.pos_embed['_'.join([dataset, str(scale)])].to(x.device)
        if self.cls_token is not None:
            cls_tokens = (self.cls_token + pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, x), dim=1)
        tokens = self.norm_pre(tokens)
        for blk in self.blocks[:-1]:
            tokens = blk(tokens)
        tokens = self.blocks[-1](tokens, mask, dataset=dataset, scale=scale)
        return tokens

    def forward(self, x):
        """
        Complete forward function during training
        """
        tokens = []
        out = {}
        pos_embed = self.pos_embed['_'.join([x['dataset'], str(x['scale'])])].to(x['label'].device)
        _, N, _ = pos_embed.shape
        for modality in self.modalities[x['dataset']]:
            if modality == "aerial" or modality == "spot" or modality == "aerial-flair" or modality == "naip":
                token = getattr(self, '_'.join(['projector', modality]))(x[modality], x['scale'])
            else:
                if '_'.join([modality, "mask"]) in list(x.keys()):
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], 
                        x['_'.join([modality, "dates"])], x['scale'], x['_'.join([modality, "mask"])])
                else:
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], x['_'.join([modality, "dates"])], x['scale'])

            if self.keep_subpatch and modality == self.modality_keep:
                token, subs = self.spatial_encoder(token, modality, x['dataset'], x['scale'], keep_subpatch=True)
                out['_'.join(['subpatches'])] = subs.view(-1, N - 1, subs.shape[1], subs.shape[2])
            else:
                token = self.spatial_encoder(token, modality, x['dataset'], x['scale'])
            if modality == "modis":
                tokens.insert(0, token.unsqueeze(1))
            else:
                token = token.view(-1, N - 1, self.embed_dim)
                tokens.append(token + pos_embed[:, 1:, :])

        tokens = torch.cat(tokens, dim=1)
        if self.cls_token is not None:
            cls_tokens = (self.cls_token + pos_embed[:, :1, :]).expand(token.shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.patch_drop(tokens)
        tokens = self.norm_pre(tokens)
        for blk in self.blocks[:-1]:
            tokens = blk(tokens)
        tokens = self.blocks[-1](tokens, dataset=x['dataset'], scale=x['scale'])
        if self.keep_subpatch:
            return tokens, out
        return tokens
    
    def forward_release(self, x, scale, output='patch', output_modality=''):
        tokens = []
        out = {}
        keep_subpatch = (output == 'dense')
        modalities = [mod for mod in x.keys() if not (mod.endswith('_dates') or mod.endswith('_mask'))]
        if keep_subpatch and output_modality == '':
            output_modality = modalities[0]
        batch_size = x[modalities[0]].shape[0]
        device = x[modalities[0]].device
        n_modalities = len(modalities)
        modis = ('modis' in modalities)
        pos_embed = None
        for modality in modalities:
            if modality == "aerial" or modality == "spot" or modality == "aerial-flair" or modality == "naip" \
                    or modality == "_rgb" or modality == "_s2_s1" or modality == "_s2":
                token = getattr(self, '_'.join(['projector', modality]))(x[modality], scale)
            else:
                if '_'.join([modality, "mask"]) in list(x.keys()):
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], 
                        x['_'.join([modality, "dates"])], scale, x['_'.join([modality, "mask"])])
                else:
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], x['_'.join([modality, "dates"])], scale)
            
            if pos_embed is None and modality != "modis":
                B, _, C = token.shape
                N = B // batch_size
                num_patches = int(N**(1/2))
                pos_embed = get_2d_sincos_pos_embed_with_scale(C, 
                                                       num_patches, 
                                                       scale, 
                                                       cls_token=True).to(device)
            if keep_subpatch and modality == output_modality:
                token, subs = self.spatial_encoder.forward_release(token, modality, scale, keep_subpatch=True)
                out['_'.join(['subpatches'])] = subs.view(-1, N, subs.shape[1], subs.shape[2])
            else:
                token = self.spatial_encoder.forward_release(token, modality, scale)
            if modality == "modis":
                tokens.insert(0, token.unsqueeze(1))
            else:
                token = token.view(-1, N, self.embed_dim)
                tokens.append(token + pos_embed[:, 1:, :])

        tokens = torch.cat(tokens, dim=1)
        if self.cls_token is not None:
            cls_tokens = (self.cls_token + pos_embed[:, :1, :]).expand(token.shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.patch_drop(tokens)
        tokens = self.norm_pre(tokens)
        for blk in self.blocks[:-1]:
            tokens = blk(tokens)
        tokens = self.blocks[-1].forward_release(tokens, n_modalities=n_modalities, modis=modis, scale=scale)
        if keep_subpatch:
            tokens = tokens[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
            dense_tokens = torch.cat([tokens, out['subpatches']], dim = 3)
            B, N, P, D = dense_tokens.shape
            patch_size = int(P**(1/2))
            size = num_patches * patch_size
            dense_tokens = dense_tokens.unsqueeze(2).permute(0, 2, 4, 1, 3)
            dense_tokens = dense_tokens.view(B, 1, D, N, patch_size, patch_size)
            dense_tokens = dense_tokens.view(B, 1, D, num_patches, num_patches, patch_size, patch_size).permute(0, 1, 2, 3, 5, 4, 6)
            dense_tokens = dense_tokens.reshape(B, 1, D, size, size).flatten(0, 1).permute(0, 2, 3, 1)
            return dense_tokens
        if output == 'tile':
            return tokens[:, 0, :]
        if output == 'patch':
            return tokens[:, 1:, :].view(batch_size, num_patches, num_patches, C)
        return tokens

def get_mask(mask, modality):
    if modality in ['alos', 'l7']:
        return torch.max(mask.flatten(1, 2), dim=1).values.flatten(1, 2)
    else:
        scale = 3
        mask = mask.flatten(1, 2).unfold(2, scale, scale).unfold(3, scale, scale)
        mask = mask.flatten(2, 3).flatten(3, 4)
        mask = mask.permute(0, 2, 1, 3).flatten(2, 3)
    return torch.max(mask, dim=2).values


def get_2d_sincos_pos_embed_with_resolution(
    embed_dim, grid_size, res, cls_token=False, modalities=[]
):
    """
    grid_size: int of the grid height and width
    res: array of size n, representing the resolution of a pixel (say, in meters),
    return:
    pos_embed: dict of [n,grid_size*grid_size, embed_dim] or [n,1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    pos_embed_final = {}
    for modality in modalities:
        grid_size_aug = max(1, int(grid_size * 10 / res[modality]))
        if modality in ["planet"]:
            grid_size_aug = grid_size
        grid_h = torch.arange(grid_size_aug, dtype=torch.float32)
        grid_w = torch.arange(grid_size_aug, dtype=torch.float32)
        grid = torch.meshgrid(
            grid_w, grid_h, indexing="xy"
        )  # here h goes first,direction reversed for numpy
        grid = torch.stack(grid, dim=0)  # 2 x h x w

        # grid = grid.reshape([2, 1, grid_size, grid_size])
        grid = torch.einsum("chw,n->cnhw", grid, torch.tensor([res[modality]]))  # 2 x n x h x w
        _, n, h, w = grid.shape
        pos_embed = get_2d_sincos_pos_embed_from_grid_torch(
            embed_dim, grid
        )  #  # (nxH*W, D/2)
        pos_embed = pos_embed.reshape(n, h * w, embed_dim)
        if cls_token:
            pos_embed = torch.cat(
                [
                    torch.zeros(
                        [n, 1, embed_dim], dtype=torch.float32, device=pos_embed.device
                    ),
                    pos_embed,
                ],
                dim=1,
            )
        pos_embed_final[modality] = pos_embed
    return pos_embed_final

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        to_2tuple = _ntuple(2)
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


@torch.no_grad()
def piecewise_index(relative_position, alpha, beta, gamma, dtype):
    """piecewise index function defined in Eq. (18) in our paper.

    Parameters
    ----------
    relative_position: torch.Tensor, dtype: long or float
        The shape of `relative_position` is (L, L).
    alpha, beta, gamma: float
        The coefficients of piecewise index function.

    Returns
    -------
    idx: torch.Tensor, dtype: long
        A tensor indexing relative distances to corresponding encodings.
        `idx` is a long tensor, whose shape is (L, L) and each element is in [-beta, beta].
    """
    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (alpha +
                                   torch.log(rp_abs_out / alpha) /
                                   math.log(gamma / alpha) *
                                   (beta - alpha)).round().clip(max=beta)).to(dtype)

    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        # round(x) when |x| <= alpha
        idx = idx.round().to(dtype)

    # assign the value when |x| > alpha
    idx[not_mask] = y_out
    return idx


def get_absolute_positions(height, width, dtype, device):
    '''Get absolute positions

    Take height = 3, width = 3 as an example:
    rows:        cols:
    1 1 1        1 2 3
    2 2 2        1 2 3
    3 3 3        1 2 3

    return stack([rows, cols], 2)

    Parameters
    ----------
    height, width: int
        The height and width of feature map
    dtype: torch.dtype
        the data type of returned value
    device: torch.device
        the device of returned value

    Return
    ------
    2D absolute positions: torch.Tensor
        The shape is (height, width, 2),
        where 2 represents a 2D position (row, col).
    '''
    rows = torch.arange(height, dtype=dtype, device=device).view(
        height, 1).repeat(1, width)
    cols = torch.arange(width, dtype=dtype, device=device).view(
        1, width).repeat(height, 1)
    return torch.stack([rows, cols], 2)



class METHOD:
    """define iRPE method IDs
    We divide the implementation of CROSS into CROSS_ROWS and CROSS_COLS.

    """
    EUCLIDEAN = 0
    QUANT = 1
    PRODUCT = 3
    CROSS = 4
    CROSS_ROWS = 41
    CROSS_COLS = 42


@torch.no_grad()
def _rp_2d_euclidean(diff, **kwargs):
    """2D RPE with Euclidean method.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """
    dis = diff.square().sum(2).float().sqrt().round()
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_quant(diff, **kwargs):
    """2D RPE with Quantization method.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """

    dis = diff.square().sum(2)
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_product(diff, **kwargs):
    """2D RPE with Product method.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """
    # convert beta to an integer since beta is a float number.
    beta_int = int(kwargs['beta'])
    S = 2 * beta_int + 1
    # the output of piecewise index function is in [-beta_int, beta_int]
    r = piecewise_index(diff[:, :, 0], **kwargs) + \
        beta_int  # [0, 2 * beta_int]
    c = piecewise_index(diff[:, :, 1], **kwargs) + \
        beta_int  # [0, 2 * beta_int]
    pid = r * S + c
    return pid


@torch.no_grad()
def _rp_2d_cross_rows(diff, **kwargs):
    """2D RPE with Cross for rows.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """
    dis = diff[:, :, 0]
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_cross_cols(diff, **kwargs):
    """2D RPE with Cross for columns.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """

    dis = diff[:, :, 1]
    return piecewise_index(dis, **kwargs)


# Define a mapping from METHOD_ID to Python function
_METHOD_FUNC = {
    METHOD.EUCLIDEAN: _rp_2d_euclidean,
    METHOD.QUANT: _rp_2d_quant,
    METHOD.PRODUCT: _rp_2d_product,
    METHOD.CROSS_ROWS: _rp_2d_cross_rows,
    METHOD.CROSS_COLS: _rp_2d_cross_cols,
}


def get_num_buckets(method, alpha, beta, gamma):
    """ Get number of buckets storing relative position encoding.
    The buckets does not contain `skip` token.

    Parameters
    ----------
    method: METHOD
        The method ID of image relative position encoding.
    alpha, beta, gamma: float
        The coefficients of piecewise index function.

    Returns
    -------
    num_buckets: int
        The number of buckets storing relative position encoding.
    """
    beta_int = int(beta)
    if method == METHOD.PRODUCT:
        # IDs in [0, (2 * beta_int + 1)^2) for Product method
        num_buckets = (2 * beta_int + 1) ** 2
    else:
        # IDs in [-beta_int, beta_int] except of Product method
        num_buckets = 2 * beta_int + 1
    return num_buckets


# (method, alpha, beta, gamma) -> (bucket_ids, num_buckets, height, width)
BUCKET_IDS_BUF = dict()


@torch.no_grad()
def get_bucket_ids_2d_without_skip(method, height, width,
                                   alpha, beta, gamma,
                                   dtype=torch.long, device=torch.device('cpu')):
    """Get bucket IDs for image relative position encodings without skip token

    Parameters
    ----------
    method: METHOD
        The method ID of image relative position encoding.
    height, width: int
        The height and width of the feature map.
        The sequence length is equal to `height * width`.
    alpha, beta, gamma: float
        The coefficients of piecewise index function.
    dtype: torch.dtype
        the data type of returned `bucket_ids`
    device: torch.device
        the device of returned `bucket_ids`

    Returns
    -------
    bucket_ids: torch.Tensor, dtype: long
        The bucket IDs which index to corresponding encodings.
        The shape of `bucket_ids` is (skip + L, skip + L),
        where `L = height * wdith`.
    num_buckets: int
        The number of buckets including `skip` token.
    L: int
        The sequence length
    """

    key = (method, alpha, beta, gamma, dtype, device)
    value = BUCKET_IDS_BUF.get(key, None)
    if value is None or value[-2] < height or value[-1] < width:
        if value is None:
            max_height, max_width = height, width
        else:
            max_height = max(value[-2], height)
            max_width = max(value[-1], width)
        # relative position encoding mapping function
        func = _METHOD_FUNC.get(method, None)
        if func is None:
            raise NotImplementedError(
                f"[Error] The method ID {method} does not exist.")
        pos = get_absolute_positions(max_height, max_width, dtype, device)

        # compute the offset of a pair of 2D relative positions
        max_L = max_height * max_width
        pos1 = pos.view((max_L, 1, 2))
        pos2 = pos.view((1, max_L, 2))
        # diff: shape of (L, L, 2)
        diff = pos1 - pos2

        # bucket_ids: shape of (L, L)
        bucket_ids = func(diff, alpha=alpha, beta=beta,
                          gamma=gamma, dtype=dtype)
        beta_int = int(beta)
        if method != METHOD.PRODUCT:
            bucket_ids += beta_int
        bucket_ids = bucket_ids.view(
            max_height, max_width, max_height, max_width)

        num_buckets = get_num_buckets(method, alpha, beta, gamma)
        value = (bucket_ids, num_buckets, height, width)
        BUCKET_IDS_BUF[key] = value
    L = height * width
    bucket_ids = value[0][:height, :width, :height, :width].reshape(L, L)
    num_buckets = value[1]

    return bucket_ids, num_buckets, L


@torch.no_grad()
def get_bucket_ids_2d(method, height, width,
                      skip, alpha, beta, gamma,
                      dtype=torch.long, device=torch.device('cpu')):
    """Get bucket IDs for image relative position encodings

    Parameters
    ----------
    method: METHOD
        The method ID of image relative position encoding.
    height, width: int
        The height and width of the feature map.
        The sequence length is equal to `height * width`.
    skip: int
        The number of skip token before spatial tokens.
        When skip is 0, no classification token.
        When skip is 1, there is a classification token before spatial tokens.
        When skip > 1, there are `skip` extra tokens before spatial tokens.
    alpha, beta, gamma: float
        The coefficients of piecewise index function.
    dtype: torch.dtype
        the data type of returned `bucket_ids`
    device: torch.device
        the device of returned `bucket_ids`

    Returns
    -------
    bucket_ids: torch.Tensor, dtype: long
        The bucket IDs which index to corresponding encodings.
        The shape of `bucket_ids` is (skip + L, skip + L),
        where `L = height * wdith`.
    num_buckets: int
        The number of buckets including `skip` token.
    """
    bucket_ids, num_buckets, L = get_bucket_ids_2d_without_skip(method, height, width,
                                                                alpha, beta, gamma,
                                                                dtype, device)

    # add an extra encoding (id = num_buckets) for the classification token
    if skip > 0:
        new_bids = bucket_ids.new_empty(size=(skip + L, skip + L))

        # if extra token exists, we add extra bucket as its encoding.
        extra_bucket_id = num_buckets
        num_buckets += 1

        new_bids[:skip] = extra_bucket_id
        new_bids[:, :skip] = extra_bucket_id
        new_bids[skip:, skip:] = bucket_ids

        bucket_ids = new_bids
    bucket_ids = bucket_ids.contiguous()
    return bucket_ids, num_buckets

def get_single_rpe_config(ratio=1.9,
                          method=METHOD.PRODUCT,
                          mode='contextual',
                          shared_head=True,
                          skip=0):
    """Get the config of single relative position encoding

    Parameters
    ----------
    ratio: float
        The ratio to control the number of buckets.
    method: METHOD
        The method ID of image relative position encoding.
        The `METHOD` class is defined in `irpe.py`.
    mode: str or None
        The mode of image relative position encoding.
        Choices: [None, 'bias', 'contextual']
    shared_head: bool
        Whether to share weight among different heads.
    skip: int
        The number of skip token before spatial tokens.
        When skip is 0, no classification token.
        When skip is 1, there is a classification token before spatial tokens.
        When skip > 1, there are `skip` extra tokens before spatial tokens.

    Returns
    -------
    config: RPEConfig
        The config of single relative position encoding.
    """
    config = {}
    # whether to share encodings across different heads
    config['shared_head'] = shared_head
    # mode: None, bias, contextual
    config['mode'] = mode
    # method: None, Bias, Quant, Cross, Product
    config['method'] = method
    # the coefficients of piecewise index function
    config['alpha'] = 1 * ratio
    config['beta'] = 2 * ratio
    config['gamma'] = 8 * ratio

    # set the number of buckets
    config['num_buckets'] = get_num_buckets(method,
                                         config['alpha'],
                                         config['beta'],
                                         config['gamma'])
    # add extra bucket for `skip` token (e.g. class token)
    if skip > 0:
        config['num_buckets'] += 1
    return config


def get_rpe_config(ratio=1.9,
                   method=METHOD.PRODUCT,
                   mode='contextual',
                   shared_head=True,
                   skip=0,
                   rpe_on='k'):
    """Get the config of relative position encoding on queries, keys and values

    Parameters
    ----------
    ratio: float
        The ratio to control the number of buckets.
    method: METHOD or str
        The method ID (or name) of image relative position encoding.
        The `METHOD` class is defined in `irpe.py`.
    mode: str or None
        The mode of image relative position encoding.
        Choices: [None, 'bias', 'contextual']
    shared_head: bool
        Whether to share weight among different heads.
    skip: int
        The number of skip token before spatial tokens.
        When skip is 0, no classification token.
        When skip is 1, there is a classification token before spatial tokens.
        When skip > 1, there are `skip` extra tokens before spatial tokens.
    rpe_on: str
        Where RPE attaches.
        "q": RPE on queries
        "k": RPE on keys
        "v": RPE on values
        "qk": RPE on queries and keys
        "qkv": RPE on queries, keys and values

    Returns
    -------
    config: RPEConfigs
        config.rpe_q: the config of relative position encoding on queries
        config.rpe_k: the config of relative position encoding on keys
        config.rpe_v: the config of relative position encoding on values
    """

    # alias
    if isinstance(method, str):
        method_mapping = dict(
            euc=METHOD.EUCLIDEAN,
            quant=METHOD.QUANT,
            cross=METHOD.CROSS,
            product=METHOD.PRODUCT,
        )
        method = method_mapping[method.lower()]
    if mode == 'ctx':
        mode = 'contextual'
    config = {}
    # relative position encoding on queries, keys and values
    kwargs = dict(
        ratio=ratio,
        method=method,
        mode=mode,
        shared_head=shared_head,
        skip=skip,
    )
    config['rpe_q'] = get_single_rpe_config(**kwargs) if 'q' in rpe_on else None
    config['rpe_k'] = get_single_rpe_config(**kwargs) if 'k' in rpe_on else None
    config['rpe_v'] = get_single_rpe_config(**kwargs) if 'v' in rpe_on else None
    return config


def build_rpe(config, head_dim, num_heads, n_modalities=1):
    """Build iRPE modules on queries, keys and values.

    Parameters
    ----------
    config: RPEConfigs
        config.rpe_q: the config of relative position encoding on queries
        config.rpe_k: the config of relative position encoding on keys
        config.rpe_v: the config of relative position encoding on values
        None when RPE is not used.
    head_dim: int
        The dimension for each head.
    num_heads: int
        The number of parallel attention heads.

    Returns
    -------
    modules: a list of nn.Module
        The iRPE Modules on [queries, keys, values].
        None when RPE is not used.
    """
    if config is None:
        return None, None, None
    rpes = [config['rpe_q'], config['rpe_k'], config['rpe_v']]
    transposeds = [True, True, False]

    def _build_single_rpe(rpe, transposed):
        if rpe is None:
            return None

        rpe_cls = iRPE if rpe['method'] != METHOD.CROSS else iRPE_Cross
        return rpe_cls(
            head_dim=head_dim,
            num_heads=1 if rpe['shared_head'] else num_heads,
            mode=rpe['mode'],
            method=rpe['method'],
            transposed=transposed,
            num_buckets=rpe['num_buckets'],
            rpe_config=rpe,
            n_modalities=n_modalities
        )
    return [_build_single_rpe(rpe, transposed)
            for rpe, transposed in zip(rpes, transposeds)]



class RPEAttention(nn.Module):
    '''
    Attention with image relative position encoding
    '''

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., n_modalities=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.n_modalities = n_modalities

        # image relative position encoding
        rpe_config = get_rpe_config(
                ratio=1.9,
                method="euc",
                mode='ctx',
                shared_head=True,
                skip=1,
                rpe_on='k',
            )
        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config,
                      head_dim=head_dim,
                      num_heads=num_heads,
                      n_modalities=n_modalities)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        if mask is None:
            height = int((N // self.n_modalities) ** .5)
        else:
            height = mask.shape[-1]

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q *= self.scale

        attn = (q @ k.transpose(-2, -1))
        # image relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q, pos=mask, height=height, width=height)

        # # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        # image relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BlockTransformer(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_modalities=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RPEAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, 
            proj_drop=drop, n_modalities=n_modalities)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)   


class TransformerMulti(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        input_res={},
        modalities={},
        scales={},
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.input_res = input_res
        datasets = list(scales.keys())
        self.predictor_pos_embed = {}
        for dataset in datasets:
            for scale in scales[dataset]:
                self.predictor_pos_embed['_'.join([dataset, str(scale)])] = get_2d_sincos_pos_embed_with_resolution(
                                                                                embed_dim,
                                                                                scale,
                                                                                input_res,
                                                                                cls_token=True,
                                                                                modalities=modalities[dataset]
                                                                            )
        # --
        self.predictor_blocks = nn.ModuleList([
            BlockTransformer(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=None, n_modalities=1,
                drop=0., attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, modality, dataset, scale, keep_subpatch=False):
        # -- concat class token to x
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)

        # -- add positional embedding to x tokens
        x += self.predictor_pos_embed['_'.join([dataset, str(scale)])][modality].to(x.device)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        if keep_subpatch:
            return x[:, 0], x[:, 1:]
        
        return x[:, 0]
    
    def forward_release(self, x, modality, scale, keep_subpatch=False):
        B, N, C = x.shape
        # -- concat class token to x
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        # -- add positional embedding to x tokens
        x += get_2d_sincos_pos_embed_with_resolution(C,
                                                        scale,
                                                        self.input_res,
                                                        cls_token=True,
                                                        modalities=[modality]
                                                    )[modality].to(x.device)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        if keep_subpatch:
            return x[:, 0], x[:, 1:]
        
        return x[:, 0]
    


EXPECTED_VERSION = "1.2.0"
assert rpe_index_cpp.version() == EXPECTED_VERSION, \
        f"""Unmatched `rpe_index_cpp` version: {rpe_index_cpp.version()}, expected version: {EXPECTED_VERSION}
Please re-build the package `rpe_ops`."""

RPEIndexFunction = None

# class RPEIndexFunction(torch.autograd.Function):
#     '''Y[b, h, i, j] = input[b, h, i, index[i, j]]'''
#     @staticmethod
#     def forward(ctx, input, index):
#         '''
#         Y[b, h, i, j] = input[b, h, i, index[i, j]]

#         Parameters
#         ----------
#         input: torch.Tensor, float32
#             The shape is (B, H, L_query, num_buckets)
#         index: torch.Tensor, int32
#             The shape is (L_query, L_key)

#         where B is the batch size, and H is the number of attention heads.

#         Returns
#         -------
#         Y: torch.Tensor, float32
#             The shape is (B, H, L_query, L_key)
#         '''

#         num_buckets = input.size(-1)
#         ctx.save_for_backward(index)
#         ctx.input_shape = input.shape
#         forward_fn = rpe_index_cpp.forward_cpu if \
#             input.device.type == 'cpu' else rpe_index_cpp.forward_gpu
#         output = forward_fn(input, index)
#         return output


class iRPE(nn.Module):
    """The implementation of image relative position encoding (excluding Cross method).

    Parameters
    ----------
    head_dim: int
        The dimension for each head.
    num_heads: int
        The number of parallel attention heads.
    mode: str or None
        The mode of image relative position encoding.
        Choices: [None, 'bias', 'contextual']
    method: METHOD
        The method ID of image relative position encoding.
        The `METHOD` class is defined in `irpe.py`.
    transposed: bool
        Whether to transpose the input feature.
        For iRPE on queries or keys, transposed should be `True`.
        For iRPE on values, transposed should be `False`.
    num_buckets: int
        The number of buckets, which store encodings.
    initializer: None or an inplace function
        [Optional] The initializer to `lookup_table`.
        Initalize `lookup_table` as zero by default.
    rpe_config: RPEConfig
        The config generated by the function `get_single_rpe_config`.
    """
    # a buffer to store bucket index
    # (key, rp_bucket, _ctx_rp_bucket_flatten)
    _rp_bucket_buf = (None, None, None)

    def __init__(self, head_dim, num_heads=8,
                 mode=None, method=None,
                 transposed=True, num_buckets=None,
                 initializer=None, rpe_config=None, n_modalities=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_modalities = n_modalities

        # relative position
        assert mode in [None, 'bias', 'contextual']
        self.mode = mode

        assert method is not None, 'method should be a METHOD ID rather than None'
        self.method = method

        self.transposed = transposed
        self.num_buckets = num_buckets

        if initializer is None:
            def initializer(x): return None
        self.initializer = initializer

        self.reset_parameters()
        trunc_normal_(self.lookup_table_weight, std=.02)

        self.rpe_config = rpe_config

    @torch.no_grad()
    def reset_parameters(self):
        # initialize the parameters of iRPE
        if self.transposed:
            if self.mode == 'bias':
                self.lookup_table_bias = nn.Parameter(
                    torch.zeros(self.num_heads, self.num_buckets))
                self.initializer(self.lookup_table_bias)
            elif self.mode == 'contextual':
                self.lookup_table_weight = nn.Parameter(
                    torch.zeros(self.num_heads,
                                self.head_dim, self.num_buckets))
                self.initializer(self.lookup_table_weight)
        else:
            if self.mode == 'bias':
                raise NotImplementedError(
                    "[Error] Bias non-transposed RPE does not exist.")
            elif self.mode == 'contextual':
                self.lookup_table_weight = nn.Parameter(
                    torch.zeros(self.num_heads,
                                self.num_buckets, self.head_dim))
                self.initializer(self.lookup_table_weight)

    def forward(self, x, height=None, width=None, pos=None, modis=False):
        """forward function for iRPE.

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor whose shape is (B, H, L, head_dim),
            where B is batch size,
                  H is the number of heads,
                  L is the sequence length,
                    equal to height * width (+1 if class token exists)
                  head_dim is the dimension of each head

        Returns
        -------
        rpe_encoding: torch.Tensor
            image Relative Position Encoding,
            whose shape is (B, H, L, L)
        """
        rp_bucket, self._ctx_rp_bucket_flatten = \
            self._get_rp_bucket(x, height=height, width=width, pos=pos, modis=modis)
        if self.transposed:
            return self.forward_rpe_transpose(x, rp_bucket)
        return self.forward_rpe_no_transpose(x, rp_bucket)

    def _get_rp_bucket(self, x, height=None, width=None, pos=None, modis=False):
        """Get relative position encoding buckets IDs corresponding the input shape

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor whose shape is (B, H, L, head_dim),
            where B is batch size,
                  H is the number of heads,
                  L is the sequence length,
                    equal to height * width (+1 if class token exists)
                  head_dim is the dimension of each head
        height: int or None
            [Optional] The height of the input
            If not defined, height = floor(sqrt(L))
        width: int or None
            [Optional] The width of the input
            If not defined, width = floor(sqrt(L))
        pos: position of tokens in the image

        Returns
        -------
        rp_bucket: torch.Tensor
            relative position encoding buckets IDs
            The shape is (L, L)
        _ctx_rp_bucket_flatten: torch.Tensor or None
            It is a private tensor for efficient computation.
        """
        B, H, L, D = x.shape
        device = x.device
        if height is None:
            E = int(math.sqrt(L // self.n_modalities))
            height = width = E

        if pos is None:
            key = (B, height, width, device, None)
            if self._rp_bucket_buf[0] == key:
                return self._rp_bucket_buf[1:3]
        else:
            key = (B, height, width, device)

        skip = 1 + int(modis)
        config = self.rpe_config
        if RPEIndexFunction is not None and self.mode == 'contextual' and self.transposed:
            # RPEIndexFunction uses int32 index.
            dtype = torch.int32
        else:
            dtype = torch.long

        rp_bucket, num_buckets = get_bucket_ids_2d(method=self.method,
                                                   height=height, width=width,
                                                   skip=skip, alpha=config['alpha'],
                                                   beta=config['beta'], gamma=config['gamma'],
                                                   dtype=dtype, device=device)
        rp_bucket = rp_bucket.unsqueeze(0).repeat(B, 1, 1)
        if pos is not None:
            rp_bucket = torch.cat([rp_bucket[:, :skip, :]] + [torch.gather(rp_bucket[:, skip:, :], dim=1, index=pos.unsqueeze(-1).repeat(1, 1, 
                                                                                                            rp_bucket.shape[2]))], dim=1)
            rp_bucket = torch.cat([rp_bucket[:, :, :skip]] + [torch.gather(rp_bucket[:, :, skip:], dim=2, index=pos.unsqueeze(1).repeat(1, 
                                                                                                            rp_bucket.shape[1], 1))], dim=2)
        rp_bucket = torch.cat([rp_bucket[:, :1, :]] + [rp_bucket[:, 1:, :] for _ in range(self.n_modalities)], dim=1)
        rp_bucket = torch.cat([rp_bucket[:, :, :1]] + [rp_bucket[:, :, 1:] for _ in range(self.n_modalities)], dim=2)
        assert num_buckets == self.num_buckets

        # transposed contextual
        _ctx_rp_bucket_flatten = None
        if self.mode == 'contextual' and self.transposed:
            if RPEIndexFunction is None:
                offset = torch.arange(0, L * self.num_buckets, self.num_buckets,
                                      dtype=rp_bucket.dtype, device=rp_bucket.device).view(-1, 1).unsqueeze(0).repeat(B, 1, 1)
                _ctx_rp_bucket_flatten = (rp_bucket + offset).flatten(1,2)
        self._rp_bucket_buf = (key, rp_bucket, _ctx_rp_bucket_flatten)
        return rp_bucket, _ctx_rp_bucket_flatten

    def forward_rpe_transpose(self, x, rp_bucket):
        """Forward function for iRPE (transposed version)
        This version is utilized by RPE on Query or Key

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor whose shape is (B, H, L, head_dim),
            where B is batch size,
                  H is the number of heads,
                  L is the sequence length,
                    equal to height * width (+1 if class token exists)
                  head_dim is the dimension of each head
        rp_bucket: torch.Tensor
            relative position encoding buckets IDs
            The shape is (L, L)

        Weights
        -------
        lookup_table_bias: torch.Tensor
            The shape is (H or 1, num_buckets)

        or

        lookup_table_weight: torch.Tensor
            The shape is (H or 1, head_dim, num_buckets)

        Returns
        -------
        output: torch.Tensor
            Relative position encoding on queries or keys.
            The shape is (B or 1, H, L, L),
            where D is the output dimension for each head.
        """

        B, L_query, L_key = rp_bucket.shape
        if L_query != x.shape[2]:
            print("début")
            print(L_query, x.shape[2])
            print("fin")
        if self.mode == 'bias':
            return self.lookup_table_bias[:, rp_bucket.flatten()].\
                view(1, self.num_heads, L_query, L_key)

        elif self.mode == 'contextual':
            """
            ret[b, h, i, j] = lookup_table_weight[b, h, i, rp_bucket[i, j]]

            ret[b, h, i * L_key + j] = \
               lookup_table[b, h, i * num_buckets + rp_buckets[i, j]]

            computational cost
            ------------------
            matmul: B * H * L_query * head_dim * num_buckets
            index: L_query + L_query * L_key + B * H * L_query * L_key
            total: O(B * H * L_query * (head_dim * num_buckets + L_key))
            """
            try:
                lookup_table = torch.matmul(
                    x.transpose(0, 1).reshape(-1, B * L_query, self.head_dim),
                    self.lookup_table_weight).\
                    view(-1, B, L_query, self.num_buckets).transpose(0, 1)
            except RuntimeError:
                print(x.shape)
                print(self.lookup_table_weight.shape)
                print(B, L_query, self.head_dim)
                raise
            if RPEIndexFunction is not None:
                return RPEIndexFunction.apply(lookup_table, rp_bucket)
            else:
                look = lookup_table.flatten(2)
                look = torch.gather(look, dim=2, index=self._ctx_rp_bucket_flatten.unsqueeze(1)).view(B, -1, L_query, L_key)
                return look
                # return lookup_table.flatten(2)[:, :, self._ctx_rp_bucket_flatten].\
                #     view(B, -1, L_query, L_key)

    def forward_rpe_no_transpose(self, x, rp_bucket):
        """Forward function for iRPE (non-transposed version)
        This version is utilized by RPE on Value.

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor whose shape is (B, H, L, head_dim),
            where B is batch size,
                  H is the number of heads,
                  L is the sequence length,
                    equal to height * width (+1 if class token exists)
                  head_dim is the dimension of each head
        rp_bucket: torch.Tensor
            relative position encoding buckets IDs
            The shape is (L, L)

        Weights
        -------
        lookup_table_weight: torch.Tensor
            The shape is (H or 1, num_buckets, head_dim)

        Returns
        -------
        output: torch.Tensor
            Relative position encoding on values.
            The shape is (B, H, L, D),
            where D is the output dimension for each head.
        """

        B = len(x)  # batch_size
        L_query, L_key = rp_bucket.shape
        assert self.mode == 'contextual', "Only support contextual \
version in non-transposed version"
        weight = self.lookup_table_weight[:, rp_bucket.flatten()].\
            view(self.num_heads, L_query, L_key, self.head_dim)
        # (H, L_query, B, L_key) @ (H, L_query, L_key, D) = (H, L_query, B, D)
        # -> (B, H, L_query, D)
        return torch.matmul(x.permute(1, 2, 0, 3), weight).permute(2, 0, 1, 3)

    def __repr__(self):
        return 'iRPE(head_dim={rpe.head_dim}, num_heads={rpe.num_heads}, \
mode="{rpe.mode}", method={rpe.method}, transposed={rpe.transposed}, \
num_buckets={rpe.num_buckets}, initializer={rpe.initializer}, \
rpe_config={rpe.rpe_config})'.format(rpe=self)


class iRPE_Cross(nn.Module):
    """The implementation of image relative position encoding (specific for Cross method).

    Parameters
    ----------
    head_dim: int
        The dimension for each head.
    num_heads: int
        The number of parallel attention heads.
    mode: str or None
        The mode of image relative position encoding.
        Choices: [None, 'bias', 'contextual']
    method: METHOD
        The method ID of image relative position encoding.
        The `METHOD` class is defined in `irpe.py`.
    transposed: bool
        Whether to transpose the input feature.
        For iRPE on queries or keys, transposed should be `True`.
        For iRPE on values, transposed should be `False`.
    num_buckets: int
        The number of buckets, which store encodings.
    initializer: None or an inplace function
        [Optional] The initializer to `lookup_table`.
        Initalize `lookup_table` as zero by default.
    rpe_config: RPEConfig
        The config generated by the function `get_single_rpe_config`.
    """

    def __init__(self, method, **kwargs):
        super().__init__()
        assert method == METHOD.CROSS
        self.rp_rows = iRPE(**kwargs, method=METHOD.CROSS_ROWS)
        self.rp_cols = iRPE(**kwargs, method=METHOD.CROSS_COLS)

    def forward(self, x, height=None, width=None):
        """forward function for iRPE.
        Compute encoding on horizontal and vertical directions separately,
        then summarize them.

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor whose shape is (B, H, L, head_dim),
            where B is batch size,
                  H is the number of heads,
                  L is the sequence length,
                    equal to height * width (+1 if class token exists)
                  head_dim is the dimension of each head
        height: int or None
            [Optional] The height of the input
            If not defined, height = floor(sqrt(L))
        width: int or None
            [Optional] The width of the input
            If not defined, width = floor(sqrt(L))

        Returns
        -------
        rpe_encoding: torch.Tensor
            Image Relative Position Encoding,
            whose shape is (B, H, L, L)
        """

        rows = self.rp_rows(x, height=height, width=width)
        cols = self.rp_cols(x, height=height, width=width)
        return rows + cols

    def __repr__(self):
        return 'iRPE_Cross(head_dim={rpe.head_dim}, \
num_heads={rpe.num_heads}, mode="{rpe.mode}", method={rpe.method}, \
transposed={rpe.transposed}, num_buckets={rpe.num_buckets}, \
initializer={rpe.initializer}, \
rpe_config={rpe.rpe_config})'.format(rpe=self.rp_rows)


def get_single_rpe_config(ratio=1.9,
                          method=METHOD.PRODUCT,
                          mode='contextual',
                          shared_head=True,
                          skip=0):
    """Get the config of single relative position encoding

    Parameters
    ----------
    ratio: float
        The ratio to control the number of buckets.
    method: METHOD
        The method ID of image relative position encoding.
        The `METHOD` class is defined in `irpe.py`.
    mode: str or None
        The mode of image relative position encoding.
        Choices: [None, 'bias', 'contextual']
    shared_head: bool
        Whether to share weight among different heads.
    skip: int
        The number of skip token before spatial tokens.
        When skip is 0, no classification token.
        When skip is 1, there is a classification token before spatial tokens.
        When skip > 1, there are `skip` extra tokens before spatial tokens.

    Returns
    -------
    config: RPEConfig
        The config of single relative position encoding.
    """
    config = {}
    # whether to share encodings across different heads
    config['shared_head'] = shared_head
    # mode: None, bias, contextual
    config['mode'] = mode
    # method: None, Bias, Quant, Cross, Product
    config['method'] = method
    # the coefficients of piecewise index function
    config['alpha'] = 1 * ratio
    config['beta'] = 2 * ratio
    config['gamma'] = 8 * ratio

    # set the number of buckets
    config['num_buckets'] = get_num_buckets(method,
                                         config['alpha'],
                                         config['beta'],
                                         config['gamma'])
    # add extra bucket for `skip` token (e.g. class token)
    if skip > 0:
        config['num_buckets'] += 1
    return config