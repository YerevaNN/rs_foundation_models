import os
import re
import math
import torch
import numpy as np
import torch.nn.functional as F

from box import Box
from torch import nn
from copy import deepcopy
from einops import rearrange
from torchvision.transforms import v2
from vit_pytorch.simple_vit import FeedForward, Attention
from einops import rearrange, reduce, repeat
from .vision_transformer import MultiLevelNeck
from pretrainedmodels.models.torchvision_models import pretrained_settings


torch.set_float32_matmul_precision("medium")
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"


new_settings = {
    "Clay": {
        "clay_v1": "/nfs/h100/raid/rs/Clay/Clay_v0_v1/clay-v1-base.ckpt",  
    },
}


pretrained_settings = deepcopy(pretrained_settings)
for model_name, sources in new_settings.items():
    if model_name not in pretrained_settings:
        pretrained_settings[model_name] = {}

    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'num_classes': 10
        }

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, out_idx):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.out_idx = out_idx
        self.embed_dim = dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
        
    def forward(self, x):
        outs = []
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            if i in self.out_idx:
                out = x[:, 1:, :]

                img_side_length = int(np.sqrt(out.shape[1]))
                out = out.view(-1, img_side_length, img_side_length, self.embed_dim)
                out = out.permute(0, 3, 1, 2)

                outs.append(out)

        return self.norm(x), outs


"""
Code from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

"""


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_2d_with_gsd(
    h, w, dim, gsd=1.0, temperature: int = 10000, dtype=torch.float32
):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** (2 * omega / dim)) * (gsd / 1.0)  # Adjusted for g

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_1d(pos, dim, temperature: int = 10000, dtype=torch.float32):
    assert (
        dim % 2 == 0
    ), "Feature dimension must be a multiple of 2 for sincos embedding"
    pos = torch.arange(pos) if isinstance(pos, int) else pos

    omega = torch.arange(dim // 2) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    scaled_pos = pos[:, None] * omega[None, :]
    pe = torch.cat((scaled_pos.sin(), scaled_pos.cos()), dim=1)

    return pe.type(dtype)


"""Dynamic Embedding from DOFA paper.
Reference:
- https://arxiv.org/abs/2403.15356
- https://github.com/zhu-xlab/DOFA
"""


class FCBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.l1 = nn.Linear(size, size)
        self.l2 = nn.Linear(size, size)

    def forward(self, x):
        y = F.gelu(self.l1(x))
        y = F.gelu(self.l2(y))
        return x + y


class WavesTransformer(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        wave_dim,
        output_dim,
        num_latent_tokens,
        embed_dim,
        is_decoder,
        num_heads=4,
        num_layers=1,
    ):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.is_decoder = is_decoder
        layer = nn.TransformerEncoderLayer(
            d_model=wave_dim,
            nhead=num_heads,
            activation="gelu",
            dropout=0,
            norm_first=False,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

        self.fc_weight = nn.Linear(wave_dim, output_dim)
        self.fc_bias = None if self.is_decoder else nn.Linear(wave_dim, embed_dim)

        self.weight_tokens = nn.Parameter(
            torch.randn(self.num_latent_tokens, wave_dim) * 0.02
        )
        self.bias_token = nn.Parameter(torch.randn(1, wave_dim) * 0.02)

    def forward(self, x):
        x = torch.cat([self.weight_tokens, x, self.bias_token], dim=0)
        out = self.encoder(x)
        weights = self.fc_weight(
            out[self.num_latent_tokens : -1] + x[self.num_latent_tokens : -1]
        )
        bias = None if self.is_decoder else self.fc_bias(out[-1])
        return weights, bias


class DynamicEmbedding(nn.Module):
    def __init__(
        self,
        wave_dim,
        num_latent_tokens,
        patch_size,
        embed_dim,
        is_decoder=False,
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.num_latent_tokens = num_latent_tokens
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.is_decoder = is_decoder
        self.output_dim = (patch_size**2) * embed_dim

        self.weight_generator = WavesTransformer(
            wave_dim,
            self.output_dim,
            self.num_latent_tokens,
            self.embed_dim,
            is_decoder,
        )
        self.fclayer = FCBlock(self.wave_dim)

        self.initialize_weights()

    def forward(self, batch, waves):
        waves = posemb_sincos_1d(waves, self.wave_dim)
        waves = waves.to(batch.device)
        waves = self.fclayer(waves)
        weight, bias = self.weight_generator(waves)

        if self.is_decoder:
            dynamic_weight = rearrange(
                weight,
                "cin (k1 k2 cout) -> (cin k1 k2) cout",
                k1=self.patch_size,
                k2=self.patch_size,
                cout=self.embed_dim,
            )
            if bias is not None:
                bias = rearrange(bias, "b -> (b)")
            dynamic_out = F.linear(batch, dynamic_weight * 0.02, bias=bias)
            x = dynamic_out
        else:
            dynamic_weight = rearrange(
                weight,
                "cin (cout k1 k2) -> cout cin k1 k2",
                k1=self.patch_size,
                k2=self.patch_size,
            )

            if bias is not None:
                bias = rearrange(bias, "b -> (b)")
            dynamic_out = F.conv2d(
                batch, dynamic_weight * 0.02, bias=bias, stride=self.patch_size
            )
            x = rearrange(dynamic_out, "b c h w -> b (h w) c")

        return x, waves

    def initialize_weights(self):
        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Encoder(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        patch_size,
        shuffle,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        out_idx, 
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.dim = dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.patch_embedding = DynamicEmbedding(
            wave_dim=128,
            num_latent_tokens=128,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=False,
        )

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
            out_idx=out_idx,
        )

    def to_patch_embed(self, cube, waves):
        """Split the input cube into patches & create embeddings per patch"""
        patches, waves_encoded = self.patch_embedding(cube, waves)  # [B L D]
        return patches, waves_encoded  # ([B L D], [N D])

    def add_encodings(self, patches, time, latlon, gsd):
        """Add position encoding to the patches"""
        B, L, D = patches.shape

        grid_size = int(math.sqrt(L))
        self.num_patches = grid_size**2

        pos_encoding = (
            posemb_sincos_2d_with_gsd(
                h=grid_size,
                w=grid_size,
                dim=(self.dim - 8),
                gsd=gsd,
            )
            .to(patches.device)
            .detach()
        )  # [L (D - 8)]

        time_latlon = torch.hstack((time, latlon)).to(patches.device).detach()  # [B 8]

        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)  # [B L (D - 8)]
        time_latlon = repeat(time_latlon, "B D -> B L D", L=L)  # [B L 8]
        # print(pos_encoding.shape)
        # print(time_latlon.shape)

        pos_metadata_encoding = torch.cat(
            (pos_encoding, time_latlon), dim=-1
        )  # [B L D]

        patches = patches + pos_metadata_encoding  # [B L D] + [B L D] -> [B L D]
        return patches  # [B L D]

    def mask_out(self, patches):
        """
        Mask out patches randomly by shuffling the patches & masking out the
        first N patches

        Parameters
        ----------
        patches : torch.Tensor A tensor of shape (B, L, D)

        Returns
        -------
        unmasked_patches : torch.Tensor
            A tensor of shape (B, L:(1 - mask_ratio), D) containing the
            embeddings of the unmasked patches.
        unmasked_indices : torch.Tensor
            A tensor of shape (B, (1 - mask_ratio)) containing the indices of
            the unmasked patches.
        masked_indices : torch.Tensor
            A tensor of shape (B, mask_ratio) containing the indices of the
            masked patches.
        masked_matrix : torch.Tensor
            A tensor of shape (B, L) containing the mask matrix, 1 indicates a masked
            patch & 0 indicates an unmasked patch.
        """
        B, L, D = patches.shape
        # assert (
        #     L == self.num_patches
        # ), f"Expected {self.num_patches} patches, got {L} patches."

        if self.shuffle:  # Shuffle the patches
            noise = torch.randn((B, L), device=patches.device)  # [B L]
        else:  # Don't shuffle, useful for interpolation & inspection of embeddings
            noise = rearrange(
                torch.arange(B * L, device=patches.device), "(B L) -> B L", B=B, L=L
            )

        random_indices = torch.argsort(noise, dim=-1)  # [B L]
        reverse_indices = torch.argsort(random_indices, dim=-1)  # [B L]

        num_masked_patches = int(
            self.mask_ratio * self.num_patches
        )  # Number of patches to be masked out
        masked_indices, unmasked_indices = (
            random_indices[:, :num_masked_patches],  # [B mask_ratio * L]
            random_indices[:, num_masked_patches:],  # [B (1 - mask_ratio) * L]
        )

        # create a mask of shape B L, where 1 indicates a masked patch
        # and 0 indicates an unmasked patch
        masked_matrix = torch.zeros((B, L), device=patches.device)  # [B L] = 0
        masked_matrix[:, :num_masked_patches] = 1  # [B mask_ratio * L] = 1
        masked_matrix = torch.gather(
            masked_matrix, dim=1, index=reverse_indices
        )  # [B L] -> [B L] - reorder the patches

        # mask out the patches
        batch_indices = rearrange(
            torch.arange(B, device=patches.device), "B -> B 1"
        )  # [B 1]
        unmasked_patches = patches[
            batch_indices, unmasked_indices, :
        ]  # [B L:(1 - mask_ratio) D]
        _ = patches[batch_indices, masked_indices, :]  # [B L:mask_ratio D]

        return (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B L:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B L]

    def forward(self, datacube):
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )  # [B C H W]

        B, C, H, W = cube.shape

        patches, waves_encoded = self.to_patch_embed(
            cube, waves
        )  # [B L D] - patchify & create embeddings per patch
        # TODO: Add time & latlon as encoding to patches
        patches = self.add_encodings(
            patches,
            time,
            latlon,
            gsd,
        )  # [B L D] - add position encoding to the embeddings

        # mask out patches
        (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        ) = self.mask_out(
            patches
        )  # [B L:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B L]

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        unmasked_patches = torch.cat(
            (cls_tokens, unmasked_patches), dim=1
        )  # [B (1 + L) D]

        # pass the unmasked patches through the transformer
        encoded_unmasked_patches, outs = self.transformer(
            unmasked_patches
        )  # [B ((1 + L)):(1 - mask_ratio)) D]

        return (
            encoded_unmasked_patches,
            outs,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B ((1 + L):(1 - mask_ratio)) D], [(1-mask_ratio)], [mask_ratio], [B L]




class ClayEncoder(nn.Module):
    """
    Classifier class uses Clay Encoder for feature extraction and a head for
    classification.

    Attributes:
        clay_encoder (Encoder): The encoder for feature extraction.
        head (nn.Sequential): The head for classification.
        device (torch.device): The device to run the model on.
    """

    def __init__(self, 
            ckpt_path,
            depth=12, 
            embed_dim=768, 
            num_heads=12, 
            mask_ratio=0.0, 
            shuffle=False,
            patch_size=8, 
            dim_head=64, 
            mlp_ratio=4.0, 
            for_cls=False, 
            out_idx=None, 
            out_channels=None,
            norm_layer=nn.LayerNorm,):
        """
        Initialize the Classifier.

        Args:
            num_classes (int, optional): The number of classes for
            classification. Defaults to 10.
            ckpt_path (str, optional): Clay MAE pretrained model checkpoint
            path. Defaults to None.
        """
        super().__init__()

        # Initialize Clay Encoder with parameters from base model. Set
        # mask_ratio to 0.0 & shuffle to False for downstream tasks.
        self.clay_encoder = Encoder(
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            shuffle=shuffle,
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            out_idx=out_idx, 
        )

        # Simple 2 layer MLP head for classification
        # self.head = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.25),
        #     nn.Linear(512, num_classes),
        # )

        # Determine the device to run the model on
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Load Clay MAE pretrained weights for the Encoder
        self.load_clay_weights(ckpt_path)

        self.for_cls = for_cls
        self.out_idx = out_idx
        self.out_channels = out_channels
        self.norm = norm_layer(embed_dim)

        if not for_cls:
            self.neck = MultiLevelNeck(in_channels=[768, 768, 768, 768], out_channels=768, scales=[4, 2, 1, 0.5])

    def load_clay_weights(self, ckpt_path):
        """
        Load the weights for Clay MAE Encoder from a checkpoint file.

        Args:
            ckpt_path (str): Clay MAE pretrained model checkpoint path.
        """

        # Load the checkpoint file
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get("state_dict")
        # for param_name, param_tensor in ckpt['state_dict'].items():
        #     print(f"Parameter: {param_name}, Shape: {param_tensor.shape}")

        # Remove model.encoder prefix for the clay encoder
        state_dict = {
            re.sub(r"^model\.encoder\.", "", name): param
            for name, param in state_dict.items()
            if name.startswith("model.encoder")
        }

        parameters = {
            name.replace("clay_encoder.", ""): param
            for name, param in self.clay_encoder.named_parameters()
        }

        # Copy the weights from the state dict to the encoder
        for name, param in parameters.items():
            if name in state_dict and param.size() == state_dict[name].size():
                param.data.copy_(state_dict[name])  # Copy the weights
            else:
                print(f"No matching parameter for {name} with size {param.size()}")


#####################################################
#####################################################
#####################################################

    def prepare_data(self, x, metadata):
        datacube = {}
        for entry in metadata:
            for key, value in entry.items():
                if key not in datacube:
                    datacube[key] = [value]
                else:
                    datacube[key].append(value)

        datacube_time = []
        # Process each time string
        for time_string in datacube['time']:
            time_list = [int(part) for part in time_string.split(':')]
            time_list.insert(0, 0)
            datacube_time.append(time_list)

        datacube.update({"latlon": torch.tensor(datacube['latlon'], dtype=torch.float32).cuda(),
                        "time": torch.tensor(datacube_time).cuda(),
                        "gsd": datacube['gsd'][0],
                        "pixels": x,
                        "waves": torch.tensor(datacube['waves'][0])})
        
        return datacube

#####################################################
#####################################################
#####################################################

    def forward(self, x, metadata):

        datacube = self.prepare_data(x, metadata)

        # Get the embeddings from the encoder
        embeddings, outs, *_ = self.clay_encoder(
            datacube
        )  # embeddings: batch x (1 + row x col) x 768


        if self.for_cls:
            return embeddings[:, 0, :]
        
        return self.neck(tuple(outs))

    

clay_encoders = {
    "clay": {
        "encoder": ClayEncoder,
        "pretrained_settings": pretrained_settings['Clay'],
        "params": {
            # "ckpt_path": '/nfs/h100/raid/rs/Clay/Clay_v0_v1/clay-v1-base.ckpt',
            "depth": 12,
            "embed_dim": 768,
            "num_heads": 12,
            "patch_size": 16,
            "out_idx": (2, 5, 8, 11),
            "out_channels": (768, 768, 768, 768)
        }
    }
}