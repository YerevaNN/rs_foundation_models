import torch.nn as nn
import torch.nn.functional as F
import torch
from copy import deepcopy
from collections import OrderedDict
from pretrainedmodels.models.torchvision_models import pretrained_settings

from ._base import EncoderMixin
from .swin_transformer_model import SwinTransformer


class SwinTransformerEncoder(SwinTransformer, EncoderMixin):
    def __init__(self, out_channels, depth=5, for_cls=False, gap=False, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        self.for_cls = for_cls
        self.gap=gap
        if self.gap:
            self.global_average_pooling = nn.AdaptiveAvgPool1d(1)
            self.norm_layer = nn.LayerNorm(self.num_features[-1]) #nn.AvgPool2d(kernel_size=8)

    def get_stages(self):
        return [nn.Identity()]

    def feature_forward(self, x):
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        if self.for_cls:
            for layer in self.layers:
                x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            return x_out
        else:
            outs = []
            for i in range(self.num_layers):
                layer = self.layers[i]
                x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    x_out = norm_layer(x_out)

                    out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                    outs.append(out)    
            return outs

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for stage in stages:
            x = stage(x)
            features.append(x)

       
        outs = self.feature_forward(x)
        if self.for_cls:
            if self.gap:
                out_cls = self.norm_layer(outs)
                out_cls = self.global_average_pooling(out_cls.transpose(1, 2))
                out_cls = torch.flatten(out_cls, 1)
                return out_cls
            else:
                return outs
        # Note: An additional interpolated feature to accommodate five-stage decoders,\
        # the additional feature will be ignored if a decoder with fewer stages is used.
        add_feature = F.interpolate(outs[0], scale_factor=2)
        features = features + [add_feature] + outs
        return features

    def load_state_dict(self, state_dict, **kwargs):

        new_state_dict = OrderedDict()

        if 'state_dict' in state_dict:
            _state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            _state_dict = state_dict['model']
        else:
            _state_dict = state_dict

        for k, v in _state_dict.items():
            if k.startswith('backbone.'):
                new_state_dict[k[9:]] = v
            else:
                new_state_dict[k] = v

        # Note: In swin seg model: `attn_mask` is no longer a class attribute for
        # multi-scale inputs; a norm layer is added for each output; the head layer
        # is removed.
        kwargs.update({'strict': False})
        super().load_state_dict(new_state_dict, **kwargs)


# https://github.com/microsoft/Swin-Transformer
new_settings = {
    "Swin-T": {
        "imagenet": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
        "ADE20k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth"
    },
    "Swin-S": {
        "imagenet": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
        "ADE20k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_small_patch4_window7_512x512.pth"
    },
    "Swin-B": {
        "imagenet": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth",
        "imagenet-22k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth",
        "ADE20k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_base_patch4_window7_512x512.pth",
        "geopile": "/nfs/ap/mnt/sxtn/cd/gfm_model/gfm.pth",
        "satlas": "/nfs/ap/mnt/sxtn/cd/satlas_model/sentinel2_swinb_si_rgb.pth",
        "satlas_ms": "/nfs/ap/mnt/sxtn/cd/satlas_model/sentinel2_swinb_si_ms.pth",
        "satlas_hr": "/nfs/ap/mnt/sxtn/cd/satlas_model/aerial_swinb_si.pth",
        "cmid": "/nfs/ap/mnt/sxtn/cd/cmid_model/CMID_Swin-B_bk_200ep"
    },
    "Swin-L": {
        "imagenet-22k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth"
    },

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

swin_transformer_encoders = {
    "Swin-T": {
        "encoder": SwinTransformerEncoder,
        "pretrained_settings": pretrained_settings["Swin-T"],
        "params": {
            "embed_dim": 96,
            "out_channels": (3, 96, 96, 192, 384, 768),
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 7,
            "ape": False,
            "drop_path_rate": 0.3,
            "patch_norm": True,
            "use_checkpoint": False
        }
    },
    "Swin-S": {
        "encoder": SwinTransformerEncoder,
        "pretrained_settings": pretrained_settings["Swin-S"],
        "params": {
            "embed_dim": 96,
            "out_channels": (3, 96, 96, 192, 384, 768),
            "depths": [2, 2, 18, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 7,
            "ape": False,
            "drop_path_rate": 0.3,
            "patch_norm": True,
            "use_checkpoint": False
        }
    },
    "Swin-B": {
        "encoder": SwinTransformerEncoder,
        "pretrained_settings": pretrained_settings["Swin-B"],
        "params": {
            "embed_dim": 128,
            "out_channels": (3, 128, 128, 256, 512, 1024),
            "depths": [2, 2, 18, 2],
            "num_heads": [4, 8, 16, 32],
            "window_size": 7,
            "ape": False,
            "drop_path_rate": 0.3,
            "patch_norm": True,
            "use_checkpoint": False
        }
    },
    "Swin-L": {
        "encoder": SwinTransformerEncoder,
        "pretrained_settings": pretrained_settings["Swin-L"],
        "params": {
            "embed_dim": 192,
            "out_channels": (3, 192, 192, 384, 768, 1536),
            "depths": [2, 2, 18, 2],
            "num_heads": [6, 12, 24, 48],
            "window_size": 7,
            "ape": False,
            "drop_path_rate": 0.3,
            "patch_norm": True,
            "use_checkpoint": False
        }
    }

}

if __name__ == "__main__":
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.randn(1, 3, 256, 256).to(device)

    model = SwinTransformerEncoder(2, window_size=8)
    # print(model)

    res = model.forward(input)
    for i in res:
        print(i.shape)
