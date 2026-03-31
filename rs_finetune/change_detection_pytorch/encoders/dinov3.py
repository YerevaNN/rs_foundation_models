import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Any, Optional
from .vision_transformer import MultiLevelNeck
import random

class SharedChannelPatchConv(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        bias: bool,
        in_channels: int,
        device: torch.device,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size[0], kernel_size[1], device=device))
        self.bias = nn.Parameter(torch.empty(out_channels, device=device)) if bias else None
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        w_exp = self.weight.repeat(c, 1, 1, 1)
        y = F.conv2d(x, w_exp, bias=None, stride=self.stride, padding=self.padding, groups=c)
        y = y.view(b, c, self.weight.shape[0], y.shape[-2], y.shape[-1]).mean(dim=1)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y

def _channel_weight_variance(patch: nn.Module) -> float:
    if isinstance(patch, nn.Conv2d):
        return float(patch.weight.var(dim=1, unbiased=False).mean().item())
    if isinstance(patch, SharedChannelPatchConv):
        return 0.0
    w = getattr(patch, "weight", None)
    if w is None:
        return 0.0
    if w.dim() < 2 or w.shape[1] == 1:
        return 0.0
    return float(w.var(dim=1, unbiased=False).mean().item())

def _weights_exactly_shared(patch: nn.Module) -> bool:
    if isinstance(patch, SharedChannelPatchConv):
        return True
    w = getattr(patch, "weight", None)
    if w is None:
        return True
    if w.dim() < 2 or w.shape[1] <= 1:
        return True
    ref = w[:, :1, ...]
    return bool(torch.all(w == ref).item())

class DinoV3(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        output_indices: Optional[list[int]] = [2, 5, 8, 11],
        output_channels: Optional[list[int]] = [768, 768, 768, 768],
        for_cls: bool = False,
        enable_sample: bool = False,
        color_blind: bool = False,
        **kwargs: dict[str, Any],
    ):
        super().__init__()
        
        self.for_cls = for_cls
        self.in_channels = in_channels
        self.enable_sample = enable_sample
        self.color_blind = color_blind
        print("color_blind: ", color_blind)
        self.model_name = kwargs.get('model_name', "facebook/dinov3-vitb16-pretrain-lvd1689m")
        
        print(f"Loading pretrained DINOv3 model: {self.model_name}")
        self.dinov3 = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        
        if self.color_blind:
            pe = getattr(self.dinov3, "embeddings", None)
            if pe is not None and hasattr(pe, "patch_embeddings") and isinstance(pe.patch_embeddings, nn.Conv2d):
                old_conv = pe.patch_embeddings
                k = old_conv.kernel_size if isinstance(old_conv.kernel_size, tuple) else (old_conv.kernel_size, old_conv.kernel_size)
                s = old_conv.stride if isinstance(old_conv.stride, tuple) else (old_conv.stride, old_conv.stride)
                p = old_conv.padding if isinstance(old_conv.padding, tuple) else (old_conv.padding, old_conv.padding)
                mod = SharedChannelPatchConv(
                    out_channels=old_conv.out_channels,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    bias=(old_conv.bias is not None),
                    in_channels=old_conv.in_channels,
                    device=old_conv.weight.device,
                )
                w_avg = old_conv.weight.data.mean(dim=1, keepdim=True)
                mod.weight.data.copy_(w_avg)
                if old_conv.bias is not None:
                    mod.bias.data.copy_(old_conv.bias.data)
                pe.patch_embeddings = mod
        
        # Set up output configuration
        self.output_indices = output_indices
        self.output_channels = output_channels
        self.feat_dim = 768
        
        if not for_cls:
            self.neck = MultiLevelNeck(
                in_channels=self.output_channels, 
                out_channels=768, 
                scales=[4, 2, 1, 0.5]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor in NCHW format, got shape {tuple(x.shape)}")
        pe = getattr(self.dinov3, "embeddings", None)
        if pe is not None and hasattr(pe, "patch_embeddings"):
            var = _channel_weight_variance(pe.patch_embeddings)
            eq = _weights_exactly_shared(pe.patch_embeddings)
            # print(f"dinov3_patch_equal={eq} var={var} color_blind={self.color_blind}")
            
        if self.training and self.enable_sample:
            c = x.shape[1]
            c_new = random.randint(1, c)
            channels = random.sample(range(c), k=c_new)
            mask = x.new_zeros(c)
            for idx in channels:
                mask[idx] = 1.0
            x = x * mask.view(1, c, 1, 1)
        if self.for_cls:
            outputs = self.dinov3(x)
            cls_features = outputs.last_hidden_state[:, 0]
            return cls_features
        else:
            outputs = self.dinov3(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            features = []
            for idx in self.output_indices:
                if idx < len(hidden_states):
                    layer_feat = hidden_states[idx][:, 5:]
                    B, N, C = layer_feat.shape
                    
                    H = W = 14
                    
                    layer_feat = layer_feat.reshape(B, H, W, C).permute(0, 3, 1, 2)
                    features.append(layer_feat)
            
            features = self.neck(features)
            return features


dinov3_encoders = {
    "dinov3_vitb16": {
        "encoder": DinoV3,
        "params": {
            "in_channels": 3,
            "output_indices": [2, 5, 8, 11],
            "output_channels": [768, 768, 768, 768],
            "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m"
        }
    }
}
