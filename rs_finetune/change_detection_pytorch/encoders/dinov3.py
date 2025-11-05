import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Any, Optional
from .vision_transformer import MultiLevelNeck

class DinoV3(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        output_indices: Optional[list[int]] = [2, 5, 8, 11],
        output_channels: Optional[list[int]] = [768, 768, 768, 768],
        for_cls: bool = False,
        **kwargs: dict[str, Any],
    ):
        super().__init__()
        
        self.for_cls = for_cls
        self.model_name = kwargs.get('model_name', "facebook/dinov3-vitb16-pretrain-lvd1689m")
        
        print(f"Loading pretrained DINOv3 model: {self.model_name}")
        self.dinov3 = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        
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

    def forward(self, x: torch.Tensor):
        if self.for_cls:
            # For classification: return CLS token only
            outputs = self.dinov3(x)
            cls_features = outputs.last_hidden_state[:, 0]
            return cls_features
        else:
            # For segmentation/change detection: return multi-layer features
            outputs = self.dinov3(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Extract features from specified layers
            features = []
            for idx in self.output_indices:
                if idx < len(hidden_states):
                    # Skip CLS token (1) + register tokens (4) = 5 tokens
                    layer_feat = hidden_states[idx][:, 5:]  # Skip CLS + 4 register tokens
                    B, N, C = layer_feat.shape
                    
                    # For 224x224 input with 16x16 patches: 224/16 = 14
                    H = W = 14
                    
                    # Reshape to spatial dimensions
                    layer_feat = layer_feat.reshape(B, H, W, C).permute(0, 3, 1, 2)
                    features.append(layer_feat)
            
            # Apply neck for multi-level feature fusion
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