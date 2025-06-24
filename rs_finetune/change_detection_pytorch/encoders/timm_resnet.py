from typing import Any

import timm
import torch
import torch.nn as nn


class TimmResnetEncoder(nn.Module):

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        feat_depth: int = 5,
        for_cls: bool = False,
        **kwargs,
    ):
        """
        Initialize the encoder.

        Args:
            name (str): Model name to load from `timm`.
            pretrained (bool): Load pretrained weights (default: True).
            in_channels (int): Number of input channels (default: 3 for RGB).
            feat_depth (int): Number of feature stages to extract (default: 5).
        """

        super().__init__()
        self.name = name

        self.model = timm.create_model(name, in_chans=in_channels, pretrained=pretrained)

        self.feat_depth = feat_depth
        self.for_cls = for_cls
        
    def forward(self, x: torch.Tensor):
        if self.for_cls:
            features = self.model.forward_features(x)
            return features
        
        features = self.model.forward_intermediates(x, intermediates_only=True)
        return features[-self.feat_depth:]


timm_resnet_encoders = {
    "timm_resnet50": {
        "encoder": TimmResnetEncoder,
        "params": {
            "name": "resnet50",
            "pretrained": True,
            "in_channels": 3,
            "feat_depth": 4,
            }
        }
    }

if __name__ == '__main__':
    model_params = timm_resnet_encoders['timm_resnet50']["params"]
    encoder = timm_resnet_encoders['timm_resnet50']["encoder"](
        **model_params
    )

    dummy_input = torch.randn(1, 3, 224, 224)
    features = encoder(dummy_input)