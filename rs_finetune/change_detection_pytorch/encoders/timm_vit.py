from typing import Any, Optional
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import timm
import torch
import torch.nn as nn
from change_detection_pytorch.encoders.vision_transformer import MultiLevelNeck


def merge_kwargs_no_duplicates(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    duplicates = a.keys() & b.keys()
    if duplicates:
        raise ValueError(f"'{duplicates}' already specified internally")

    return a | b

def sample_block_indices_uniformly(n: int, total_num_blocks: int) -> list[int]:
    """
    Sample N block indices uniformly from the total number of blocks.
    """
    return [
        int(total_num_blocks / n * block_depth) - 1 for block_depth in range(1, n + 1)
    ]
    
def validate_output_indices(
    output_indices: list[int], model_num_blocks: int, feat_depth: int
):
    """
    Validate the output indices are within the valid range of the model and the
    length of the output indices is equal to the feat_depth of the encoder.
    """
    for output_index in output_indices:
        if output_index < -model_num_blocks or output_index >= model_num_blocks:
            raise ValueError(
                f"Output indices for feature extraction should be in range "
                f"[-{model_num_blocks}, {model_num_blocks}), because the model has {model_num_blocks} blocks, "
                f"got index = {output_index}."
            )


def preprocess_output_indices(
    output_indices: Optional[list[int]], model_num_blocks: int, feat_depth: int
) -> list[int]:
    """
    Preprocess the output indices for the encoder.
    """

    # Refine encoder output indices
    if output_indices is None:
        output_indices = sample_block_indices_uniformly(feat_depth, model_num_blocks)
    elif not isinstance(output_indices, (list, tuple)):
        raise ValueError(
            f"`output_indices` for encoder should be a list/tuple/None, got {type(output_indices)}"
        )
    validate_output_indices(output_indices, model_num_blocks, feat_depth)

    return output_indices


class TimmViTEncoder(nn.Module):

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        feat_depth: int = 4,
        output_indices: Optional[list[int]] = None,
        for_cls: bool = False,
        **kwargs: dict[str, Any],
    ):
        """
        Args:
            name (str): ViT model name to load from `timm`.
            pretrained (bool): Load pretrained weights (default: True).
            in_channels (int): Number of input channels (default: 3 for RGB).
            feat_depth (int): Number of feature stages to extract (default: 4).
            output_indices (Optional[list[int] | int]): Indices of blocks in the model to be used for feature extraction.
            **kwargs: Additional arguments passed to `timm.create_model`.
        """
        super().__init__()

        if isinstance(output_indices, (list, tuple)) and len(output_indices) != feat_depth:
            raise ValueError(
                f"Length of output indices for feature extraction should be equal to the feat_depth of the encoder "
                f"architecture, got output indices length - {len(output_indices)}, encoder feat_depth - {feat_depth}"
            )

        self.name = name
        self.for_cls = for_cls
        # Load a timm model
        encoder_kwargs = dict(in_chans=in_channels, pretrained=pretrained)
        encoder_kwargs = merge_kwargs_no_duplicates(encoder_kwargs, kwargs)
        self.model = timm.create_model(name, **encoder_kwargs)

        if not hasattr(self.model, "forward_intermediates"):
            raise ValueError(
                f"Encoder `{name}` does not support `forward_intermediates` for feature extraction. "
                f"Please update `timm` or use another encoder."
            )

        # Get all the necessary information about the model
        feature_info = self.model.feature_info

        # import pdb; pdb.set_trace()
        # Additional checks
        model_num_blocks = len(feature_info)
        if feat_depth > model_num_blocks:
            raise ValueError(
                f"feat_depth of the encoder cannot exceed the number of blocks in the model "
                f"got {feat_depth} feat_depth, model has {model_num_blocks} blocks"
            )

        # Preprocess the output indices, uniformly sample from model_num_blocks if None
        output_indices = preprocess_output_indices(
            output_indices, model_num_blocks, feat_depth
        )
        
        # Private attributes for model forward
        self._has_cls_token = getattr(self.model, "has_cls_token", False)
        self.out_idx = output_indices

        # Public attributes
        self.out_channels = [feature_info[i]["num_chs"] for i in output_indices]
        self.input_size = self.model.pretrained_cfg.get("input_size", None)
        self.is_fixed_input_size = self.model.pretrained_cfg.get(
            "fixed_input_size", False
        )
        
        if not for_cls:
            self.neck = MultiLevelNeck(in_channels=[768, 768, 768, 768], out_channels=768, scales=[4, 2, 1, 0.5])

    def forward(self, x: torch.Tensor):
        if self.for_cls:
            output = self.model.forward_features(x)
            return output[:, 0]
        
        features = self.model.forward_intermediates(
            x,
            indices=self.out_idx,
            intermediates_only=True,
        )
        features = self.neck(features)
        return features


timm_vit_encoders = {
    "timm_vit-b": {
        "encoder": TimmViTEncoder,
        "params": {
            "name": "vit_base_patch16_224",
            "output_indices": (2, 5, 8, 11),
            "pretrained": True,
            "in_channels": 3,
            "feat_depth": 4,
            }

        }
    }
  
if __name__ == '__main__':
    model_params = timm_vit_encoders['timm_vit-b']["params"]
    encoder = timm_vit_encoders['timm_vit-b']["encoder"](
        for_cls=False,
        **model_params
    )
    print(f"Selected output indices: {encoder.out_channels}")

    dummy_input = torch.randn(1, 3, 224, 224)
    features = encoder(dummy_input)    
    


