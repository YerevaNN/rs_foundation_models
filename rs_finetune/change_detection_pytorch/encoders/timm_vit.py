from typing import Any, Optional
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import timm
import torch
import torch.nn as nn
from change_detection_pytorch.encoders.vision_transformer import MultiLevelNeck



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
        self.dinov3_weights_path = kwargs.get('dinov3_weights_path', None)
        
        # Load a timm model
        encoder_kwargs = dict(in_chans=in_channels, pretrained=pretrained)
        # encoder_kwargs = merge_kwargs_no_duplicates(encoder_kwargs, kwargs)
        self.model = timm.create_model(name, **encoder_kwargs)

        # Load DINOv3 weights if specified
        if self.dinov3_weights_path and os.path.exists(self.dinov3_weights_path):
            self._load_dinov3_weights(self.dinov3_weights_path)

        self.output_channels = (768, 768, 768, 768)
        
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

    def _load_dinov3_weights(self, weights_path: str):
        """Load DINOv3 weights from checkpoint file."""
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Map DINOv3 weight names to timm ViT names and handle special cases
            mapped_state_dict = {}
            qkv_weights = {}  # Store q, k, v weights to combine later
            qkv_biases = {}   # Store q, k, v biases to combine later
            
            for key, value in state_dict.items():
                mapped_key = self._map_dinov3_to_timm(key)
                if mapped_key:
                    if 'attn.qkv' in mapped_key:
                        # Store q, k, v weights separately to combine later
                        layer_num = key.split('.')[1]  # Extract layer number
                        if 'weight' in key:
                            if layer_num not in qkv_weights:
                                qkv_weights[layer_num] = {'q': None, 'k': None, 'v': None}
                            if 'q_proj' in key:
                                qkv_weights[layer_num]['q'] = value
                            elif 'k_proj' in key:
                                qkv_weights[layer_num]['k'] = value
                            elif 'v_proj' in key:
                                qkv_weights[layer_num]['v'] = value
                        elif 'bias' in key:
                            if layer_num not in qkv_biases:
                                qkv_biases[layer_num] = {'q': None, 'k': None, 'v': None}
                            if 'q_proj' in key:
                                qkv_biases[layer_num]['q'] = value
                            elif 'k_proj' in key:
                                qkv_biases[layer_num]['k'] = value
                            elif 'v_proj' in key:
                                qkv_biases[layer_num]['v'] = value
                    else:
                        mapped_state_dict[mapped_key] = value
                else:
                    print(f"Skipping unmapped DINOv3 key: {key}")
            
            # Combine q, k, v weights and biases into qkv format
            for layer_num in qkv_weights:
                if all(v is not None for v in qkv_weights[layer_num].values()):
                    # Concatenate q, k, v weights: [q, k, v] -> [qkv]
                    qkv_weight = torch.cat([
                        qkv_weights[layer_num]['q'],
                        qkv_weights[layer_num]['k'], 
                        qkv_weights[layer_num]['v']
                    ], dim=0)
                    mapped_state_dict[f'blocks.{layer_num}.attn.qkv.weight'] = qkv_weight
                    print(f"Combined qkv weights for layer {layer_num}")
                
                if all(v is not None for v in qkv_biases[layer_num].values()):
                    # Concatenate q, k, v biases
                    qkv_bias = torch.cat([
                        qkv_biases[layer_num]['q'],
                        qkv_biases[layer_num]['k'],
                        qkv_biases[layer_num]['v']
                    ], dim=0)
                    mapped_state_dict[f'blocks.{layer_num}.attn.qkv.bias'] = qkv_bias
                    print(f"Combined qkv biases for layer {layer_num}")
            
            # Filter out non-matching keys and load weights
            model_state_dict = self.model.state_dict()
            filtered_state_dict = {}
            
            for key, value in mapped_state_dict.items():
                if key in model_state_dict and value.shape == model_state_dict[key].shape:
                    filtered_state_dict[key] = value
            
            if filtered_state_dict:
                self.model.load_state_dict(filtered_state_dict, strict=False)
                print(f"Loaded {len(filtered_state_dict)} DINOv3 layers from {weights_path}")
                
                # Print first number from each loaded layer
                print("\nFirst number from each DINOv3 layer:")
                for name, param in filtered_state_dict.items():
                    if param.numel() > 0:
                        first_value = param.flatten()[0].item()
                        print(f"Name: {name}, Shape: {param.shape}, First value: {first_value}")
                print()
                
                # Check which timm ViT layers didn't get DINOv3 weights
                model_state_dict = self.model.state_dict()
                unloaded_layers = []
                for key in model_state_dict.keys():
                    if key not in filtered_state_dict:
                        unloaded_layers.append(key)
                
                if unloaded_layers:
                    print(f"\n⚠️  {len(unloaded_layers)} timm ViT layers did NOT get DINOv3 weights:")
                    for layer in unloaded_layers[:20]:  # Show first 20
                        print(f"  - {layer}")
                    if len(unloaded_layers) > 20:
                        print(f"  ... and {len(unloaded_layers) - 20} more")
                    print()
                else:
                    print("✅ All timm ViT layers got DINOv3 weights!")
                    print()
                
                # Check which DINOv3 layers don't have matches in timm ViT
                original_dinov3_keys = set(state_dict.keys())
                mapped_dinov3_keys = set(mapped_state_dict.keys())
                unmapped_dinov3_keys = original_dinov3_keys - set(key for key in state_dict.keys() if self._map_dinov3_to_timm(key))
                
                if unmapped_dinov3_keys:
                    print(f"\n⚠️  {len(unmapped_dinov3_keys)} DINOv3 layers did NOT have matches in timm ViT:")
                    for layer in sorted(list(unmapped_dinov3_keys))[:20]:  # Show first 20
                        print(f"  - {layer}")
                    if len(unmapped_dinov3_keys) > 20:
                        print(f"  ... and {len(unmapped_dinov3_keys) - 20} more")
                    print()
                else:
                    print("✅ All DINOv3 layers had matches in timm ViT!")
                    print()
                
                # Summary statistics
                print(f"\n📊 Weight Loading Summary:")
                print(f"  - DINOv3 layers total: {len(original_dinov3_keys)}")
                print(f"  - Successfully mapped: {len(mapped_state_dict)}")
                print(f"  - Successfully loaded: {len(filtered_state_dict)}")
                print(f"  - timm ViT layers total: {len(model_state_dict)}")
                print(f"  - timm ViT layers with weights: {len(filtered_state_dict)}")
                print(f"  - timm ViT layers without weights: {len(unloaded_layers)}")
                print(f"  - Coverage: {len(filtered_state_dict)/len(model_state_dict)*100:.1f}%")
                print()
            else:
                print(f"No compatible DINOv3 weights found in {weights_path}")
                
        except Exception as e:
            print(f"Error loading DINOv3 weights from {weights_path}: {e}")

    def _map_dinov3_to_timm(self, dinov3_key: str) -> str:
        """Map DINOv3 weight names to timm ViT names."""
        # Handle special cases first
        if dinov3_key == 'embeddings.cls_token':
            return 'cls_token'
        elif dinov3_key == 'embeddings.mask_token':
            return 'mask_token'
        elif dinov3_key == 'embeddings.register_tokens':
            return 'register_tokens'
        elif dinov3_key == 'embeddings.patch_embeddings.weight':
            return 'patch_embed.proj.weight'
        elif dinov3_key == 'embeddings.patch_embeddings.bias':
            return 'patch_embed.proj.bias'
        elif dinov3_key == 'embeddings.position_embeddings':
            return 'pos_embed'
        elif dinov3_key == 'norm.weight':
            return 'norm.weight'
        elif dinov3_key == 'norm.bias':
            return 'norm.bias'
        
        # Handle transformer layers (layer.0, layer.1, etc.)
        if dinov3_key.startswith('layer.'):
            # Extract layer number and rest of the key
            parts = dinov3_key.split('.', 2)  # Split into ['layer', '0', 'rest...']
            if len(parts) >= 3:
                layer_num = parts[1]
                rest = parts[2]
                
                # Map the rest of the key
                if rest == 'norm1.weight':
                    return f'blocks.{layer_num}.norm1.weight'
                elif rest == 'norm1.bias':
                    return f'blocks.{layer_num}.norm1.bias'
                elif rest == 'norm2.weight':
                    return f'blocks.{layer_num}.norm2.weight'
                elif rest == 'norm2.bias':
                    return f'blocks.{layer_num}.norm2.bias'
                elif rest == 'attention.q_proj.weight':
                    return f'blocks.{layer_num}.attn.qkv.weight'  # Will be combined later
                elif rest == 'attention.q_proj.bias':
                    return f'blocks.{layer_num}.attn.qkv.bias'    # Will be combined later
                elif rest == 'attention.k_proj.weight':
                    return f'blocks.{layer_num}.attn.qkv.weight'  # Will be combined later
                elif rest == 'attention.k_proj.bias':
                    return f'blocks.{layer_num}.attn.qkv.bias'    # Will be combined later
                elif rest == 'attention.v_proj.weight':
                    return f'blocks.{layer_num}.attn.qkv.weight'  # Will be combined later
                elif rest == 'attention.v_proj.bias':
                    return f'blocks.{layer_num}.attn.qkv.bias'    # Will be combined later
                elif rest == 'attention.o_proj.weight':
                    return f'blocks.{layer_num}.attn.proj.weight'
                elif rest == 'attention.o_proj.bias':
                    return f'blocks.{layer_num}.attn.proj.bias'
                elif rest == 'mlp.up_proj.weight':
                    return f'blocks.{layer_num}.mlp.fc1.weight'
                elif rest == 'mlp.up_proj.bias':
                    return f'blocks.{layer_num}.mlp.fc1.bias'
                elif rest == 'mlp.down_proj.weight':
                    return f'blocks.{layer_num}.mlp.fc2.weight'
                elif rest == 'mlp.down_proj.bias':
                    return f'blocks.{layer_num}.mlp.fc2.bias'
                elif rest == 'layer_scale1.lambda1':
                    return f'blocks.{layer_num}.ls1.gamma'
                elif rest == 'layer_scale2.lambda1':
                    return f'blocks.{layer_num}.ls2.gamma'
        
        # If no mapping found, return None to skip this weight
        return None

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
        },
    "dinov3_vitb16": {
        "encoder": TimmViTEncoder,
        "params": {
            "name": "vit_base_patch16_224",
            "output_indices": (2, 5, 8, 11),
            "pretrained": False,
            "in_channels": 3,
            "feat_depth": 4,
            "dinov3_weights_path": "/nfs/h100/raid/rs/dinov3_vitb_checkpoint.pth"
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
    


