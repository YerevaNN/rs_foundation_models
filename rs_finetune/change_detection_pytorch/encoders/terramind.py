import torch
import torch.nn as nn
import math
from typing import List, Dict, Union, Optional
from functools import partial

from terratorch import BACKBONE_REGISTRY
from ._base import EncoderMixin
from .vision_transformer import MultiLevelNeck


# Mapping from common Sentinel band names to TerraMind's expected band names
TERRAMIND_BAND_MAPPING = {
    # Sentinel-2 bands
    'B02': 'BLUE',
    'B03': 'GREEN',
    'B04': 'RED',
    'B08': 'NIR',  # Standard NIR
    'B8A': 'NIR_NARROW',  # Narrow NIR (typically used for NIR_NARROW)
    'B11': 'SWIR_1',
    'B12': 'SWIR_2',
    # Sentinel-1 bands
    'VV': 'VV',
    'VH': 'VH',
    # Additional Sentinel-2 bands (optional mappings)
    'B01': 'COASTAL_AEROSOL',  # Not commonly used in TerraMind
    'B05': 'RED_EDGE_1',  # Not commonly used in TerraMind
    'B06': 'RED_EDGE_2',  # Not commonly used in TerraMind
    'B07': 'RED_EDGE_3',  # Not commonly used in TerraMind
    'B09': 'WATER_VAPOR',  # Not commonly used in TerraMind
}


def map_bands_to_terramind(bands: Optional[Dict[str, List[str]]]) -> Optional[Dict[str, List[str]]]:
    """
    Convert common Sentinel band names (B02, B03, etc.) to TerraMind's expected band names
    (BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2).
    
    Args:
        bands: Dict mapping modality names to lists of band names.
               Example: {'S2L2A': ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']}
                       or {'S2L2A': ['BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2']}
    
    Returns:
        Dict with band names mapped to TerraMind's expected format.
        Example: {'S2L2A': ['BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2']}
    """
    if bands is None:
        return None
    
    # Check if bands is already a dict with keys (modality names)
    print("bands: ", bands)
    if isinstance(bands, dict):
        # Check if band names are already in TerraMind format
        # TerraMind format: uppercase with underscores (e.g., 'BLUE', 'GREEN', 'NIR_NARROW')
        all_already_mapped = True
        for modality, band_list in bands.items():
            if not isinstance(band_list, list):
                all_already_mapped = False
                break
            for band in band_list:
                # Check if band is already in TerraMind format
                # TerraMind format bands are in TERRAMIND_BAND_MAPPING.values()
                if band not in TERRAMIND_BAND_MAPPING.values() and band in TERRAMIND_BAND_MAPPING:
                    # Band is in mapping keys (e.g., 'B02', 'B03'), needs mapping
                    all_already_mapped = False
                    break
            if not all_already_mapped:
                break
        
        # If all bands are already in TerraMind format, return as-is
        if all_already_mapped:
            return bands
        
        # Otherwise, do the mapping
    mapped_bands = {}
    for modality, band_list in bands.items():
            if not isinstance(band_list, list):
                mapped_bands[modality] = band_list
                continue
    mapped_band_list = []
    for band in band_list:
                # Check if band is already in TerraMind format
        if band in TERRAMIND_BAND_MAPPING.values():
            mapped_band_list.append(band)
        elif band in TERRAMIND_BAND_MAPPING:
            mapped_band_list.append(TERRAMIND_BAND_MAPPING[band])
        else:
            # If band not in mapping, keep as is (might be a custom band name)
            mapped_band_list.append(band)
            mapped_bands[modality] = mapped_band_list
    
    return mapped_bands
    
    # Legacy support: if bands is a list, treat as S2L2A bands
    if isinstance(bands, list):
        mapped_band_list = []
        for band in bands:
            if band in TERRAMIND_BAND_MAPPING.values():
                mapped_band_list.append(band)
            elif band in TERRAMIND_BAND_MAPPING:
                mapped_band_list.append(TERRAMIND_BAND_MAPPING[band])
            else:
                mapped_band_list.append(band)
        return {'S2L2A': mapped_band_list}
    
    return bands


class TerraMindEncoder(nn.Module, EncoderMixin):
    """TerraMind encoder wrapper for change_detection_pytorch
    
    Wraps TerraMindViT from terratorch to work with the existing encoder interface.
    Supports both classification (returns pooled features) and segmentation (returns multi-level features).
    
    For segmentation, this follows the pattern from:
    https://github.com/IBM/terramind/blob/main/notebooks/terramind_v1_small_sen1floods11.ipynb
    which uses SelectIndices, ReshapeTokensToImage, and LearnedInterpolateToPyramidal.
    """
    
    def __init__(
        self,
        model_name: str = 'terramind_v1_small',
        pretrained: bool = True,
        modalities: List[str] = ['S2L2A', 'S1GRD'],
        img_size: int = 224,
        patch_size: int = 16,
        for_cls: bool = False,
        out_idx: Optional[List[int]] = None,
        depth: int = 5,
        scales: List[float] = [4, 2, 1, 0.5],
        bands: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        """
        Args:
            model_name: TerraMind model variant (e.g., 'terramind_v1_small', 'terramind_v1_base')
            pretrained: Whether to load pretrained weights
            modalities: List of modality names (e.g., ['S2L2A', 'S1GRD'])
            img_size: Input image size
            patch_size: Patch size (typically 16)
            for_cls: If True, used for classification (returns pooled features). If False, for segmentation.
            out_idx: Indices of transformer layers to extract features from. 
                     Default: [2, 5, 8, 11] for base, [5, 11, 17, 23] for large
            depth: Number of feature levels (unused if out_idx is specified)
            scales: Scale factors for MultiLevelNeck (only used for segmentation)
            bands: Dict mapping modality names to band names. 
                   Example: {'S2L2A': ['BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2']}
            **kwargs: Additional arguments passed to TerraMindViT
        """
        super().__init__()
        
        self.model_name = model_name
        self.for_cls = for_cls
        self.modalities = modalities
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Map bands from common Sentinel format to TerraMind format.
        # Supports either:
        #   * dict of modality -> list of band names
        #   * list of band names, treated as S2L2A bands
        if bands is not None:
            # Normalize to dict form
            if isinstance(bands, dict):
                bands_dict = bands
            else:
                # Accept list/tuple/str as shorthand for S2L2A
                if isinstance(bands, (list, tuple)):
                    band_list = list(bands)
                else:
                    band_list = [bands]
                bands_dict = {"S2L2A": band_list}

            mapped_bands: Dict[str, List[str]] = {}
            for modality, band_list in bands_dict.items():
                if not isinstance(band_list, (list, tuple)):
                    mapped_bands[modality] = band_list  # type: ignore[assignment]
                    continue
                new_list: List[str] = []
                for b in band_list:
                    # If already a TerraMind band name (e.g. 'BLUE', 'VV'), keep as-is
                    if b in TERRAMIND_BAND_MAPPING.values():
                        new_list.append(b)
                    # If it's a known Sentinel code (e.g. 'B02'), map it
                    elif b in TERRAMIND_BAND_MAPPING:
                        new_list.append(TERRAMIND_BAND_MAPPING[b])
                    else:
                        # Unknown string: pass through unchanged
                        new_list.append(b)
                mapped_bands[modality] = new_list

            self.bands = mapped_bands
        else:
            self.bands = None
        # Build TerraMind model
     
        terramind_kwargs = {
            'pretrained': pretrained,
            'modalities': modalities,
            'img_size': img_size,
            'patch_size': patch_size,
        }
        # Add bands parameter if provided (already mapped to TerraMind format)
        if self.bands is not None:
            terramind_kwargs['bands'] = self.bands
        
      
        self.model = BACKBONE_REGISTRY.build(model_name, **terramind_kwargs)
        
        # Get model dimensions from out_channels or default based on model name
        if hasattr(self.model, 'out_channels') and isinstance(self.model.out_channels, list) and len(self.model.out_channels) > 0:
            self.embed_dim = self.model.out_channels[0]
        else:
            # Default dimensions based on model name
            if 'tiny' in model_name:
                self.embed_dim = 192
            elif 'small' in model_name:
                self.embed_dim = 384
            elif 'base' in model_name:
                self.embed_dim = 768
            elif 'large' in model_name:
                self.embed_dim = 1024
            else:
                self.embed_dim = 384  # Default to small
        
        # Set output indices based on model depth
        encoder_depth = len(self.model.encoder) if hasattr(self.model, 'encoder') else 12
        if out_idx is None:
            if encoder_depth == 24:  # large
                out_idx = [5, 11, 17, 23]
            else:  # tiny, small, base (12 layers)
                out_idx = [2, 5, 8, 11]
        
        self.out_idx = out_idx
        self._depth = len(out_idx) - 1
        
        # Set output channels based on merge method
        # TerraMindViT.out_channels already accounts for merge_method (concat or mean)
        if hasattr(self.model, 'out_channels') and isinstance(self.model.out_channels, list) and len(self.model.out_channels) > 0:
            out_ch = self.model.out_channels[0]
        else:
            # Fallback: check merge method
            if hasattr(self.model, 'merge_method') and self.model.merge_method == 'concat':
                # If concatenating modalities, channels = embed_dim * num_modalities
                num_modalities = len(modalities) if isinstance(modalities, list) else 1
                out_ch = self.embed_dim * num_modalities
            else:
                out_ch = self.embed_dim
        
        # Output channels for each feature level (same for all levels)
        self._out_channels = [out_ch] * (self._depth + 1)
        self._in_channels = 3  # Default, but TerraMind handles modalities differently
        
        # For segmentation, add neck for multi-level features
        if not for_cls:
            self.neck = MultiLevelNeck(
                in_channels=[out_ch] * len(out_idx),
                out_channels=out_ch,
                scales=scales
            )
        self.output_channels = tuple(self._out_channels)
        print("bands: ", self.bands)
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W) or dict of modality tensors {modality_name: tensor}
            **kwargs: Additional arguments
        
        Returns:
            - For classification: pooled features (B, embed_dim)
            - For segmentation: list of feature maps (B, C, H', W')
        """
        # Convert single tensor to dict format if needed
        if isinstance(x, torch.Tensor):
            # If single tensor, use first modality
            if len(self.modalities) == 1:
                x = {self.modalities[0]: x}
            else:
                # Try to split tensor across modalities
                # This is a fallback - ideally inputs should be provided as dict
                total_channels = x.shape[1]
                split_channels = total_channels // len(self.modalities)
                modality_inputs = {}
                for i, mod in enumerate(self.modalities):
                    start_ch = i * split_channels
                    end_ch = (i + 1) * split_channels if i < len(self.modalities) - 1 else total_channels
                    modality_inputs[mod] = x[:, start_ch:end_ch, :, :]
                x = modality_inputs
        
        # Forward through TerraMind model
        # TerraMindViT expects dict input with modality keys
        outputs = self.model(x)

        # outputs is a list of transformer layer outputs (B, L, D) where L is number of tokens
        if self.for_cls:
            # For classification: return pooled features (mean pool over tokens)
            # TerraMind doesn't have cls token, so we use mean pooling
            last_output = outputs[-1]  # Last layer output
            # Mean pool over sequence dimension (excluding any special tokens if present)
            pooled = last_output.mean(dim=1)  # (B, D)
            return pooled
        else:
            # For segmentation: extract features at specified indices and reshape to spatial
            features = []
            
            # Check if register tokens are present (they would be at the beginning)
            num_register_tokens = 0
            if hasattr(self.model, 'num_register_tokens') and self.model.num_register_tokens > 0:
                num_register_tokens = self.model.num_register_tokens
            
            # Calculate spatial dimensions from img_size and patch_size
            hw = self.img_size // self.patch_size
            for idx in self.out_idx:
                if idx < len(outputs):
                    # outputs[idx] shape: (B, L, D) where L = num_patches (or num_patches + register_tokens)
                    token_features = outputs[idx]
                    
                    # Remove register tokens if present
                    if num_register_tokens > 0:
                        token_features = token_features[:, num_register_tokens:, :]
                    
                    # Calculate expected number of patches
                    expected_patches = hw * hw
                    actual_patches = token_features.shape[1]
                    
                    if actual_patches != expected_patches:
                        # Try to infer spatial dimensions from number of patches
                        hw_actual = int(math.sqrt(actual_patches))
                        if hw_actual * hw_actual == actual_patches:
                            hw = hw_actual
                        else:
                            # If not a perfect square, use the stored hw value
                            # and take only the expected number of tokens
                            token_features = token_features[:, :expected_patches, :]
                    
                    # Reshape from (B, L, D) to (B, H, W, D) then to (B, D, H, W)
                    feature_map = token_features.reshape(token_features.shape[0], hw, hw, token_features.shape[2])
                    feature_map = feature_map.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
                    features.append(feature_map)
            # Pass through neck for multi-level features
            if len(features) > 0 and hasattr(self, 'neck'):
                features = self.neck(tuple(features))
                return list(features) if isinstance(features, tuple) else features
            
            return features
    
    def get_stages(self):
        """Required by EncoderMixin but not used for ViT-based encoders"""
        return []
    
    def set_in_channels(self, in_channels, pretrained=True):
        """TerraMind handles input channels through modalities, so this is a no-op"""
        # TerraMind processes modalities separately, so we don't need to modify input channels
        pass


# TerraMind encoder registry entries
terramind_encoders = {
    "terramind_v1_tiny": {
        "encoder": TerraMindEncoder,
        "pretrained_settings": {},  # Pretrained handled by terratorch
        "params": {
            "model_name": "terramind_v1_tiny",
            "modalities": ["S2L2A", "S1GRD"],
            "img_size": 224,
            "patch_size": 16,
            "out_idx": [2, 5, 8, 11],
            "depth": 4,
            "out_channels": (192, 192, 192, 192),
            "bands": None,  # Can be overridden, e.g., {'S2L2A': ['BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2']}
        }
    },
    "terramind_v1_small": {
        "encoder": TerraMindEncoder,
        "pretrained_settings": {},
        "params": {
            "model_name": "terramind_v1_small",
            "modalities": ["S2L2A", "S1GRD"],
            "img_size": 224,
            "patch_size": 16,
            "out_idx": [2, 5, 8, 11],
            "depth": 4,
            "out_channels": (384, 384, 384, 384),
            "bands": None,  # Can be overridden, e.g., {'S2L2A': ['BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2']}
        }
    },
    "terramind_v1_base": {
        "encoder": TerraMindEncoder,
        "pretrained_settings": {},
        "params": {
            "model_name": "terramind_v1_base",
            "modalities": ["S2L2A"],
            "img_size": 224,
            "patch_size": 16,
            "out_idx": [2, 5, 8, 11],
            "depth": 4,
            "out_channels": (768, 768, 768, 768),
            "bands": None,  # Only RGB bands from Sentinel-2
        }
    },
    "terramind_v1_large": {
        "encoder": TerraMindEncoder,
        "pretrained_settings": {},
        "params": {
            "model_name": "terramind_v1_large",
            "modalities": ["S2L2A", "S1GRD"],
            "img_size": 224,
            "patch_size": 16,
            "out_idx": [5, 11, 17, 23],
            "depth": 4,
            "out_channels": (1024, 1024, 1024, 1024),
            "bands": None,  # Can be overridden, e.g., {'S2L2A': ['BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2']}
        }
    }, 
}
