"""
Utility functions to adapt TerraMind weights from RGB to RGBN (or other band configurations)
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple


def adapt_terramind_projection_weights(
    rgb_weight: torch.Tensor,
    rgbn_weight: torch.Tensor,
    patch_size: int = 16
) -> torch.Tensor:
    """
    Adapt TerraMind projection layer weights from RGB to RGBN.
    
    The projection layer maps from patch features to embedding dimension.
    With RGB: input is patch_size^2 * 3 = 768 features per patch
    With RGBN: input is patch_size^2 * 4 = 1024 features per patch
    
    Args:
        rgb_weight: RGB projection weight [out_features, in_features]
        rgbn_weight: RGBN projection weight [out_features, in_features] 
        patch_size: Patch size (default 16)
    
    Returns:
        Adapted weight tensor with same shape as rgbn_weight
    """
    out_rgb, in_rgb = rgb_weight.shape
    out_rgbn, in_rgbn = rgbn_weight.shape
    
    # Expected dimensions
    expected_rgb = patch_size * patch_size * 3  # 16*16*3 = 768
    expected_rgbn = patch_size * patch_size * 4  # 16*16*4 = 1024
    print(f"RGB shape: {rgb_weight.shape}, RGBN shape: {rgbn_weight.shape}")
    print(f"Expected RGB: {expected_rgb}, Expected RGBN: {expected_rgbn}")
    print(f"In RGB: {in_rgb}, In RGBN: {in_rgbn}")
    print(f"Out RGB: {out_rgb}, Out RGBN: {out_rgbn}")
    # Handle case where RGB model has 256 input (might be a different architecture)
    # and RGBN needs 1024
    
    # Standard case: RGB 768 -> RGBN 1024
    if in_rgb == expected_rgb and in_rgbn == expected_rgbn:
        new_weight = rgbn_weight.clone()
        
        # Reshape to spatial: [out, 16, 16, channels]
        rgb_reshaped = rgb_weight.view(out_rgb, patch_size, patch_size, 3)
        rgbn_reshaped = new_weight.view(out_rgbn, patch_size, patch_size, 4)
        
        # Copy RGB channels (first 3)
        rgbn_reshaped[:, :, :, :3] = rgb_reshaped
        
        # Initialize NIR (4th channel) with mean of RGB
        rgb_mean = rgb_reshaped.mean(dim=3, keepdim=True)  # [out, 16, 16, 1]
        rgbn_reshaped[:, :, :, 3:4] = rgb_mean
        
        # Reshape back to [out, 1024]
        new_weight = rgbn_reshaped.view(out_rgbn, in_rgbn)
        
        return new_weight
    
    # Fallback: if shapes don't match expected, try simple expansion
    if in_rgb < in_rgbn and out_rgb == out_rgbn:
        new_weight = rgbn_weight.clone()
        # Copy RGB portion
        new_weight[:, :in_rgb] = rgb_weight
        # Initialize remaining with mean of RGB
        rgb_mean = rgb_weight.mean(dim=1, keepdim=True)  # [out, 1]
        new_weight[:, in_rgb:] = rgb_mean.expand(-1, in_rgbn - in_rgb)
        return new_weight
    
    # If we can't adapt, return the target shape with mean initialization
    new_weight = rgbn_weight.clone()
    rgb_mean = rgb_weight.mean()
    new_weight[:] = rgb_mean
    return new_weight


def adapt_terramind_state_dict(
    rgb_state_dict: Dict[str, torch.Tensor],
    rgbn_model: nn.Module,
    patch_size: int = 16
) -> Dict[str, torch.Tensor]:
    """
    Adapt a TerraMind RGB state_dict to work with an RGBN model.
    
    This function:
    1. Copies all matching weights
    2. Adapts projection layer weights from RGB (3 bands) to RGBN (4 bands)
       by copying RGB weights and initializing NIR band with mean of RGB
    
    Args:
        rgb_state_dict: State dict from RGB-trained TerraMind model
        rgbn_model: RGBN TerraMind model (target)
        patch_size: Patch size used in the model (default 16)
    
    Returns:
        Adapted state dict that can be loaded into rgbn_model
    """
    adapted = {}
    target_sd = rgbn_model.state_dict()
    
    print("Adapting TerraMind weights from RGB to RGBN...")
    print("-" * 80)
    
    # Find projection layer keys
    proj_keys = [k for k in target_sd.keys() if 'proj.weight' in k and 'encoder_embeddings' in k]
    
    for key in target_sd.keys():
        if key not in rgb_state_dict:
            # Key not in source, skip (will be missing in load_state_dict)
            continue
        
        src_tensor = rgb_state_dict[key]
        dst_tensor = target_sd[key]
        
        if not isinstance(src_tensor, torch.Tensor) or not isinstance(dst_tensor, torch.Tensor):
            adapted[key] = src_tensor
            continue
        
        # Check if this is a projection layer that needs adaptation
        if key in proj_keys:
            if src_tensor.shape != dst_tensor.shape:
                print(f"\nAdapting projection layer: {key}")
                print(f"  RGB shape:  {src_tensor.shape}")
                print(f"  RGBN shape: {dst_tensor.shape}")
                
                adapted_weight = adapt_terramind_projection_weights(
                    src_tensor, dst_tensor, patch_size
                )
                adapted[key] = adapted_weight
                print(f"  ✓ Adapted successfully")
            else:
                # Same shape, just copy
                adapted[key] = src_tensor
        else:
            # Not a projection layer, copy if shapes match
            if src_tensor.shape == dst_tensor.shape:
                adapted[key] = src_tensor
            else:
                # Shape mismatch in non-projection layer - skip
                print(f"  ⚠ Shape mismatch (non-projection): {key}")
                print(f"    RGB: {src_tensor.shape}, RGBN: {dst_tensor.shape}")
    
    print("-" * 80)
    print(f"Adaptation complete. {len(adapted)}/{len(target_sd)} keys adapted.")
    
    return adapted


def replace_terramind_projection_layer(
    model: nn.Module,
    num_bands: int,
    patch_size: int = 16,
    embed_dim: int = 768
) -> nn.Module:
    """
    Replace TerraMind's projection layer to handle different number of bands.
    
    The projection layer needs to have input_features = patch_size^2 * num_bands
    
    Args:
        model: TerraMind model
        num_bands: Number of bands (3 for RGB, 4 for RGBN, etc.)
        patch_size: Patch size (default 16)
        embed_dim: Embedding dimension (default 768 for base)
    
    Returns:
        Model with replaced projection layer
    """
    expected_in_features = patch_size * patch_size * num_bands
    
    # Find the projection layer in encoder_embeddings
    if hasattr(model, 'model') and hasattr(model.model, 'encoder_embeddings'):
        embeddings = model.model.encoder_embeddings
        for key, embedding in embeddings.items():
            if hasattr(embedding, 'proj'):
                proj = embedding.proj
                current_in_features = proj.in_features
                
                if current_in_features != expected_in_features:
                    print(f"Replacing projection layer in {key}")
                    print(f"  Current: in_features={current_in_features}, out_features={proj.out_features}")
                    print(f"  New: in_features={expected_in_features}, out_features={embed_dim}")
                    
                    # Create new projection layer
                    new_proj = nn.Linear(expected_in_features, embed_dim, bias=proj.bias is not None)
                    
                    # Initialize with old weights if possible
                    if current_in_features <= expected_in_features:
                        # Copy old weights to beginning
                        new_proj.weight.data[:, :current_in_features] = proj.weight.data
                        # Initialize rest with mean
                        mean_weight = proj.weight.data.mean(dim=1, keepdim=True)
                        new_proj.weight.data[:, current_in_features:] = mean_weight.expand(-1, expected_in_features - current_in_features)
                    else:
                        # Old is larger, just take first part
                        new_proj.weight.data = proj.weight.data[:, :expected_in_features]
                    
                    if proj.bias is not None:
                        new_proj.bias.data = proj.bias.data
                    
                    # Replace the layer
                    embedding.proj = new_proj
                    
                    # Move to same device as the rest of the model
                    device = next(model.parameters()).device if list(model.parameters()) else 'cpu'
                    new_proj.to(device)
                    
                    print(f"  ✓ Projection layer replaced and moved to {device}")
    
    return model


def load_terramind_rgb_to_rgbn(
    checkpoint_path: str,
    rgbn_model: nn.Module,
    patch_size: int = 16,
    device: str = 'cpu',
    num_bands: int = 4
) -> Tuple[nn.Module, Dict]:
    """
    Load TerraMind RGB checkpoint and adapt it for RGBN model.
    
    Args:
        checkpoint_path: Path to RGB checkpoint file
        rgbn_model: RGBN TerraMind model to load weights into
        patch_size: Patch size (default 16)
        device: Device to load checkpoint on
        num_bands: Number of bands in target model (default 4 for RGBN)
    
    Returns:
        Tuple of (model with loaded weights, load_state_dict message)
    """
    # First, replace projection layer to handle correct number of bands
    rgbn_model = replace_terramind_projection_layer(rgbn_model, num_bands, patch_size)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        rgb_state_dict = checkpoint['state_dict']
    else:
        rgb_state_dict = checkpoint
    
    # Adapt weights
    adapted_sd = adapt_terramind_state_dict(rgb_state_dict, rgbn_model, patch_size)
    
    # Load into model
    msg = rgbn_model.load_state_dict(adapted_sd, strict=False)
    
    return rgbn_model, msg


def find_embedding_keys(encoder_embeddings):
    """
    Inspect encoder_embeddings (ModuleDict) and guess S2 / S1 keys.
    """
    keys = list(encoder_embeddings.keys())
    s2_key = None
    s1_key = None

    for k in keys:
        kl = k.lower()
        if "sen2" in kl or "s2l2a" in kl:
            s2_key = k
        if "sen1" in kl or "s1grd" in kl:
            s1_key = k

    return s2_key, s1_key


def copy_shared_encoder_weights(src_model, dst_model):
    """
    Copy shared encoder backbone weights (transformer blocks) from src_model to dst_model.
    This copies everything except modality-specific embeddings.
    """
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()

    copied = 0
    skipped = 0
    for k in dst_sd.keys():
        # Skip modality-specific embeddings (they will be handled separately)
        if 'encoder_embeddings' in k:
            skipped += 1
            continue
        if k in src_sd and src_sd[k].shape == dst_sd[k].shape:
            dst_sd[k] = src_sd[k]
            copied += 1
    dst_model.load_state_dict(dst_sd, strict=False)
    return copied, skipped


def init_s2_part_from_s2_model(s2_model, s2s1_model):
    """
    Initialize the S2 part of S2+S1 model by copying S2 embedding weights from S2-only model.
    Copies all S2 embedding parameters including projection layers (adapting if needed).
    """
    if not hasattr(s2_model, "encoder_embeddings") or not hasattr(
        s2s1_model, "encoder_embeddings"
    ):
        print("Models do not expose encoder_embeddings; skipping S2 part init.")
        return

    s2_embs = s2_model.encoder_embeddings
    s2s1_embs = s2s1_model.encoder_embeddings

    s2_key_single, _ = find_embedding_keys(s2_embs)
    s2_key_multi, _ = find_embedding_keys(s2s1_embs)
    
    if s2_key_single is None:
        print("Could not find S2 embedding in S2-only model; skipping S2 part init.")
        return
    if s2_key_multi is None:
        print("Could not find S2 embedding in S2+S1 model; skipping S2 part init.")
        return

    emb_s2_src = s2_embs[s2_key_single]
    emb_s2_tgt = s2s1_embs[s2_key_multi]

    # Use state_dict to get all parameters including nested ones
    src_sd = emb_s2_src.state_dict()
    tgt_sd = emb_s2_tgt.state_dict()

    with torch.no_grad():
        copied_params = 0
        adapted_params = 0
        for name, p_src in src_sd.items():
            if name in tgt_sd:
                p_tgt = tgt_sd[name]
                if p_tgt.shape == p_src.shape:
                    # Exact shape match - direct copy
                    p_tgt.copy_(p_src)
                    copied_params += 1
                elif 'proj.weight' in name:
                    # Projection weight with shape mismatch - try to adapt
                    out_src, in_src = p_src.shape
                    out_tgt, in_tgt = p_tgt.shape
                    
                    if out_src == out_tgt and in_src != in_tgt:
                        # Same output dim, different input dim - copy what we can
                        min_in = min(in_src, in_tgt)
                        p_tgt[:, :min_in] = p_src[:, :min_in]
                        if in_tgt > in_src:
                            # Initialize extra dimensions with mean
                            mean_weight = p_src.mean(dim=1, keepdim=True)
                            p_tgt[:, in_src:] = mean_weight.expand(-1, in_tgt - in_src)
                        adapted_params += 1
                    elif in_src == in_tgt and out_src != out_tgt:
                        # Same input dim, different output dim - copy what we can
                        min_out = min(out_src, out_tgt)
                        p_tgt[:min_out, :] = p_src[:min_out, :]
                        if out_tgt > out_src:
                            # Initialize extra dimensions with mean
                            mean_weight = p_src.mean(dim=0, keepdim=True)
                            p_tgt[out_src:, :] = mean_weight.expand(out_tgt - out_src, -1)
                        adapted_params += 1
                elif 'proj.bias' in name:
                    # Projection bias - copy if output dim matches
                    if len(p_src) == len(p_tgt):
                        p_tgt.copy_(p_src)
                        copied_params += 1
                    else:
                        min_len = min(len(p_src), len(p_tgt))
                        p_tgt[:min_len] = p_src[:min_len]
                        adapted_params += 1
        
        # Load the updated state dict back into the target embedding
        emb_s2_tgt.load_state_dict(tgt_sd, strict=False)

    return copied_params, adapted_params


def adapt_terramind_s2_to_s2s1(s2_model, s2s1_model):
    """
    Adapt TerraMind S2-only model weights to S2+S1 model.
    
    This function:
    1. Copies shared encoder backbone weights (transformer blocks)
    2. Initializes S2 part from S2-only model (including projection layer)
    3. Leaves S1 part randomly initialized
    
    Args:
        s2_model: TerraMind model trained on S2-only
        s2s1_model: TerraMind model with S2+S1 modalities (target)
    
    Returns:
        Tuple of (num_copied_shared, num_copied_s2, num_adapted_s2)
    """
    print("Adapting TerraMind weights from S2-only to S2+S1...")
    print("-" * 80)
    
    # Step 1: Copy shared encoder weights
    copied_shared, skipped_shared = copy_shared_encoder_weights(s2_model, s2s1_model)
    print(f"Copied {copied_shared} shared encoder parameters (transformer blocks, etc.)")
    print(f"Skipped {skipped_shared} modality-specific embedding parameters (handled separately)")
    
    # Step 2: Initialize S2 part from S2-only model
    copied_s2, adapted_s2 = init_s2_part_from_s2_model(s2_model, s2s1_model)
    print(f"Initialized S2 part: copied {copied_s2} parameters, adapted {adapted_s2} parameters")
    print("S1 part remains randomly initialized")
    print("-" * 80)
    
    return copied_shared, copied_s2, adapted_s2

