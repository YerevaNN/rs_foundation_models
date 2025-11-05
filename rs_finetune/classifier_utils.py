import torch
import torch.nn as nn
import numpy as np

from change_detection_pytorch.encoders import (vit_encoders, swin_transformer_encoders, timm_vit_encoders, timm_resnet_encoders,
                                               prithvi_encoders, clay_encoders, dinov2_encoders,
                                               dofa_encoders, chi_vit_encoders, anysat_encoders, croma_encoders, terrafm_encoders)


from change_detection_pytorch.encoders._utils import load_pretrained, adjust_state_dict_prefix


timm_encoders = timm_vit_encoders.copy()
timm_encoders.update(timm_resnet_encoders)

def adapt_rgb_conv_layer_to_multiband(old_conv: nn.Conv2d, new_in_channels: int) -> nn.Conv2d:

    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )
    
    old_weights = old_conv.weight.data
    averaged_weights = old_weights.mean(dim=1, keepdim=True)
    new_weights = averaged_weights.repeat(1, new_in_channels, 1, 1)
    new_weights = new_weights / new_in_channels
    new_conv.weight.data.copy_(new_weights)

    if old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data)
    
    return new_conv

def adapt_rgb_conv_layer_to_multiband_preserve_rgb(old_conv: nn.Conv2d, new_in_channels: int = 4) -> nn.Conv2d:
    """
    Adapts a conv layer to handle multiband input while preserving existing band weights.
    For new bands, uses the average of existing band weights.
    
    Args:
        old_conv: Original convolution layer
        new_in_channels: New number of input channels
    """
    old_in_channels = old_conv.in_channels
    
    # Create new conv layer on the SAME DEVICE as the old one
    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    ).to(old_conv.weight.device)  # Move to same device as original
    
    old_weights = old_conv.weight.data
    new_weights = torch.zeros(
        old_conv.out_channels, 
        new_in_channels, 
        *old_conv.kernel_size, 
        device=old_weights.device,  # Use same device
        dtype=old_weights.dtype
    )
    
    # Preserve existing band weights
    if new_in_channels >= old_in_channels:
        new_weights[:, :old_in_channels, :, :] = old_weights
        
        # For new bands, use average of existing band weights
        if new_in_channels > old_in_channels:
            existing_avg = old_weights.mean(dim=1, keepdim=True)
            remaining_channels = new_in_channels - old_in_channels
            new_weights[:, old_in_channels:, :, :] = existing_avg.repeat(1, remaining_channels, 1, 1)
    else:
        # If reducing channels, keep only the first new_in_channels
        new_weights = old_weights[:, :new_in_channels, :, :]
    
    new_conv.weight.data.copy_(new_weights)

    if old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data)
    
    return new_conv

def adapt_rgb_conv3d_layer_to_multiband(old_conv: nn.Conv3d, new_in_channels: int) -> nn.Conv3d:

    new_conv = nn.Conv3d(
        in_channels=new_in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )
    
    old_weights = old_conv.weight.data
    averaged_weights = old_weights.mean(dim=1, keepdim=True)
    new_weights = averaged_weights.repeat(1, new_in_channels, 1, 1, 1)
    new_weights = new_weights / new_in_channels
    new_conv.weight.data.copy_(new_weights)

    if old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data)
    
    return new_conv

def adapt_rgb_conv3d_layer_to_multiband_preserve_rgb(old_conv: nn.Conv3d, new_in_channels: int) -> nn.Conv3d:
    """
    Adapts a 3D conv layer to handle multiband input while preserving existing band weights.
    For new bands, uses the average of existing band weights.
    
    Args:
        old_conv: Original 3D convolution layer
        new_in_channels: New number of input channels
    """
    old_in_channels = old_conv.in_channels
    
    new_conv = nn.Conv3d(
        in_channels=new_in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )
    
    old_weights = old_conv.weight.data
    new_weights = torch.zeros(
        old_conv.out_channels, 
        new_in_channels, 
        *old_conv.kernel_size, 
        device=old_weights.device, 
        dtype=old_weights.dtype
    )
    
    # Preserve existing band weights
    if new_in_channels >= old_in_channels:
        new_weights[:, :old_in_channels, :, :, :] = old_weights
        
        # For new bands, use average of existing band weights
        if new_in_channels > old_in_channels:
            existing_avg = old_weights.mean(dim=1, keepdim=True)
            remaining_channels = new_in_channels - old_in_channels
            new_weights[:, old_in_channels:, :, :, :] = existing_avg.repeat(1, remaining_channels, 1, 1, 1)
    else:
        # If reducing channels, keep only the first new_in_channels
        new_weights = old_weights[:, :new_in_channels, :, :, :]
    
    new_conv.weight.data.copy_(new_weights)

    if old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data)
    
    return new_conv

def adapt_encoder_for_multiband_eval(encoder, multiband_channel_count = 4):
    """
    Adapts an encoder to handle multiband input while preserving RGB weights.
    This function should be called after loading checkpoint weights.
    
    Args:
        encoder: The encoder to adapt
        multiband_channel_count: Number of input channels (e.g., 4 for RGB+N)
    """
    if hasattr(encoder, 'patch_embed') and hasattr(encoder.patch_embed, 'proj'):
        # Standard ViT, Swin, etc.
        old_conv = encoder.patch_embed.proj
        encoder.patch_embed.proj = adapt_rgb_conv_layer_to_multiband_preserve_rgb(
            old_conv=old_conv, 
            new_in_channels=multiband_channel_count
        )
        # Update input channel count
        if hasattr(encoder.patch_embed, 'in_chans'):
            encoder.patch_embed.in_chans = multiband_channel_count
        if hasattr(encoder, 'in_chans'):
            encoder.in_chans = multiband_channel_count
            
    elif hasattr(encoder, 'model') and hasattr(encoder.model, 'patch_embed') and hasattr(encoder.model.patch_embed, 'proj'):
        # timm ViT, some other wrapped models
        old_conv = encoder.model.patch_embed.proj
        encoder.model.patch_embed.proj = adapt_rgb_conv_layer_to_multiband_preserve_rgb(
            old_conv=old_conv, 
            new_in_channels=multiband_channel_count
        )
        # Update input channel count
        if hasattr(encoder.model.patch_embed, 'in_chans'):
            encoder.model.patch_embed.in_chans = multiband_channel_count
        if hasattr(encoder.model, 'in_chans'):
            encoder.model.in_chans = multiband_channel_count
            
    elif hasattr(encoder, 'model') and hasattr(encoder.model, 'model') and hasattr(encoder.model.model, 'patch_embed') and hasattr(encoder.model.model.patch_embed, 'proj'):
        old_conv = encoder.model.model.patch_embed.proj
        encoder.model.model.patch_embed.proj = adapt_rgb_conv_layer_to_multiband_preserve_rgb(
            old_conv=old_conv, 
            new_in_channels=multiband_channel_count
        )
        if hasattr(encoder.model.model.patch_embed, 'in_chans'):
            encoder.model.model.patch_embed.in_chans = multiband_channel_count
            
    elif hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'patch_embed') and hasattr(encoder.backbone.patch_embed, 'proj'):
        old_conv = encoder.backbone.patch_embed.proj
        encoder.backbone.patch_embed.proj = adapt_rgb_conv_layer_to_multiband_preserve_rgb(
            old_conv=old_conv, 
            new_in_channels=multiband_channel_count
        )
        # Update input channel count
        if hasattr(encoder.backbone.patch_embed, 'in_chans'):
            encoder.backbone.patch_embed.in_chans = multiband_channel_count
        if hasattr(encoder.backbone, 'in_chans'):
            encoder.backbone.in_chans = multiband_channel_count
            
    elif hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'backbone') and hasattr(encoder.backbone.backbone, 'patch_embed') and hasattr(encoder.backbone.backbone.patch_embed, 'proj'):
        old_conv = encoder.backbone.backbone.patch_embed.proj
        encoder.backbone.backbone.patch_embed.proj = adapt_rgb_conv_layer_to_multiband_preserve_rgb(
            old_conv=old_conv, 
            new_in_channels=multiband_channel_count
        )
        # Update input channel count
        if hasattr(encoder.backbone.backbone.patch_embed, 'in_chans'):
            encoder.backbone.backbone.patch_embed.in_chans = multiband_channel_count
            
    elif hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'features') and hasattr(encoder.backbone.features, '[0]') and hasattr(encoder.backbone.features[0], '[0]'):
        # Swin transformer with features
        old_conv = encoder.backbone.features[0][0]
        encoder.backbone.features[0][0] = adapt_rgb_conv_layer_to_multiband_preserve_rgb(
            old_conv=old_conv, 
            new_in_channels=multiband_channel_count
        )
        # Update input channel count
        if hasattr(encoder.backbone, 'in_channels'):
            encoder.backbone.in_channels = multiband_channel_count
            
    elif hasattr(encoder, 'model') and hasattr(encoder.model, 'conv1'):
        # ResNet-style models
        old_conv = encoder.model.conv1
        encoder.model.conv1 = adapt_rgb_conv_layer_to_multiband_preserve_rgb(
            old_conv=old_conv, 
            new_in_channels=multiband_channel_count
        )
        # Update input channel count
        if hasattr(encoder.model, 'in_channels'):
            encoder.model.in_channels = multiband_channel_count
            
    elif hasattr(encoder, 'embeddings') and hasattr(encoder.embeddings, 'patch_embeddings'):
        # DINOv3 from transformers
        old_conv = encoder.embeddings.patch_embeddings
        encoder.embeddings.patch_embeddings = adapt_rgb_conv_layer_to_multiband_preserve_rgb(
            old_conv=old_conv, 
            new_in_channels=multiband_channel_count
        )
        # Update input channel count in config if available
        if hasattr(encoder.config, 'num_channels'):
            encoder.config.num_channels = multiband_channel_count
            
    else:
        print(f"Warning: Could not find conv layer to adapt in encoder type: {type(encoder)}")
        print(f"Available attributes: {[attr for attr in dir(encoder) if not attr.startswith('_')]}")
        return False
    
    print(f"Successfully adapted encoder to {multiband_channel_count} channels")

def load_encoder(encoder_name='ibot-B', encoder_weights='imagenet', 
                 enable_sample=False, shared_proj=False, add_ch_embed=False, 
                 enable_multiband_input=False, multiband_channel_count=12):
    
    if 'timm' in encoder_name.lower():
        Encoder = timm_encoders[encoder_name]["encoder"]
        params = timm_encoders[encoder_name]["params"]
        params.update(for_cls=True)
        
        if enable_multiband_input:
            params["in_channels"] = multiband_channel_count
        
        encoder = Encoder(**params)
        
        if enable_multiband_input and hasattr(encoder.model, 'conv1'):
            old_conv = encoder.model.conv1
            encoder.model.conv1 = adapt_rgb_conv_layer_to_multiband(old_conv=old_conv, 
                                                new_in_channels=multiband_channel_count)
        elif enable_multiband_input and hasattr(encoder.model, 'patch_embed') and hasattr(encoder.model.patch_embed, 'proj'):
            old_conv = encoder.model.patch_embed.proj
            encoder.model.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(old_conv=old_conv, 
                                                new_in_channels=multiband_channel_count)
        
        return encoder
    
    elif 'swin' in encoder_name.lower():
        if 'satlas_ms' in encoder_weights.lower():
            import satlaspretrain_models

            weights_manager = satlaspretrain_models.Weights()
            encoder = weights_manager.get_pretrained_model(model_identifier="Sentinel2_SwinB_SI_MS")
                        
            if enable_multiband_input:
                old_conv = encoder.backbone.backbone.features[0][0]
                encoder.backbone.backbone.features[0][0] = adapt_rgb_conv_layer_to_multiband(
                    old_conv=old_conv,
                    new_in_channels=multiband_channel_count
                    )
        else:
            Encoder = swin_transformer_encoders[encoder_name]["encoder"]
            params = swin_transformer_encoders[encoder_name]["params"]
            gap = False if 'satlas' in encoder_weights else True
            params.update(for_cls=True, gap=gap, window_size=8)
            
            if enable_multiband_input:
                params["in_channels"] = multiband_channel_count

            encoder = Encoder(**params)
            settings = swin_transformer_encoders[encoder_name]["pretrained_settings"][encoder_weights]
            checkmoint_model = load_pretrained(encoder, settings["url"], 'cpu')
            msg = encoder.load_state_dict(checkmoint_model, strict=False)
            print(msg)
            
            if enable_multiband_input:
                old_conv = encoder.patch_embed.proj
                encoder.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(old_conv=old_conv, 
                                                    new_in_channels=multiband_channel_count)

    elif 'ibot' in encoder_name.lower():
        Encoder = vit_encoders[encoder_name]["encoder"]
        params = vit_encoders[encoder_name]["params"]
        params.update(for_cls=True)
        encoder = Encoder(**params)
        if encoder_weights == 'random':
            return encoder
        else:
            settings = vit_encoders[encoder_name]["pretrained_settings"][encoder_weights]
            if 'imagenet' in settings["url"]:
                state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))['state_dict']
            else:
                state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))['teacher']
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = encoder.load_state_dict(state_dict, strict=False)
            print(msg)
            if enable_multiband_input:
                old_conv = encoder.patch_embed.proj
                encoder.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(old_conv=old_conv, 
                                                    new_in_channels=multiband_channel_count)
    elif 'dino' in encoder_name.lower():
        if 'sat' in encoder_name.lower():
            Encoder = dinov2_encoders[encoder_name]["encoder"]
            params = dinov2_encoders[encoder_name]["params"]
            params.update(classification=True)
            encoder = Encoder(**params).eval()
            
            if enable_multiband_input:
                if hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'backbone') and hasattr(encoder.backbone.backbone, 'patch_embed') and hasattr(encoder.backbone.backbone.patch_embed, 'proj'):
                    old_conv = encoder.backbone.backbone.patch_embed.proj
                    encoder.backbone.backbone.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(
                        old_conv=old_conv,
                        new_in_channels=multiband_channel_count
                    )
            # path = '/nfs/ap/mnt/frtn/rs-results/dinov2_sat/SSLhuge_satellite.pth'
            # encoder = SSLAE(pretrained=path, huge=True, classification=True).eval()
        elif "v3" in encoder_name.lower():
            print("=" * 100)
            print("Loading Dinov3 encoder")
            print("=" * 100)
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m",
                                                trust_remote_code=True)

            if enable_multiband_input:
                old_conv = encoder.embeddings.patch_embeddings
                encoder.embeddings.patch_embeddings = adapt_rgb_conv_layer_to_multiband(old_conv=old_conv, 
                                                new_in_channels=multiband_channel_count)
        else:
            encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').eval()

            if enable_multiband_input:
                old_conv = encoder.patch_embed.proj
                encoder.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(old_conv=old_conv, 
                                                new_in_channels=multiband_channel_count)

    elif 'cvit-pretrained' in encoder_name.lower():
        Encoder = chi_vit_encoders[encoder_name]["encoder"]
        params = chi_vit_encoders[encoder_name]["params"]
        params.update(return_feats=False)
        params.update(enable_sample=enable_sample)
        params.update(shared_proj=shared_proj)
        params.update(add_ch_embed=add_ch_embed)

        encoder = Encoder(**params)
        
        # Load weights
        settings = chi_vit_encoders[encoder_name]["pretrained_settings"][encoder_weights]
        state_dict = torch.load(settings["url"], map_location=torch.device('cpu'), weights_only=False)['teacher']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = encoder.load_state_dict(state_dict, strict=False)
        print(msg)
    
    elif 'cvit' in encoder_name.lower():
        encoder = torch.hub.load('insitro/ChannelViT', 'so2sat_channelvit_small_p8_with_hcs_random_split_supervised', pretrained=True)

    elif 'anysat' in encoder_name.lower():
        Encoder = anysat_encoders[encoder_name]["encoder"]
        params = anysat_encoders[encoder_name]["params"].copy()
        params['for_cls'] = True
        params['out_idx'] = None
        params['out_channels'] = None
        encoder = Encoder(**params)
        pretrained_encoder = encoder.from_pretrained('base', flash_attn=False)
        encoder.model.load_state_dict(pretrained_encoder.model.state_dict(), strict=False)
    
    elif 'croma' in encoder_name.lower():
        Encoder = croma_encoders[encoder_name]["encoder"]
        params = croma_encoders[encoder_name]["params"]
        params.update(for_cls=True)
        encoder = Encoder(**params)

    elif 'terrafm' in encoder_name.lower():
        Encoder = terrafm_encoders[encoder_name]["encoder"]
        params = terrafm_encoders[encoder_name]["params"]
        params.update(for_cls=True)
        encoder = Encoder(**params)
        
        settings = terrafm_encoders[encoder_name]["pretrained_settings"][encoder_weights]
        state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))
        msg = encoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded TerraFM pretrained weights from {settings['url']}")
        print(f"Missing keys: {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")
        
        if enable_multiband_input:
            old_conv = encoder.patch_embed.proj
            encoder.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(old_conv=old_conv, 
                                                new_in_channels=multiband_channel_count)
    
    elif 'prithvi' in encoder_name.lower():
        Encoder = prithvi_encoders[encoder_name]["encoder"]
        params = prithvi_encoders[encoder_name]["params"]
        params.update(for_cls=True)
                
        encoder = Encoder(**params)
        settings = prithvi_encoders[encoder_name]["pretrained_settings"][encoder_weights]
        state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))
        del state_dict['pos_embed']
        del state_dict['decoder_pos_embed']

        msg = encoder.load_state_dict(state_dict, strict=False)
        print(msg)
        
        if enable_multiband_input:
            old_conv = encoder.patch_embed.proj
            encoder.patch_embed.proj = adapt_rgb_conv3d_layer_to_multiband(old_conv=old_conv, 
                                                new_in_channels=multiband_channel_count)
            params["in_chans"] = multiband_channel_count

    elif 'clay' in encoder_name.lower():
        Encoder = clay_encoders[encoder_name]["encoder"]
        params = clay_encoders[encoder_name]["params"]
        params.update(for_cls=True)
        encoder = Encoder(**params)

    elif 'dofa' in encoder_name.lower():
        Encoder = dofa_encoders[encoder_name]["encoder"]
        params = dofa_encoders[encoder_name]["params"]
        params.update(for_cls=True)
        params.update(global_pool=False)
        encoder = Encoder(**params)

        settings = dofa_encoders[encoder_name]["pretrained_settings"][encoder_weights]
        state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))
        msg = encoder.load_state_dict(state_dict, strict=False)
        print(msg)

    return encoder