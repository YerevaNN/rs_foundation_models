import torch
import torch.nn as nn
import numpy as np

from change_detection_pytorch.encoders import (vit_encoders, swin_transformer_encoders, timm_vit_encoders, timm_resnet_encoders,
                                               prithvi_encoders, clay_encoders, dinov2_encoders, 
                                               dofa_encoders, sd_cvit_encoders, anysat_encoders, croma_encoders)


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

    if old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data)
    
    return new_conv

def load_encoder(encoder_name='ibot-B', encoder_weights='imagenet', 
                 enable_sample=False, shared_proj=False, add_ch_embed=False, 
                 enable_multiband_input=False, multiband_channel_count=12):
    
    if 'timm' in encoder_name.lower():
        Encoder = timm_encoders[encoder_name]["encoder"]
        params = timm_encoders[encoder_name]["params"]
        params.update(for_cls=True)
        encoder = Encoder(**params)
        return encoder
    
    elif 'swin' in encoder_name.lower():
        if 'satlas_ms' in encoder_weights.lower():
            import satlaspretrain_models

            weights_manager = satlaspretrain_models.Weights()
            encoder = weights_manager.get_pretrained_model(model_identifier="Sentinel2_SwinB_SI_MS")
        else:
            Encoder = swin_transformer_encoders[encoder_name]["encoder"]
            params = swin_transformer_encoders[encoder_name]["params"]
            gap = False if 'satlas' in encoder_weights else True
            params.update(for_cls=True, gap=gap, window_size=8)

            encoder = Encoder(**params)
            settings = swin_transformer_encoders[encoder_name]["pretrained_settings"][encoder_weights]
            checkmoint_model = load_pretrained(encoder, settings["url"], 'cpu')
            msg = encoder.load_state_dict(checkmoint_model, strict=False)
            print(msg)

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
            # path = '/nfs/ap/mnt/frtn/rs-results/dinov2_sat/SSLhuge_satellite.pth'
            # encoder = SSLAE(pretrained=path, huge=True, classification=True).eval()
        else:
            encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').eval()

            if enable_multiband_input:
                old_conv = encoder.patch_embed.proj
                encoder.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(old_conv=old_conv, 
                                                    new_in_channels=multiband_channel_count)

    elif 'cvit-pretrained' in encoder_name.lower():
        Encoder = sd_cvit_encoders[encoder_name]["encoder"]
        params = sd_cvit_encoders[encoder_name]["params"]
        params.update(return_feats=False)
        params.update(enable_sample=enable_sample)
        params.update(shared_proj=shared_proj)
        params.update(add_ch_embed=add_ch_embed)

        encoder = Encoder(**params)
        
        # Load weights
        settings = sd_cvit_encoders[encoder_name]["pretrained_settings"][encoder_weights]
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
        encoder = Encoder(**params)

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