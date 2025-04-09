from change_detection_pytorch.encoders import (vit_encoders, swin_transformer_encoders, 
                                               prithvi_encoders, clay_encoders, dinov2_encoders, 
                                               dofa_encoders, sd_cvit_encoders, anysat_encoders, croma_encoders)


from change_detection_pytorch.encoders._utils import load_pretrained, adjust_state_dict_prefix
import torch

def load_encoder(encoder_name='ibot-B', encoder_weights='imagenet', enable_sample=False, shared_proj=False):
    
        if 'swin' in encoder_name.lower():
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

        elif 'cvit-pretrained' in encoder_name.lower():
            Encoder = sd_cvit_encoders[encoder_name]["encoder"]
            params = sd_cvit_encoders[encoder_name]["params"]
            params.update(return_feats=False)
            params.update(enable_sample=enable_sample)
            params.update(shared_proj=shared_proj)
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
            # encoder = torch.hub.load('gastruc/anysat', 'anysat', pretrained=True, force_reload=True, flash_attn=False)
            Encoder = anysat_encoders[encoder_name]["encoder"]
            params = anysat_encoders[encoder_name]["params"]
            encoder = Encoder(**params)
            encoder = encoder.from_pretrained('base', flash_attn=False)
        
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