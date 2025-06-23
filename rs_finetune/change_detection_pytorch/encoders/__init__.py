import functools
import re
import torch
import torch.utils.model_zoo as model_zoo

from ._preprocessing import preprocess_input
from .resnet import resnet_encoders
from .swin_transformer import swin_transformer_encoders
from .vision_transformer import vit_encoders
from .vision_transformer_overlap import vit_overlap_encoders
from .channel_vit import cvit_encoders
from .sd_channel_vit import sd_cvit_encoders
from .prithvi import prithvi_encoders

from .clay import clay_encoders
from .dofa import dofa_encoders
# from .dofa_pangea import dofa_encoders
from .dinov2_sat import SSLAE, dinov2_encoders
from .anysat import anysat_encoders
from .croma import croma_encoders
# from .prithvi_pangea import prithvi_encoders_pangea
# from .hrnet import hrnet_encoders
from ._utils import load_pretrained, adjust_state_dict_prefix
from .utils_anysat import PatchLTAEMulti, PatchMLPMulti, AnyModule, TransformerMulti

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
encoders = {}
encoders.update(resnet_encoders)
encoders.update(swin_transformer_encoders)
encoders.update(vit_encoders)
encoders.update(cvit_encoders)
encoders.update(sd_cvit_encoders)
encoders.update(vit_overlap_encoders)
# encoders.update(prithvi_encoders)
encoders.update(clay_encoders)
encoders.update(dinov2_encoders)
# encoders.update(dofa_encoders)
encoders.update(dofa_encoders)
encoders.update(anysat_encoders)
encoders.update(croma_encoders)
encoders.update(prithvi_encoders)

def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, scales=[4, 2, 1, 0.5], enable_sample=False, **kwargs):
    if weights =='':
        weights = None
    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    # params.update(scales=scales)

    if 'cvit-pretrained' in name.lower():
        params.update(return_feats=True)
        params.update(enable_sample=enable_sample)
    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError("Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                weights, name, list(encoders[name]["pretrained_settings"].keys()),
            ))
        try:
            if 'ibot' in name:
                if 'imagenet' in settings["url"]:
                    state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))['state_dict']
                else:
                    state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))['teacher']
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                msg = encoder.load_state_dict(state_dict, strict=False)
                print('Pretrained weights found at {} and loaded with msg: {}'.format(settings["url"], msg))
            elif 'vit-s8' in name:
                state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))['teacher']
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                msg = encoder.load_state_dict(state_dict, strict=False)
                print('Pretrained weights found at {} and loaded with msg: {}'.format(settings["url"], msg))
            elif 'cvit-pretrained' in name.lower():
                state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))['teacher']
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                msg = encoder.load_state_dict(state_dict, strict=False)
                print('Pretrained weights found at {} and loaded with msg: {}'.format(settings["url"], msg))
            elif 'cvit' in name.lower():
                model = torch.hub.load('insitro/ChannelViT', settings["url"], pretrained=True)
                encoder.load_state_dict(model.state_dict(), strict=False)
                encoder.out_channels = (384, 384, 384, 384)
                encoder.out_idx = (2, 5, 8, 11)
            elif 'prithvi' in name.lower():
                state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))

                del state_dict['pos_embed']
                del state_dict['decoder_pos_embed']
                msg = encoder.load_state_dict(state_dict, strict=False)

                encoder.out_channels = params['out_channels']
                print('Pretrained weights found at {} and loaded with msg: {}'.format(settings["url"], msg))
            elif 'dofa' in name.lower():
                state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))
                msg = encoder.load_state_dict(state_dict, strict=False)

            elif 'clay' in name.lower() or 'anysat' in name.lower():
                pass
            else:
                encoder.load_state_dict(model_zoo.load_url(settings["url"], map_location=torch.device('cpu')))
        except Exception as e:
            print(e)
            try:
                if 'satlas' in weights:
                    checkpoint = torch.load(settings["url"])
                    checkpoint_model = adjust_state_dict_prefix(checkpoint, 'backbone', 'backbone.', prefix_allowed_count=0)
                        
                elif 'geopile' in weights:
                    checkpoint_model = load_pretrained(encoder, settings["url"], 'cpu')
                else:
                    checkpoint_model = torch.load(settings["url"])
                    
                msg = encoder.load_state_dict(checkpoint_model, strict=False)
                print('Pretrained weights found at {} and loaded with msg: {}'.format(settings["url"], msg))
            except KeyError:
                print('Cant find model')


    ######## !!!!!!  TODO  ###########
    
    # if (('ibot' not in name) and 
    #     ('dinov2' not in name) and 
    #     ('dino-mc' not in name) and 
    #     ('cvit' not in name.lower()) and 
    #     ('prithvi' not in name.lower()) and 
    #     ('clay' not in name.lower())):
    #     encoder.set_in_channels(in_channels, pretrained=weights is not None)
    if output_stride != 32:
        encoder.make_dilated(output_stride)
    
    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained="imagenet"):
    settings = encoders[encoder_name]["pretrained_settings"]

    if pretrained not in settings.keys():
        raise ValueError("Available pretrained options {}".format(settings.keys()))

    formatted_settings = {}
    formatted_settings["input_space"] = settings[pretrained].get("input_space")
    formatted_settings["input_range"] = settings[pretrained].get("input_range")
    formatted_settings["mean"] = settings[pretrained].get("mean")
    formatted_settings["std"] = settings[pretrained].get("std")
    return formatted_settings


def get_preprocessing_fn(encoder_name, pretrained="imagenet"):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)
