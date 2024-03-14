import functools

import torch
import torch.utils.model_zoo as model_zoo

from ._preprocessing import preprocess_input
from .densenet import densenet_encoders
from .dpn import dpn_encoders
from .efficientnet import efficient_net_encoders
from .inceptionresnetv2 import inceptionresnetv2_encoders
from .inceptionv4 import inceptionv4_encoders
from .mobilenet import mobilenet_encoders
from .resnet import resnet_encoders
from .senet import senet_encoders
from .timm_efficientnet import timm_efficientnet_encoders
from .timm_gernet import timm_gernet_encoders
from .timm_mobilenetv3 import timm_mobilenetv3_encoders
from .timm_regnet import timm_regnet_encoders
from .timm_res2net import timm_res2net_encoders
from .timm_resnest import timm_resnest_encoders
from .timm_sknet import timm_sknet_encoders
from .timm_universal import TimmUniversalEncoder
from .vgg import vgg_encoders
from .xception import xception_encoders
from .swin_transformer import swin_transformer_encoders
from .mit_encoder import mit_encoders
from .vision_transformer import vit_encoders

# from .hrnet import hrnet_encoders
from ._utils import load_pretrained, adjust_state_dict_prefix

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
encoders = {}
encoders.update(resnet_encoders)
encoders.update(dpn_encoders)
encoders.update(vgg_encoders)
encoders.update(senet_encoders)
encoders.update(densenet_encoders)
encoders.update(inceptionresnetv2_encoders)
encoders.update(inceptionv4_encoders)
encoders.update(efficient_net_encoders)
encoders.update(mobilenet_encoders)
encoders.update(xception_encoders)
encoders.update(timm_efficientnet_encoders)
encoders.update(timm_resnest_encoders)
encoders.update(timm_res2net_encoders)
encoders.update(timm_regnet_encoders)
encoders.update(timm_sknet_encoders)
encoders.update(timm_mobilenetv3_encoders)
encoders.update(timm_gernet_encoders)
encoders.update(swin_transformer_encoders)
encoders.update(mit_encoders)
encoders.update(vit_encoders)
# encoders.update(hrnet_encoders)


def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):

    if name.startswith("tu-"):
        name = name[3:]
        encoder = TimmUniversalEncoder(
            name=name,
            in_channels=in_channels,
            depth=depth,
            output_stride=output_stride,
            pretrained=weights is not None,
            **kwargs
        )
        return encoder

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
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
            else:
                encoder.load_state_dict(model_zoo.load_url(settings["url"], map_location=torch.device('cpu')))
        except Exception as e:
            print(e)
            try:
                if 'satlas' in weights:
                    checkpoint = torch.load(settings["url"])
                    checkmoint_model = adjust_state_dict_prefix(checkpoint, 'backbone', 'backbone.', prefix_allowed_count=0)
                else:
                    checkmoint_model = load_pretrained(encoder, settings["url"], 'cpu')
                msg = encoder.load_state_dict(checkmoint_model, strict=False)
                print('Pretrained weights found at {} and loaded with msg: {}'.format(settings["url"], msg))
            except KeyError:
                print('Cant find model')

    if 'ibot' not in name:
        encoder.set_in_channels(in_channels, pretrained=weights is not None)
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
