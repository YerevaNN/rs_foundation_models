import torch
import torch.nn as nn
import numpy as np
from scipy import interpolate


def remap_pretrained_keys_swin(model, checkpoint_model):
    state_dict = model.state_dict()
    
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                print(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    print(f"{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1 ** 0.5)
                    dst_size = int(L2 ** 0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().cpu().numpy()
                        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                            relative_position_bias_table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model

def adjust_state_dict_prefix(state_dict, needed, prefix=None, prefix_allowed_count=None):
    """
    Adjusts the keys in the state dictionary by replacing 'backbone.backbone' prefix with 'backbone'.

    Args:
        state_dict (dict): Original state dictionary with 'backbone.backbone' prefixes.

    Returns:
        dict: Modified state dictionary with corrected prefixes.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Assure we're only keeping keys that we need for the current model component. 
        if not needed in key:
            continue

        # Update the key prefixes to match what the model expects.
        if prefix is not None:
            while key.count(prefix) > prefix_allowed_count:
                key = key.replace(prefix, '', 1)

        new_state_dict[key] = value
        
    return new_state_dict

def load_pretrained(model, path, DEVICE):
    checkpoint = torch.load(path, map_location=torch.device(DEVICE))
    checkpoint_model = checkpoint['model']
    
    if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')}

        checkpoint = remap_pretrained_keys_swin(model, checkpoint_model)
    else:
        raise NotImplementedError
    # MATIAS: If using ImageNet (3 channel) pretrained weights for Sentinel-2 (12 band) data
    if model.patch_embed.proj.weight.shape != checkpoint_model['patch_embed.proj.weight'].shape:
        temp = model.patch_embed.proj.weight.data.cpu()
        if checkpoint_model['patch_embed.proj.weight'].shape[1]==1:
            # greyscale pretrained model
            temp = checkpoint_model['patch_embed.proj.weight'].repeat(1, temp.shape[1],1,1)
        elif checkpoint_model['patch_embed.proj.weight'].shape[1] == 12 and temp.shape[1] == 3: 
            # For 12 band pretrained, the order is CGBR...
            temp[:,:,:,:] = checkpoint_model['patch_embed.proj.weight'][:,[3,2,1],:,:]
        elif checkpoint_model['patch_embed.proj.weight'].shape[1] == 8:
            #SpaceNet superres pretrain
            min_channels = min(temp.shape[1],  checkpoint_model['patch_embed.proj.weight'].shape[1])
            temp[:,:min_channels,:,:] = checkpoint_model['patch_embed.proj.weight'][:,:min_channels,:,:]
        else:
            temp[:,[3,2,1],:,:] = checkpoint_model['patch_embed.proj.weight']
        checkpoint_model['patch_embed.proj.weight'] = temp

    # msg = model.load_state_dict(checkpoint_model, strict=False)
    # print(msg)
    del checkpoint
    torch.cuda.empty_cache()

    return checkpoint_model


def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break
    
    weight = module.weight.detach()
    module.in_channels = new_in_channels
    
    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(
                module.out_channels,
                new_in_channels // module.groups,
                *module.kernel_size
            )
        )
        module.reset_parameters()
    
    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)
    
    else:
        new_weight = torch.Tensor(
            module.out_channels,
            new_in_channels // module.groups,
            *module.kernel_size
        )

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)


def replace_strides_with_dilation(module, dilation_rate):
    """Patch Conv2d modules replacing strides with dilation"""
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

            # Kostyl for EfficientNet
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()
