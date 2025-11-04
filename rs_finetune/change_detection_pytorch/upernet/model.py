from ..base import ClassificationHead, SegmentationHead, SegmentationModel
from ..encoders import get_encoder
from .decoder import UPerNetDecoder
from .decoder_pangea import SiamUPerNet

from typing import Optional
import torch

class UPerNet(SegmentationModel):
    """UPerNet_is a fully convolution neural network for image semantic segmentation.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_psp_channels: A number of filters in Spatial Pyramid
        decoder_pyramid_channels: A number of convolution filters in Feature Pyramid of FPN_
        decoder_segmentation_channels: A number of convolution filters in segmentation blocks of FPN_
        decoder_merge_policy: Determines how to merge pyramid features inside FPN. Available options are **add** and **cat**
        decoder_dropout: Spatial dropout rate in range (0, 1) for feature pyramid in FPN_
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
        siam_encoder: Whether using siamese branch. Default is True
        fusion_form: The form of fusing features from two branches. Available options are **"concat"**, **"sum"**, and **"diff"**.
            Default is **concat**

    Returns:
        ``torch.nn.Module``: **UPerNet**

    .. _UPerNet:
        https://arxiv.org/abs/1807.10221

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_psp_channels: int = 512,
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 256,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
        siam_encoder: bool = True,
        fusion_form: str = "concat",
        freeze_encoder: bool = False,
        pretrained: bool = False,
        channels = [0, 1, 2],
        out_size = 224,
        enable_sample: bool = False,
        enable_multiband_input: bool = False,
        multiband_channel_count: int = 12,
        **kwargs
    ):
        super().__init__()

        self.siam_encoder = siam_encoder
        self.encoder_name = encoder_name
        self.channels = channels
        self.enable_multiband_input = enable_multiband_input
        self.multiband_channel_count = multiband_channel_count

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            enable_sample=enable_sample,
        )

        if not self.siam_encoder:
            self.encoder_non_siam = get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )

        if enable_multiband_input:
            self._adapt_encoder_for_multiband(multiband_channel_count)
            if not self.siam_encoder:
                self._adapt_encoder_for_multiband_non_siam(multiband_channel_count)

        # self.decoder = UPerNetDecoder(
        #     encoder_channels=self.encoder.out_channels,
        #     encoder_depth=encoder_depth,
        #     psp_channels=decoder_psp_channels,
        #     pyramid_channels=decoder_pyramid_channels,
        #     segmentation_channels=decoder_segmentation_channels,
        #     dropout=decoder_dropout,
        #     merge_policy=decoder_merge_policy,
        #     fusion_form=fusion_form,
        #     pretrained=pretrained
        # )
        self.decoder = SiamUPerNet(
            encoder_channels=self.encoder.output_channels,
            num_classes=classes,
            finetune=freeze_encoder,
            strategy=fusion_form,
            out_size=out_size,
            channels=self.encoder.output_channels
        )

        self.segmentation_head = SegmentationHead(
            in_channels=out_size,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
            align_corners=False,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "upernet-{}".format(encoder_name)
        self.freeze_encoder = freeze_encoder
        self.initialize()
        
    def _adapt_encoder_for_multiband(self, new_in_channels: int):
        from classifier_utils import adapt_rgb_conv_layer_to_multiband, adapt_rgb_conv3d_layer_to_multiband
        
        if hasattr(self.encoder, 'dinov3'):
            if hasattr(self.encoder.dinov3, 'embeddings'):
                embeddings = self.encoder.dinov3.embeddings
                if hasattr(embeddings, 'patch_embeddings'):
                    patch_embeddings = embeddings.patch_embeddings
                    if isinstance(patch_embeddings, torch.nn.Conv2d):
                        old_conv = patch_embeddings
                        embeddings.patch_embeddings = adapt_rgb_conv_layer_to_multiband(
                            old_conv=old_conv, 
                            new_in_channels=new_in_channels
                        )
        elif hasattr(self.encoder, 'backbone') and hasattr(self.encoder.backbone, 'patch_embed') and hasattr(self.encoder.backbone.patch_embed, 'proj'):
            old_conv = self.encoder.backbone.patch_embed.proj
            if isinstance(old_conv, torch.nn.Conv3d):
                self.encoder.backbone.patch_embed.proj = adapt_rgb_conv3d_layer_to_multiband(
                    old_conv=old_conv, 
                    new_in_channels=new_in_channels
                )
            else:
                self.encoder.backbone.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(
                    old_conv=old_conv, 
                    new_in_channels=new_in_channels
                )
            print(f"New conv shape: {self.encoder.backbone.patch_embed.proj.weight.shape}")
            print("+"*100)
        elif hasattr(self.encoder, 'model'):
            if hasattr(self.encoder.model, 'conv1'):
                old_conv = self.encoder.model.conv1
                self.encoder.model.conv1 = adapt_rgb_conv_layer_to_multiband(
                    old_conv=old_conv, 
                    new_in_channels=new_in_channels
                )
            elif hasattr(self.encoder.model, 'patch_embed') and hasattr(self.encoder.model.patch_embed, 'proj'):
                old_conv = self.encoder.model.patch_embed.proj
                if isinstance(old_conv, torch.nn.Conv3d):
                    self.encoder.model.patch_embed.proj = adapt_rgb_conv3d_layer_to_multiband(
                        old_conv=old_conv, 
                        new_in_channels=new_in_channels
                    )
                else:
                    self.encoder.model.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(
                        old_conv=old_conv, 
                        new_in_channels=new_in_channels
                    )
        elif hasattr(self.encoder, 'patch_embed') and hasattr(self.encoder.patch_embed, 'proj'):
            old_conv = self.encoder.patch_embed.proj
            if isinstance(old_conv, torch.nn.Conv3d):
                self.encoder.patch_embed.proj = adapt_rgb_conv3d_layer_to_multiband(
                    old_conv=old_conv, 
                    new_in_channels=new_in_channels
                )
            else:
                self.encoder.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(
                    old_conv=old_conv, 
                    new_in_channels=new_in_channels
                )

        if hasattr(self.encoder, 'output_channels') and isinstance(self.encoder.output_channels, tuple):
            # Replace the first element (input channels) with the new channel count
            old_channels = list(self.encoder.output_channels)
            old_channels[0] = new_in_channels
            self.encoder.output_channels = tuple(old_channels)

    def _adapt_encoder_for_multiband_non_siam(self, new_in_channels: int):
        from classifier_utils import adapt_rgb_conv_layer_to_multiband, adapt_rgb_conv3d_layer_to_multiband
        
        if hasattr(self.encoder_non_siam, 'model'):
            if hasattr(self.encoder_non_siam.model, 'conv1'):
                old_conv = self.encoder_non_siam.model.conv1
                self.encoder_non_siam.model.conv1 = adapt_rgb_conv_layer_to_multiband(
                    old_conv=old_conv, 
                    new_in_channels=new_in_channels
                )
            elif hasattr(self.encoder_non_siam.model, 'patch_embed') and hasattr(self.encoder_non_siam.model.patch_embed, 'proj'):
                old_conv = self.encoder_non_siam.model.patch_embed.proj
                if isinstance(old_conv, torch.nn.Conv3d):
                    self.encoder_non_siam.model.patch_embed.proj = adapt_rgb_conv3d_layer_to_multiband(
                        old_conv=old_conv, 
                        new_in_channels=new_in_channels
                    )
                else:
                    self.encoder_non_siam.model.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(
                        old_conv=old_conv, 
                        new_in_channels=new_in_channels
                    )
        elif hasattr(self.encoder_non_siam, 'patch_embed') and hasattr(self.encoder_non_siam.patch_embed, 'proj'):
            old_conv = self.encoder_non_siam.patch_embed.proj
            if isinstance(old_conv, torch.nn.Conv3d):
                self.encoder_non_siam.patch_embed.proj = adapt_rgb_conv3d_layer_to_multiband(
                    old_conv=old_conv, 
                    new_in_channels=new_in_channels
                )
            else:
                self.encoder_non_siam.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(
                    old_conv=old_conv, 
                    new_in_channels=new_in_channels
                )

    def forward(self, x1, x2, metadata):
        """Sequentially pass `x1` `x2` trough model`s encoder, decoder and heads"""
        if self.enable_multiband_input:
            if x1.shape[1] < self.multiband_channel_count:
                num_missing = self.multiband_channel_count - x1.shape[1]
                zeros = torch.zeros(x1.shape[0], num_missing, x1.shape[2], x1.shape[3], dtype=x1.dtype, device=x1.device)
                x1 = torch.cat([x1, zeros], dim=1)
            if x2.shape[1] < self.multiband_channel_count:
                num_missing = self.multiband_channel_count - x2.shape[1]
                zeros = torch.zeros(x2.shape[0], num_missing, x2.shape[2], x2.shape[3], dtype=x2.dtype, device=x2.device)
                x2 = torch.cat([x2, zeros], dim=1)
        
        return self.base_forward(x1, x2, metadata)
