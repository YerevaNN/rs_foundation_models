from ..base import ClassificationHead, SegmentationHead, SegmentationModel
from ..encoders import get_encoder
from .seg_decoder import UPerNetDecoderSeg
import torch
from typing import Optional

class UPerNetSeg(SegmentationModel):
    """UPerNet_ is a fully convolution neural network for image semantic segmentation.

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
        freeze_encoder: bool = False,
        pretrained: bool = False,
        channels = [0, 1, 2],
        **kwargs
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.channels = channels

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UPerNetDecoderSeg(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            psp_channels=decoder_psp_channels,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
            pretrained=pretrained
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
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
        self.softmax = torch.nn.Softmax(dim=1)
        self.initialize()

    def base_forward(self, x, metadata=None):
        channels = self.channels
        """Sequentially pass `x1` `x2` trough model`s encoder, decoder and heads"""
        if self.freeze_encoder:
            with torch.no_grad():
                if 'cvit-pretrained' in self.encoder_name.lower():
                    f = self.encoder(x, channels)
                elif 'cvit' in self.encoder_name.lower():
                    channels = torch.tensor([channels]).cuda()
                    f = self.encoder(x, extra_tokens={"channels":channels})
                elif 'clay' in self.encoder_name.lower():
                    f = self.encoder(x, metadata)
                elif 'dofa' in self.encoder_name.lower():
                    f = self.encoder(x, metadata[0]['waves'])
                else:
                    f = self.encoder(x)
        else:
            if 'cvit-pretrained' in self.encoder_name.lower():
                f = self.encoder(x, channels)
            elif 'cvit' in self.encoder_name.lower():
                channels = torch.tensor([channels]).cuda()
                f = self.encoder(x, extra_tokens={"channels":channels})
            elif 'clay' in self.encoder_name.lower():
                f = self.encoder(x, metadata)
            elif 'dofa' in self.encoder_name.lower():
                f = self.encoder(x, metadata[0]['waves'])
            else:
                f = self.encoder(x)
                
        decoder_output = self.decoder(*f)

        # TODO: features = self.fusion_policy(features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            raise AttributeError("`classification_head` is not supported now.")
            # labels = self.classification_head(features[-1])
            # return masks, labels

        masks = self.softmax(masks)
        return masks

    def forward(self, x, metadata):
        return self.base_forward(x, metadata)