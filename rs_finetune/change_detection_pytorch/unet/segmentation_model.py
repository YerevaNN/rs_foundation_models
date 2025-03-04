from typing import Optional, Union, List
from .seg_decoder import UnetDecoderSeg
from ..encoders import get_encoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead
import torch

class UnetSeg(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder* 
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial 
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
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
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        scales=[4, 2, 1, 0.5],
        channels = [0, 1, 2],
        enable_sample: bool = False,
        freeze_encoder = False,
        **kwargs
    ):
        super().__init__()

        self.channels = channels
        self.freeze_encoder = freeze_encoder
        self.encoder_name =  encoder_name

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            scales = scales,
            enable_sample=enable_sample,
        )

        self.decoder = UnetDecoderSeg(
            encoder_channels=(768, 768, 768, 768),
            decoder_channels=decoder_channels,
            n_blocks=len(scales),
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None
        self.softmax = torch.nn.Softmax(dim=1)

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def base_forward(self, x, metadata=None):
        channels = self.channels
        """Sequentially pass `x1` `x2` trough model`s encoder, decoder and heads"""
        if self.freeze_encoder:
            with torch.no_grad():
                if 'cvit' in self.encoder_name.lower():
                    channels = torch.tensor([channels]).cuda()
                    f = self.encoder(x, extra_tokens={"channels":channels})
                elif 'clay' in self.encoder_name.lower():
                    f = self.encoder(x, metadata)
                else:
                    f = self.encoder(x)
        else:
            if 'cvit' in self.encoder_name.lower():
                channels = torch.tensor([channels]).cuda()
                f = self.encoder(x, extra_tokens={"channels":channels})
            elif 'clay' in self.encoder_name.lower():
                f = self.encoder(x, metadata)
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