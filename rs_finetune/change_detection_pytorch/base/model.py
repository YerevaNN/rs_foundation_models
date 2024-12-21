import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def base_forward(self, x1, x2, metadata=None):
        channels = self.channels
        """Sequentially pass `x1` `x2` trough model`s encoder, decoder and heads"""
        if self.freeze_encoder:
            with torch.no_grad():
                if 'cvit' in self.encoder_name.lower():
                    channels = torch.tensor([channels]).cuda()
                    f1 = self.encoder(x1, extra_tokens={"channels":channels})
                    f2 = self.encoder(x2, extra_tokens={"channels":channels}) if self.siam_encoder else self.encoder_non_siam(x2, extra_tokens={"channels":channels})
                elif 'clay' in self.encoder_name.lower():
                    f1 = self.encoder(x1, metadata)
                    f2 = self.encoder(x2, metadata) if self.siam_encoder else self.encoder_non_siam(x2, metadata)
                else:
                    f1 = self.encoder(x1)
                    f2 = self.encoder(x2) if self.siam_encoder else self.encoder_non_siam(x2)
        else:
            if 'cvit' in self.encoder_name.lower():
                channels = torch.tensor([channels]).cuda()
                f1 = self.encoder(x1, extra_tokens={"channels":channels})
                f2 = self.encoder(x2, extra_tokens={"channels":channels}) if self.siam_encoder else self.encoder_non_siam(x2, extra_tokens={"channels":channels})
            elif 'clay' in self.encoder_name.lower():
                f1 = self.encoder(x1, metadata)
                f2 = self.encoder(x2, metadata) if self.siam_encoder else self.encoder_non_siam(x2, metadata)
            else:
                f1 = self.encoder(x1)
                f2 = self.encoder(x2) if self.siam_encoder else self.encoder_non_siam(x2)
                
        features = f1, f2
        decoder_output = self.decoder(*features)

        # TODO: features = self.fusion_policy(features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            raise AttributeError("`classification_head` is not supported now.")
            # labels = self.classification_head(features[-1])
            # return masks, labels

        return masks

    def forward(self, x1, x2, metadata):
        """Sequentially pass `x1` `x2` trough model`s encoder, decoder and heads"""
        return self.base_forward(x1, x2, metadata)

    def predict(self, x1, x2):
        """Inference method. Switch model to `eval` mode, call `.forward(x1, x2)` with `torch.no_grad()`

        Args:
            x1, x2: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x1, x2)

        return x
