import torch.nn as nn
import torch

from . import base
from . import functional as F
from ..base.modules import Activation

# See change_detection_pytorch/losses
# class JaccardLoss(base.Loss):
#
#     def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels
#
#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return 1 - F.jaccard(
#             y_pr, y_gt,
#             eps=self.eps,
#             threshold=None,
#             ignore_channels=self.ignore_channels,
#         )
#
#
# class DiceLoss(base.Loss):
#
#     def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.beta = beta
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels
#
#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return 1 - F.f_score(
#             y_pr, y_gt,
#             beta=self.beta,
#             eps=self.eps,
#             threshold=None,
#             ignore_channels=self.ignore_channels,
#         )


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_WithLogitsLoss = nn.BCEWithLogitsLoss()
        # BCELoss()
        
    @property
    def __name__(self):
        return "dice_bce_loss"

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_out, y_true):
        num_classes = y_out.size(1)
        y_true = torch.nn.functional.one_hot(y_true, num_classes=num_classes)
        y_true = y_true.permute(0, 3, 1, 2).float()

        y_pred = torch.sigmoid(y_out)
        
        a =  self.bce_WithLogitsLoss(y_out, y_true)
        b =  self.soft_dice_loss(y_pred, y_true)
        
        return (0.5 * a + 0.5 * b) * 2
    