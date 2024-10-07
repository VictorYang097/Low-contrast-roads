import torch
import torch.nn.functional as F


def dice_bce_loss_func(y_pred, y_true):
    """
    Segmentation loss
    dice loss + bce loss
    """
    a = F.binary_cross_entropy(y_pred, y_true)

    smooth = 0.0
    i = torch.sum(y_true)
    j = torch.sum(y_pred)
    intersection = torch.sum(y_true * y_pred)
    score = (2. * intersection + smooth) / (i + j + smooth)
    b = 1 - score.mean()

    return a + b
