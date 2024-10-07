import torch.nn.functional as F
from torch import nn


def cus_sample(feat, **kwargs):
    """
    :param feat: input features
    :param kwargs: size or scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
