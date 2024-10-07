import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.LCMorph.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from lib.LCMorph.DSConv import DySnakeConv


class MPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm, inp=False):
        super(MPBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.bn1 = BatchNorm(in_channels // 2)
        self.relu1 = nn.ReLU()
        self.inp = inp

        self.dsconv = DySnakeConv(inc=in_channels // 2, ouc=out_channels, k=3, kDSC=9, act=True)

        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.dsconv(x)

        if self.inp:
            x = F.interpolate(x, scale_factor=2)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                

class DSConvDecoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(DSConvDecoder, self).__init__()
        if backbone == 'resnet101':
            in_inplanes = 256
        else:
            raise NotImplementedError

        self.decoder4 = MPBlock(in_inplanes, 256, BatchNorm)
        self.decoder3 = MPBlock(512, 128, BatchNorm)
        self.decoder2 = MPBlock(256, 64, BatchNorm, inp=True)
        self.decoder1 = MPBlock(128, 64, BatchNorm, inp=True)

        self.conv_e4 = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())

        self.conv_e3 = nn.Sequential(nn.Conv2d(512, 128, 1, bias=False),
                                     BatchNorm(128),
                                     nn.ReLU())

        self.conv_e2 = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False),
                                     BatchNorm(64),
                                     nn.ReLU())

        self._init_weight()

    def forward(self, e2, e3, e4, e5):
        d4 = torch.cat((self.decoder4(e5), self.conv_e4(e4)), dim=1)
        d3 = torch.cat((self.decoder3(d4), self.conv_e3(e3)), dim=1)
        d2 = torch.cat((self.decoder2(d3), self.conv_e2(e2)), dim=1)
        d1 = self.decoder1(d2)
        x = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, backbone, BatchNorm):
    return DSConvDecoder(num_classes, backbone, BatchNorm)
