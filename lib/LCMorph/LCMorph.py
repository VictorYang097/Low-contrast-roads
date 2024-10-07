import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.LCMorph.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from lib.LCMorph.aspp import build_aspp
from lib.LCMorph.decoder import build_decoder
from lib.LCMorph.resnet import build_backbone
from lib.LCMorph.FAM import FAM
from lib.LCMorph.utils import cus_sample, BasicConv2d


# Refine features based on a coarse mask
class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.upsample = cus_sample

    def forward(self, attention, x2, x3, x4):
        x2 = x2 + torch.mul(x2, self.upsample(attention, scale_factor=2))
        x3 = x3 + torch.mul(x3, attention)
        x4 = x4 + torch.mul(x4, attention)

        return x2, x3, x4
    

# Fuse frequency features at different levels
class NeighborFusionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborFusionDecoder, self).__init__()
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        
        x2_1 = self.conv_upsample1(x1) * x2  
        
        x3_1 = self.conv_upsample2(x2_1) * (self.conv_upsample3(x2) * x3) 
       
        x2_2 = torch.cat((x2_1, self.conv_upsample4(x1_1)), 1) 
        x2_2 = self.conv_concat2(x2_2)  # (128*22*22)
        
        x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2)), 1) 
        x3_2 = self.conv_concat3(x3_2)  

        x = self.conv4(x3_2)
        x = self.conv5(x)  

        return x                                


# Model structure of our LCMorph
class LCMorph(nn.Module):
    def __init__(self, backbone='resnet101', output_stride=8, num_classes=1, channel=64,
                 sync_bn=False, freeze_bn=False):
        super(LCMorph, self).__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        """
         Frequency-enhanced localization
        """
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        
        self.conv3 = BasicConv2d(512, 128, kernel_size=1, stride=1, padding=0, dilation=1, relu=True, bn=True)
        self.conv4 = BasicConv2d(1024, 256, kernel_size=1, stride=1, padding=0, dilation=1, relu=True, bn=True)
        self.conv5 = BasicConv2d(2048, 512, kernel_size=1, stride=1, padding=0, dilation=1, relu=True, bn=True)
        
        self.fam3 = FAM(128, channel)
        self.fam4 = FAM(256, channel)
        self.fam5 = FAM(512, channel)

        self.NFD = NeighborFusionDecoder(channel)

        self.aspp = build_aspp('resnet101', output_stride, BatchNorm)

        self.refine = Refine()

        """
         Morphology-enhanced extraction
        """
        self.decoder = build_decoder(num_classes, 'resnet101', BatchNorm)

        self.seg_branch = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, num_classes, kernel_size=1, stride=1))

        self.freeze_bn = freeze_bn

    def forward(self, input):
        e2, e3, e4, e5 = self.backbone(input)

        f3 = self.fam3(self.conv3(e3))
        f4 = self.fam4(self.conv4(e4))
        f5 = self.fam5(self.conv5(e5))

        locate = self.NFD(f5, f4, f3)

        x2, x3, x4 = self.refine(locate.sigmoid(), e2, e3, e4)

        x5 = self.aspp(e5)

        x = self.decoder(x2, x3, x4, x5)
        seg = self.seg_branch(x)
        coarse = F.interpolate(locate, scale_factor=8, mode='bilinear', align_corners=False)

        return torch.sigmoid(coarse), torch.sigmoid(seg)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()
