import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from methods.module.base_model import BasicModelClass
from methods.module.conv_block import ConvBNReLU
from utils.builder import MODELS
from utils.ops import cus_sample
# from methods import FasterNet
from functools import partial
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from einops import rearrange
from methods import pvtv2_encoder
from methods import mamba2
# from mamba_ssm.modules.mamba2_simple import Mamba2Simple
from safetensors.torch import load_file
from methods import pvt
from transformers import AutoFeatureExtractor, SwinForImageClassification, SwinModel, ResNetModel



class ConvBNReLU_BR(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, d=1, p=1):
        super(ConvBNReLU_BR, self).__init__()
        self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=k, dilation=d, padding=p)
        # self.conv2 = ConvBNReLU(out_dim, out_dim // 2, kernel_size=3, dilation=7, padding=7)
        # self.conv3 = ConvBNReLU(out_dim // 2, out_dim, kernel_size=3, dilation=1, padding=1)
        self.conv4 = ConvBNReLU(out_dim, out_dim * 2, kernel_size=1, dilation=1, padding=0)
        self.conv5 = ConvBNReLU(out_dim * 2, out_dim, kernel_size=1, dilation=1, padding=0)
        # self.conv2 = MMA(out_dim, out_dim)
        # self.conv2 = ConvBNReLU(out_dim * 2, out_dim, kernel_size=k, dilation=d, padding=p)

        # self.conv6 = ConvBNReLU(out_dim, out_dim // 8, kernel_size=1, dilation=1, padding=0)
        # self.conv7 = ConvBNReLU(out_dim // 8, out_dim, kernel_size=1, dilation=1, padding=0)


    def forward(self, x):
        conv1 = self.conv1(x)
        # conv1 = self.conv1(torch.cat((x, x1), dim=1))
        # conv2 = self.conv3(self.conv2(conv1))
        conv3 = self.conv5(self.conv4(conv1))
        # conv3 = self.conv2(conv1, conv3)
        # conv3 = self.conv2(torch.cat((conv1, conv3), dim=1))
        # conv4 = self.conv7(self.conv6(conv1))

        return conv1 + conv3
        # return conv3


class ConvBNReLU_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, d=1, p=1):
        super(ConvBNReLU_MLP, self).__init__()
        self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=k, dilation=d, padding=p)
        # self.conv2 = ConvBNReLU(out_dim, out_dim // 2, kernel_size=3, dilation=7, padding=7)
        # self.conv3 = ConvBNReLU(out_dim // 2, out_dim, kernel_size=3, dilation=1, padding=1)
        self.conv4 = ConvBNReLU(out_dim, out_dim * 2, kernel_size=3, dilation=7, padding=7)
        self.conv5 = ConvBNReLU(out_dim * 2, out_dim, kernel_size=3, dilation=1, padding=1)

        # self.conv6 = ConvBNReLU(out_dim, out_dim // 8, kernel_size=1, dilation=1, padding=0)
        # self.conv7 = ConvBNReLU(out_dim // 8, out_dim, kernel_size=1, dilation=1, padding=0)


    def forward(self, x):
        conv1 = self.conv1(x)
        # conv1 = self.conv1(torch.cat((x, x1), dim=1))
        # conv2 = self.conv3(self.conv2(conv1))
        conv3 = self.conv5(self.conv4(conv1))
        # conv4 = self.conv7(self.conv6(conv1))

        # return conv1 + conv2 + conv3
        return conv1 + conv3



class ASPP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ASPP, self).__init__()
        self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
        self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
        self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
        self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))


class dASPP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(dASPP, self).__init__()

        out_dim = out_dim // 1

        self.conv01 = ConvBNReLU(out_dim * 1, out_dim, 3, 1, 2, dilation=2)

        self.conv02 = ConvBNReLU(out_dim * 2, out_dim, 3, 1, 3, dilation=3)
        self.conv021 = ConvBNReLU(out_dim * 2, out_dim, 3, 1, 2, dilation=2)

        self.conv03 = ConvBNReLU(out_dim * 3, out_dim, 3, 1, 5, dilation=5)
        self.conv032 = ConvBNReLU(out_dim * 2, out_dim, 3, 1, 3, dilation=3)
        self.conv031 = ConvBNReLU(out_dim * 2, out_dim, 3, 1, 2, dilation=2)

        self.conv04 = ConvBNReLU(out_dim * 4, out_dim, 3, 1, 7, dilation=7)
        self.conv043 = ConvBNReLU(out_dim * 2, out_dim, 3, 1, 5, dilation=5)
        self.conv042 = ConvBNReLU(out_dim * 2, out_dim, 3, 1, 3, dilation=3)
        self.conv041 = ConvBNReLU(out_dim * 2, out_dim, 3, 1, 2, dilation=2)

        self.conv05 = ConvBNReLU(out_dim * 5, out_dim, 3, 1, 9, dilation=9)
        self.conv054 = ConvBNReLU(out_dim * 2, out_dim, 3, 1, 7, dilation=7)
        self.conv053 = ConvBNReLU(out_dim * 2, out_dim, 3, 1, 5, dilation=5)
        self.conv052 = ConvBNReLU(out_dim * 2, out_dim, 3, 1, 3, dilation=3)
        self.conv051 = ConvBNReLU(out_dim * 2, out_dim, 3, 1, 2, dilation=2)






        # self.conv01 = ConvBNReLU_BR(in_dim, out_dim, 3, 1, 1)
        # self.conv02 = ConvBNReLU_BR(out_dim * 2, out_dim, 3, 2, 2)
        # self.conv03 = ConvBNReLU_BR(out_dim * 2, out_dim, 3, 3, 3)
        # self.conv04 = ConvBNReLU_BR(out_dim * 2, out_dim, 3, 5, 5)
        # self.conv05 = ConvBNReLU_BR(out_dim * 2, out_dim, 3, 7, 7)

        # self.conv01 = ConvBNReLU(in_dim, out_dim, 3, 1, 1, dilation=1)
        # self.conv02 = ConvBNReLU(out_dim * 2, out_dim, 3, 1, 2, dilation=2)
        # self.conv021 = ConvBNReLU(out_dim * 1, out_dim, 3, 1, 2, dilation=2)
        # self.conv03 = ConvBNReLU(out_dim * 2, out_dim, 3, 1, 3, dilation=3)
        # self.conv031 = ConvBNReLU(out_dim * 1, out_dim, 3, 1, 3, dilation=3)
        # self.conv04 = ConvBNReLU(out_dim * 2, out_dim, 3, 1, 5, dilation=5)
        # self.conv041 = ConvBNReLU(out_dim * 1, out_dim, 3, 1, 5, dilation=5)
        # self.conv05 = ConvBNReLU(out_dim * 2, out_dim, 3, 1, 7, dilation=7)


        # self.conv1 = ConvBNReLU(out_dim * 5, out_dim, 3, 1, 1, dilation=1)
        # self.conv2 = ConvBNReLU(out_dim * 5, out_dim, 3, 1, 2, dilation=2)
        # self.conv3 = ConvBNReLU(out_dim * 5, out_dim, 3, 1, 3, dilation=3)
        # self.conv4 = ConvBNReLU(out_dim * 5, out_dim, 3, 1, 5, dilation=5)
        # self.conv5 = ConvBNReLU(out_dim * 5, out_dim, 3, 1, 7, dilation=7)

        # self.conv010 = ConvBNReLU(out_dim, out_dim, 3, 1, 7, dilation=7)
        # self.conv020 = ConvBNReLU(out_dim * 1, out_dim, 3, 1, 5, dilation=5)
        # self.conv030 = ConvBNReLU(out_dim * 1, out_dim, 3, 1, 3, dilation=3)
        # self.conv040 = ConvBNReLU(out_dim * 1, out_dim, 3, 1, 2, dilation=2)
        # self.conv050 = ConvBNReLU(out_dim * 1, out_dim, 3, 1, 1, dilation=1)

        # self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        # self.conv2 = MBConvBlock(in_dim, out_dim, k_d=3, p_d=2, d=2)
        # self.conv3 = MBConvBlock(in_dim, out_dim, k_d=3, p_d=5, d=5)
        # self.conv4 = MBConvBlock(in_dim, out_dim, k_d=3, p_d=7, d=7)
        # self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)

        # self.m1 = MMA(out_dim, out_dim)
        # self.m2 = MMA(out_dim, out_dim)
        # self.m3 = MMA(out_dim, out_dim)
        # self.m4 = MMA(out_dim, out_dim)
        # self.m5 = MMA(out_dim, out_dim)

        self.fuse = ConvBNReLU(out_dim * 5, out_dim * 1, 3, 1, 1)

    def forward(self, x):
        # x = self.conv0(x)

        # conv1 = self.conv1(x)
        # conv2 = self.conv2(x)
        # conv3 = self.conv3(x)
        # conv4 = self.conv4(x)
        # # conv5 = self.conv5(x)
        # out = self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

        # x1, x2, x3, x4, x5 = self.conv0(x).chunk(5, dim=1)
        # x0 = self.conv00(x)

        # x1 = self.conv05(x)
        # x2 = self.conv04(x1)
        # x3 = self.conv03(x2)
        # x4 = self.conv02(x3)
        # x5 = self.conv01(x4)

        # conv1 = self.conv01(x)
        # conv2 = self.m2(x, self.conv02(x))
        # conv3 = self.m3(conv2, self.conv03(conv2))
        # conv4 = self.m4(conv2, self.conv04(conv2))
        # conv5 = self.m5(conv4, self.conv05(conv4))

        # x = torch.cat((self.conv01(x), self.conv02(x), self.conv03(x), self.conv04(x), self.conv05(x)), dim=1)

        # x1 = self.conv05(x) + x
        # x2 = self.conv04(x1) + x + x1
        # x3 = self.conv03(x2) + x + x1 + x2
        # x4 = self.conv02(x3) + x + x1 + x2 + x3
        # x5 = self.conv01(x4) + x + x1 + x2 + x3 + x4
        #
        # conv1 = self.conv1(torch.cat((x1, x2, x3, x4, x5), 1))
        # conv2 = self.conv2(torch.cat((conv1, x2, x3, x4, x5), 1))
        # conv3 = self.conv3(torch.cat((conv1, conv2, x3, x4, x5), 1))
        # conv4 = self.conv4(torch.cat((conv1, conv2, conv3, x4, x5), 1))
        # conv5 = self.conv5(torch.cat((conv1, conv2, conv3, conv4, x5), 1))

        # x1 = self.conv01(x)
        # x2 = self.conv02(torch.cat((x1, x), 1))
        # x3 = self.conv03(torch.cat((x2, (x1 + x)), 1))
        # x4 = self.conv04(torch.cat((x3, (x2 + x1 + x)), 1))
        # x5 = self.conv05(torch.cat((x4, (x3 + x2 + x1 + x)), 1))

        # x1 = self.m1(x, self.conv01(x))
        # x2 = self.conv02(torch.cat((x1, x), 1))
        # x21 = self.conv021((x1 + x))
        # x3 = self.conv03(torch.cat((x2, x21), 1))
        # x31 = self.conv031((x2 + x21))
        # x4 = self.conv04(torch.cat((x3, x31), 1))
        # x41 = self.conv041((x3 + x31))
        # x5 = self.conv05(torch.cat((x4, x41), 1))



        # out1 = self.conv010(conv5)
        # out2 = self.conv020(conv4) + out1
        # out3 = self.conv030(conv3) + out2
        # out4 = self.conv040(conv2) + out3
        # out5 = self.conv050(conv1) + out4

        # m1 = self.m1(x1)
        # m2 = self.m2(x2)
        # m3 = self.m3(x3)
        # m4 = self.m4(x4)
        # m5 = self.m5(x5)

        # x1 = self.conv01(x)
        # x2 = self.conv02(torch.cat((x1, x), 1))
        # x3 = self.conv03(torch.cat((x2, x1), 1))
        # x4 = self.conv04(torch.cat((x3, x2), 1))
        # x5 = self.conv05(torch.cat((x4, x3), 1))

        x1 = self.conv01(x)

        x2 = self.conv02(torch.cat((x1, x), 1))
        x1 = self.conv021(torch.cat((x2, x1), 1))

        x3 = self.conv03(torch.cat((x2, x1, x), 1))
        x2 = self.conv032(torch.cat((x3, x2), 1))
        x1 = self.conv031(torch.cat((x2, x1), 1))

        x4 = self.conv04(torch.cat((x3, x2, x1, x), 1))
        x3 = self.conv043(torch.cat((x4, x3), 1))
        x2 = self.conv042(torch.cat((x3, x2), 1))
        x1 = self.conv041(torch.cat((x2, x1), 1))

        x5 = self.conv05(torch.cat((x4, x3, x2, x1, x), 1))
        conv5 = x5
        conv4 = self.conv054(torch.cat((x5, x4), 1))
        conv3 = self.conv053(torch.cat((x4, x3), 1))
        conv2 = self.conv052(torch.cat((x3, x2), 1))
        conv1 = self.conv051(torch.cat((x2, x1), 1))






        # conv0 = self.conv0(torch.cat((x1, x2, x3, x4, x5), 1))

        # conv1 = self.conv1(torch.cat((x1, x2, x3, x4, x5), 1))
        # conv2 = self.conv2(torch.cat((conv1, x2, x3, x4, x5), 1))
        # conv3 = self.conv3(torch.cat((conv1, conv2, x3, x4, x5), 1))
        # conv4 = self.conv4(torch.cat((conv1, conv2, conv3, x4, x5), 1))
        # conv5 = self.conv5(torch.cat((conv1, conv2, conv3, conv4, x5), 1))



        # conv1 = self.conv1(x) * self.m1(x)
        # conv2 = self.conv2(torch.cat((x, conv1), 1)) * self.m2(conv1)
        # conv3 = self.conv3(torch.cat((x, conv1, conv2), 1)) * self.m1(conv2)
        # conv4 = self.conv4(torch.cat((x, conv1, conv2, conv3), 1)) * self.m1(conv3)
        # conv5 = self.conv5(torch.cat((x, conv1, conv2, conv3, conv4), 1)) * self.m1(conv4)
        #
        # conv1 = self.conv1(x)
        # conv2 = self.conv2(torch.cat((x, conv1), 1))
        # conv3 = self.conv3(torch.cat((x, conv1, conv2), 1))
        # conv4 = self.conv4(torch.cat((x, conv1, conv2, conv3), 1))
        # conv5 = self.conv5(torch.cat((x, conv1, conv2, conv3, conv4), 1))

        # conv2 = self.conv2(torch.cat((x2, conv1), 1))
        # conv3 = self.conv3(torch.cat((x3, conv2), 1))
        # conv4 = self.conv4(torch.cat((x4, conv3), 1))
        # conv5 = self.conv5(torch.cat((x5, conv4), 1))

        # x4 = self.conv6(torch.cat((x5, x4), 1))
        # x3 = self.conv7(torch.cat((x4, x3), 1))
        # x2 = self.conv8(torch.cat((x3, x2), 1))
        # conv1 = self.conv9(torch.cat((x2, x1), 1))
        #
        # conv2 = self.conv10(torch.cat((x2, conv1), 1))
        # conv3 = self.conv11(torch.cat((x3, conv2), 1))
        # conv4 = self.conv12(torch.cat((x4, conv3), 1))
        # conv5 = self.conv13(torch.cat((x5, conv4), 1))

        out = self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1)) # CBR
        # out = x5
        # out = self.fuse(conv1 * m1 + conv2 * m2 + conv3 * m3 + conv4 * m4 + conv5 * m5)
        # out = self.fuse(conv1 + conv2 + conv3 + conv4 + conv5)
        # out = self.fuse(torch.cat((conv2, conv4, conv5), 1))
        # out = self.fuse(x5)
        return out


class dASPP1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(dASPP1, self).__init__()

        out_dim = out_dim // 1

        # self.conv01 = ConvBNReLU(in_dim, out_dim, 3, 1, 1, dilation=1)
        self.conv02 = ConvBNReLU(out_dim // 4, out_dim // 4, 3, 1, 2, dilation=2)
        self.conv03 = ConvBNReLU(out_dim // 4, out_dim // 4, 3, 1, 3, dilation=3)
        self.conv04 = ConvBNReLU(out_dim // 4, out_dim // 4, 3, 1, 5, dilation=5)
        self.conv05 = ConvBNReLU(out_dim // 4, out_dim // 4, 3, 1, 7, dilation=7)

        # self.conv1 = ConvBNReLU(out_dim * 1, out_dim, 3, 1, 1, dilation=1)
        self.conv2 = ConvBNReLU(out_dim * 1, out_dim // 4, 3, 1, 2, dilation=2)
        self.conv3 = ConvBNReLU(out_dim * 1, out_dim // 4, 3, 1, 3, dilation=3)
        self.conv4 = ConvBNReLU(out_dim * 1, out_dim // 4, 3, 1, 5, dilation=5)
        self.conv5 = ConvBNReLU(out_dim * 1, out_dim // 4, 3, 1, 7, dilation=7)

        # self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        # self.conv2 = MBConvBlock(in_dim, out_dim, k_d=3, p_d=2, d=2)
        # self.conv3 = MBConvBlock(in_dim, out_dim, k_d=3, p_d=5, d=5)
        # self.conv4 = MBConvBlock(in_dim, out_dim, k_d=3, p_d=7, d=7)
        # self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)

        # self.m1 = MSCA(out_dim, 1)
        self.m2 = MSCA(out_dim // 4, 1)
        self.m3 = MSCA(out_dim // 4, 1)
        self.m4 = MSCA(out_dim // 4, 1)
        self.m5 = MSCA(out_dim // 4, 1)

        self.fuse = ConvBNReLU(out_dim * 1, out_dim * 1, 3, 1, 1)

    def forward(self, x):
        x2, x3, x4, x5 = x.chunk(4, dim=1)
        # x0 = self.conv00(x)

        # x1 = self.conv05(x)
        # x2 = self.conv04(x1)
        # x3 = self.conv03(x2)
        # x4 = self.conv02(x3)
        # x5 = self.conv01(x4)

        # x1 = self.conv01(x)
        x2 = self.m2(self.conv02(x2))
        x3 = self.m3(self.conv03(x3))
        x4 = self.m4(self.conv04(x4))
        x5 = self.m5(self.conv05(x5))
        # x = (torch.cat((x2, x3, x4, x5), 1))

        conv2 = self.conv2(torch.cat((x2, x3, x4, x5), 1))
        conv3 = self.conv3(torch.cat((conv2, x3, x4, x5), 1))
        conv4 = self.conv4(torch.cat((conv2, conv3, x4, x5), 1))
        conv5 = self.conv5(torch.cat((conv2, conv3, conv4, x5), 1))
        # conv2 = self.conv2(torch.cat((x2, conv1), 1))
        # conv3 = self.conv3(torch.cat((x3, conv1, conv2), 1))
        # conv4 = self.conv4(torch.cat((x4, conv1, conv2, conv3), 1))
        # conv5 = self.conv5(torch.cat((x5, conv1, conv2, conv3, conv4), 1))

        # conv1 = self.conv1(x)
        # conv2 = self.conv2(torch.cat((x, conv1), 1))
        # conv3 = self.conv3(torch.cat((x, conv1, conv2), 1))
        # conv4 = self.conv4(torch.cat((x, conv1, conv2, conv3), 1))
        # conv5 = self.conv5(torch.cat((x, conv1, conv2, conv3, conv4), 1))

        out = self.fuse(torch.cat((conv2, conv3, conv4, conv5), 1))
        # out = self.fuse(conv1 + conv2 + conv3 + conv4 + conv5)
        # out = self.fuse(torch.cat((conv2, conv4, conv5), 1))
        # out = self.fuse(conv5)
        return out


# class M_add(nn.Module):
#
#     def __init__(self, inplanes, r=False):
#         super(M_add, self).__init__()

#         # self.conv_1 = nn.Conv2d(inplanes * 2, 2, 1, 1, 0)
#         self.conv_2 = MBConvBlock(inplanes * 2, inplanes)
#         # self.r = r
#         # self.avg1 = nn.AdaptiveAvgPool2d((1, 1))
#
#     def forward(self, x1, x2):
#         # x0_0 = self.conv_s0(x)
#         global x0
#
#         # avg1 = self.avg1(x1)
#         # avg2 = self.avg1(x2)
#         # a1, a2 = self.conv_1(torch.cat([x1, x2], dim=1)).softmax(dim=1).chunk(2, dim=1)
#         x0 = self.conv_2(torch.cat([x1, x2], dim=1))
#         # x0 = self.conv_2(x1 + x2)
#         # x0 = a1 * x1 + a2 * x2
#         # x0 = self.conv_2(a1 * x1 + a2 * x2)
#
#         return x0


# class M_fuse(nn.Module):
#
#     def __init__(self, inplanes):
#         super(M_fuse, self).__init__()
#
#         self.conv_1 = ConvBNReLU(inplanes, inplanes // 2, 1, 1, 0)
#         self.conv_2 = ConvBNReLU(inplanes, inplanes, 1, 1, 0)
#         # self.conv_3 = ConvBNReLU(inplanes * 2, inplanes, 3, 1, 1)
#         # self.conv_4 = ConvBNReLU(inplanes * 2, inplanes, 3, 1, 1)
#
#     def forward(self, x1, x2):
#         x1_a, x1_b = x1.chunk(2, dim=1)
#         x2_a, x2_b = x2.chunk(2, dim=1)
#         x3_a = x1_a + x2_a
#         x3_b = self.conv_1(torch.cat([x1_b, x2_b], dim=1))
#         x3 = self.conv_2(torch.cat([x3_a, x3_b], dim=1))
#
#         return x3


class TransLayer(nn.Module):
    def __init__(self, out_c, last_module=dASPP):
        super().__init__()
        self.c5_down = nn.Sequential(
            ConvBNReLU(512, out_c, 3, 1, 1), # pvt、pvt_v2
            # ConvBNReLU(1024, out_c, 3, 1, 1), # swin-base
            # ConvBNReLU(2048, out_c, 3, 1, 1), # ResNet
        )
        self.c4_down = nn.Sequential(
            ConvBNReLU(320, out_c, 3, 1, 1), # pvt、pvt_v2
            # ConvBNReLU(512, out_c, 3, 1, 1), # swin-base
            # ConvBNReLU(1024, out_c, 3, 1, 1), # ResNet
        )
        self.c3_down = nn.Sequential(
            ConvBNReLU(128, out_c, 3, 1, 1), # pvt、pvt_v2
            # ConvBNReLU(256, out_c, 3, 1, 1), # swin-base
            # ConvBNReLU(512, out_c, 3, 1, 1), # ResNet
        )
        self.c2_down = nn.Sequential(
            ConvBNReLU(64, out_c, 3, 1, 1), # pvt、pvt_v2
            # ConvBNReLU(128, out_c, 3, 1, 1), # swin-base
            # ConvBNReLU(256, out_c, 3, 1, 1), # ResNet
        )
        self.c1_down = nn.Sequential(
            ConvBNReLU(64, out_c, 3, 1, 1), # pvt、pvt_v2
            # ConvBNReLU(128, out_c, 3, 1, 1), # swin-base
            # ConvBNReLU(256, out_c, 3, 1, 1), # ResNet
        )

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        # assert len(xs) == 5
        # print(xs)
        c1, c2, c3, c4, c5 = xs
        # c2, c3, c4, c5 = xs
        c5 = self.c5_down(c5)
        # c5 = self.conv_mlp5(c5) + c5
        c4 = self.c4_down(c4)
        # c4 = self.conv_mlp4(c4) + c4
        c3 = self.c3_down(c3)
        # c3 = self.conv_mlp3(c3) + c3
        c2 = self.c2_down(c2)
        # c2 = self.conv_mlp2(c2) + c2
        c1 = self.c1_down(c1)
        # c1 = self.conv_mlp1(c1) + c1

        # c4 = self.c4_down(c4)
        # c3 = self.c3_down(c3)
        # c2 = self.c2_down(c2)
        # c1 = self.c1_down(c1)
        # return c5, c4, c3, c2, c1
        return c5, c4, c3, c2, c1


class MSCA(nn.Module):
    def __init__(self, channel1, channel2=1):
        super(MSCA, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.M_sa1 = nn.Sequential(

            # nn.AvgPool2d(2, 2, 0),

            # nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.channel1, channel2, 3, 1, 1),
            # nn.Conv2d(1, 1, 7, 1, 3),
            nn.ReLU(True),
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            # nn.ConvTranspose2d(1, 1, 2, 2, 0),
            # nn.Conv2d(self.channel1 // 2, 1, 3, 1, 1),
            nn.Conv2d(channel2, channel2, 3, 1, 1),
            # nn.Upsample(scale_factor=2, mode='bilinear'),

            # nn.Softmax(dim=1)
            nn.Sigmoid(),

        )

        # self.M_ca1 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(self.channel1, self.channel1 // 2, 1, 1, 0),
        #     # nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.Conv2d(self.channel1 // 2, self.channel1, 1, 1, 0),
        #     # nn.BatchNorm2d(248),
        #     # nn.Softmax(dim=1),
        #     nn.Sigmoid()
        # )
        # # ConvBNReLU(inplanes, outplanes, k_d, s, p_d, dilation=d, groups=1)
        # # self.fuse_s = ConvBNReLU(self.channel1, self.channel1, 3, 1, 1, groups=self.channel1)
        # self.fuse = ConvBNReLU(self.channel1, self.channel1 * 2, 3, 1, 1, groups=1)

    def forward(self, x):
        # x0 = self.M_sa1(x) * x
        # x1 = self.fuse(torch.cat([self.M_sa1(x) * x, self.M_ca1(x) * x], dim=1))
        # x1 = (2 * self.M_sa1(x) - 1) * x + (2 * self.M_ca1(x) - 1) * x
        # x1 = (self.M_sa1(x)) * x

        # x1 = (self.M_sa1(x)) * x + (self.M_ca1(x)) * x
        x1 = (self.M_sa1(x))
        # x2, x3 = self.fuse(x1).chunk(2, dim=1)
        # x1 = (self.M_sa1(x)) + (self.M_ca1(x))

        # x1 = (self.M_ca1(x)) * x
        # x1 = (self.M_sa1(x1)) * x1 + x1
        # return (self.M_sa1(x)) * x, (self.M_ca1(x)) * x
        return x1


class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce


class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.fp = ConvBNReLU(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        self.fn = ConvBNReLU(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

    def forward(self, x, in_map):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction

        f_feature = x * in_map
        b_feature = x * (1 - in_map)

        fp = self.fp(f_feature)
        fn = self.fn(b_feature)

        refine1 = x - (self.alpha * fp)
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)

        return refine2


class Focus1(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus1, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        # self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
        #                         nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        #
        # self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        # self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)

        # self.fconv1 = MBConvBlock(self.channel1 * 3, self.channel2)
        # self.fconv2 = MBConvBlock(self.channel1 * 3, self.channel2)
        # self.fconv3 = MBConvBlock(self.channel1 * 3, self.channel2)

        # self.fconv1 = ConvBNReLU_BR(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        # self.fconv2 = ConvBNReLU_BR(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        # self.fconv3 = ConvBNReLU_BR(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        # self.fconv4 = ConvBNReLU_BR(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        self.fconv1 = ConvBNReLU(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        self.fconv2 = ConvBNReLU(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        self.fconv3 = ConvBNReLU(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        # self.fconv4 = ConvBNReLU(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        # self.fconv5 = ConvBNReLU(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        # self.fconv6 = ConvBNReLU(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)

        # self.input1 = nn.Sequential(nn.Conv2d(self.channel1, self.channel2 * 3, 3, 1, 1), nn.BatchNorm2d(self.channel2 * 3))
        # self.input1 = ConvBNReLU(self.channel1 * 1, self.channel2 * 3, 3, 1, 1)
        # self.conv2 = ConvBNReLU(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        # self.conv1 = ConvBNReLU_BR(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        # self.conv2 = ConvBNReLU_BR(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)

        # self.conv1 = GSA(self.channel1 * 1, self.channel2 * 1)
        # self.conv2 = GSA(self.channel1 * 1, self.channel2 * 1)
        # self.conv3 = ConvBNReLU(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)

        # self.conv11 = ConvBNReLU(self.channel1 * 2, self.channel2 * 1, 3, 1, 1)
        # self.conv22 = ConvBNReLU(self.channel1 * 2, self.channel2 * 1, 3, 1, 1)
        # self.conv33 = ConvBNReLU(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        # self.conv_e = ConvBNReLU(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        # self.conv_m = ConvBNReLU(self.channel1 * 1, self.channel2 * 1, 3, 1, 1)
        # self.conv00 = nn.Conv2d(self.channel1 * 1, 1, 1, 1, 0)
        # self.e1 = MMA(64, 64)

        self.a1 = Attention1(self.channel1)
        # self.a2 = Attention1(self.channel1)
        # self.a11 = Attention1(self.channel1)
        # self.a22 = Attention1(self.channel1)
        # self.out_layer_00 = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # self.out_layer_01 = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # self.out_layer_02 = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # self.a11 = Attention1(self.channel1)
        # self.a22 = Attention1(self.channel1)
        # self.g1 = GSA(self.channel1 * 1, self.channel2 * 1)
        # self.g2 = GSA(self.channel1 * 1, self.channel2 * 1)
        # self.m1 = MBSA('cuda:1', self.channel1 * 1, self.channel2 * 1)
        # self.m2 = MBSA('cuda:1', self.channel1 * 1, self.channel2 * 1)
        # self.a3 = Attention1(self.channel1)

        # self.d1 = dASPP(self.channel1, self.channel1)
        # self.d2 = dASPP(self.channel1, self.channel1)
        # self.d3 = dASPP(self.channel1, self.channel1)

        # self.conv4 = ConvBNReLU(self.channel1 * 2, self.channel2, 3, 1, 1)
        # self.conv11 = ConvBNReLU(self.channel1 * 2, self.channel2, 3, 1, 1)
        # self.conv22 = ConvBNReLU(self.channel1 * 2, self.channel2, 3, 1, 1)
        # self.conv33 = ConvBNReLU(self.channel1 * 2, self.channel2, 3, 1, 1)
        # self.conv4 = MBConvBlock(self.channel1 * 2, self.channel2)
        # self.conv5 = MBConvBlock(self.channel1 * 2, self.channel2)
        # self.conv11 = ConvBNReLU(self.channel2, self.channel2, 1, 1, 0)
        # self.conv22 = ConvBNReLU(self.channel2, self.channel2, 1, 1, 0)
        # self.conv33 = ConvBNReLU(self.channel2, self.channel2, 1, 1, 0)

        # self.conv11 = MBConvBlock(self.channel1 * 2, self.channel2)
        # self.conv22 = MBConvBlock(self.channel1 * 2, self.channel2)
        # self.conv33 = MBConvBlock(self.channel1 * 2, self.channel2)

        # self.msca1 = MSCA(self.channel2, self.channel2)
        # self.msca2 = MSCA(self.channel2, self.channel2)
        # self.msca3 = MSCA(self.channel2, self.channel2)
        # self.gsa1 = GSA(self.channel2, self.channel2)

    def forward(self, x, mask=None, edge=None, x_e=None, back=None):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction

        # e0 = x * x_e
        # e1, e2 = (self.fconv3(e0)).chunk(2, dim=1)
        # edge = self.conv_e(edge)
        # mask = self.conv_m(mask)
        # e4 = self.conv33(torch.cat([x, edge], dim=1))
        # e4 = self.conv33(x + edge)
        # e4 = x
        # e4 = refine0
        # e4 = edge
        # e3 = self.a3(e1, e2, e4)
        # refine1_0 = x + e3
        # refine1_0 = torch.cat([refine0, e3], dim=1)
        # refine1 = self.conv3(refine1_0)
        # refine1 = self.e1(x, x_e)
        refine1 = x
        # out0 = self.out_layer_00(e4)
        # refine1 = self.relu1(self.bn1(refine1_0))

        mask = mask
        # mask = mask
        # f0, b0, refine1 = self.input1(x).chunk(3, dim=1)
        # b1 = b0 * (1 - mask)
        # f1 = f0 * (mask)
        # b0 = (edge + mask) * (1 - x_m - x_e)
        b0 = refine1 * (1 - mask)
        f0 = refine1 * (mask)
        e0 = refine1 * (edge)
        # ne0 = refine1 * (1 - edge)
        e1 = self.fconv3(e0)
        b1 = self.fconv1(b0)
        f1 = self.fconv2(f0)
        # b1 = b0
        # f1 = f0
        # e0 = refine1 * (edge)
        # ne0 = refine1 * (1 - edge)
        # e1 = self.fconv3(e0)

        # ne1 = self.fconv4(ne0)
        # e1 = f0 * (edge)
        # ne1 = b0 * (1 - edge)

        # refine3 = self.conv1(self.a1(f1, b1, refine1, mask))
        refine3 = self.a1(f1, b1, e1, mask)
        # refine3 = self.conv1(self.a1(f1, b1, refine1, mask, e1, ne1))

        # e0 = refine3 * (edge)
        # ne0 = refine3 * (1 - edge)
        # # e1 = self.fconv3(e0)
        # # ne1 = self.fconv4(ne0)
        # e1 = e0
        # ne1 = ne0
        #
        # refine3 = self.conv2(self.a2(e1, ne1, refine3, mask))


        # b0 = back
        # b0 = torch.repeat_interleave((1 - x_m - x_e), 64, dim=1)
        # b0 = x * (1 - x_m - x_e)

        # b0_f = refine1 * (x_m + x_e)
        # edge_b = edge * (1 - x_m - x_e)
        # b0 = refine1 * (1 - x_m)
        # eb0 = refine1 * (x_e)
        # b0_e = self.fconv2r(torch.cat([b0, eb0], dim=1))
        # b0 = x * (1 - x_m)
        # b_a = self.fconv2(b0)
        # b_a = b0
        # b3 = self.msca2(b_a)
        # b1, b2 = (self.fconv2(b0)).chunk(2, dim=1)
        # b1, b2 = (self.fconv2(b0) * (1 - x_m - x_e)).chunk(2, dim=1)
        # b1 = self.fconv2(b0)
        # b2 = refine1
        # eb1, eb2 = self.fconv2r(eb0).chunk(2, dim=1)
        # b1, b2 = self.fconv2(torch.cat([edge_b, b0], dim=1)).chunk(2, dim=1)

        # b3 = self.a2(b0, b0, refine1)
        # b4 = self.conv22(torch.cat([refine1, edge], dim=1))
        # b4 = self.fconv2r(b0_f)
        # b4 = refine1
        # b5 = refine1
        # b4, b5 = self.fconv2r(refine1).chunk(2, dim=1)
        # b3 = self.a2(b1, b2, b4)
        # b3 = self.a2(b1, b2, b0, e0, x_e0)
        # out1 = self.out_layer_01(b3)
        # eb3 = self.a22(eb1, eb2, b4)
        # b3 = self.g2(b0) * b4
        # b3 = self.m2(b0) * b4
        # b3 = self.m2(b0, b4)
        # refine2_0 = b3
        # refine2_0 = b5 - b3
        # refine2_0 = torch.cat([refine1, b3], dim=1)
        # refine2 = self.conv2(torch.cat([refine2_0, edge], dim=1))
        # refine2 = self.conv2(refine2_0)
        # refine2 = self.conv2(b3)
        # refine2 = self.relu2(self.bn2(refine2_0))
        # refine2 = refine1

        # refine2_1, refine2_2 = x.chunk(2, dim=1)
        # refine2_1 = refine2_1 * (x_m + x_e)
        # f0 = torch.cat([refine2_1, refine2_2], dim=1)
        # f0 = x * (x_m + x_e)
        # f0 = (edge + mask) * (x_m + x_e)
        # f0 = refine2 * (x_m + x_e)
        # f0 = torch.repeat_interleave((x_m + x_e), 64, dim=1)
        # f0 = refine2
        # f0 = refine1
        # f0_b = refine2 * (1 - x_m - x_e)
        # edge_f = edge * (x_m + x_e)
        # f0 = refine2 * (x_m)
        # ef0 = refine2 * (x_e)
        # f0_e = self.fconv1r(torch.cat([f0, ef0], dim=1))
        # f0 = refine2 * (x_m)
        # f_a = self.fconv1(f0)
        # f_a = f0
        # f3 = self.msca1(f_a)

        # f0 = refine1
        # f1, f2 = (self.fconv4(f0 * (mask))).chunk(2, dim=1)
        # e1 = self.fconv5(f0 * (edge))
        # x_e1 = self.fconv6(x_e)
        # f1, f2 = (self.fconv1(f0)).chunk(2, dim=1)
        # f1, f2 = (self.fconv1(f0) * (x_m + x_e)).chunk(2, dim=1)
        # f1, f2 = self.fconv1(torch.cat([edge_f, f0], dim=1)).chunk(2, dim=1)
        # f1 = self.fconv1(f0)
        # f2 = refine2
        # ef1, ef2 = self.fconv1r(ef0).chunk(2, dim=1)
        # f3 = self.a1(f0, f0, x)
        # f4 = self.conv11(torch.cat([refine2, b4], dim=1))
        # f4 = self.fconv1r(f0_b)
        # f4 = refine2
        # f4 = back
        # f5 = refine2
        # f4, f5 = self.fconv1r(refine2).chunk(2, dim=1)
        # f3 = self.a1(f1, f2, f4)
        # f3 = self.a1(f1, f2, f0, e1, x_e1)
        # out2 = self.out_layer_02(f3)
        # ef3 = self.a11(ef1, ef2, f4)
        # f3 = self.g1(f0) * f4
        # f3 = self.m1(f0) * f4
        # f3 = self.m1(f0, f4)
        # refine3_0 = f3
        # refine3_0 = f5 + f3

        # refine3_0 = torch.cat([refine2, f3], dim=1)
        # refine3 = self.relu3(self.bn3(refine3_0))
        # refine3_0 = refine3_0 + self.gsa1(refine3_0)
        # refine3 = self.conv1(torch.cat([refine3_0, edge], dim=1))
        # refine3 = self.conv1(refine1 + b3 + f3)
        # refine3 = self.conv1(refine3_0)
        # refine3 = refine1

        return refine3
        # return out0 - out1 + out2


class Attention1(nn.Module):
    def __init__(self, indim=64, dim=64, num_heads=8, bias=True):
        super(Attention1, self).__init__()
        self.num_heads = num_heads
        self.num_heads2 = num_heads * 2
        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # self.qkv_0 = nn.Sequential(nn.Conv1d(dim * 2, dim * 2, 9, 1, 4), nn.BatchNorm1d(dim * 2))
        # self.qkv_1 = nn.Sequential(nn.Conv1d(dim, dim, 9, 1, 4), nn.BatchNorm1d(dim), nn.ReLU())
        # self.qkv_0 = nn.Sequential(nn.Conv1d(dim, dim, 9, 1, 4), nn.BatchNorm1d(dim), nn.ReLU())
        # self.qkv_1 = nn.Sequential(nn.Conv1d(dim, dim, 9, 1, 4), nn.BatchNorm1d(dim), nn.ReLU())
        # self.qkv_2 = nn.Sequential(nn.Conv1d(dim, dim, 9, 1, 4), nn.BatchNorm1d(dim), nn.ReLU())

        # self.qkv_3 = nn.Sequential(nn.Conv1d(dim, dim, 9, 1, 4), nn.BatchNorm1d(dim), nn.ReLU())
        # self.qkv_4 = nn.Sequential(nn.Conv1d(dim, dim, 9, 1, 4), nn.BatchNorm1d(dim), nn.ReLU())
        #
        # self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        # self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        # self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # self.qkv1conv = ConvBNReLU(num_heads, num_heads, 3, 1, 1)
        # self.qkv1conv = nn.Sequential(nn.Conv2d(num_heads, num_heads, 3, 1, 1))
        # self.qkv2conv = nn.Sequential(nn.Conv2d(num_heads, num_heads, 3, 1, 1))
        # self.qkv3conv = nn.Sequential(nn.Conv2d(num_heads, num_heads, 3, 1, 1))
        # self.qkv0conv = nn.Sequential(nn.Conv2d(num_heads, num_heads, 3, 1, 1), nn.BatchNorm2d(num_heads))
        # self.qkv1conv = nn.Sequential(nn.Conv2d(num_heads * 2, num_heads, 3, 1, 1), nn.BatchNorm2d(num_heads))
        # self.qkv2conv = nn.Sequential(nn.Conv2d(num_heads * 1, num_heads, 3, 1, 1), nn.BatchNorm2d(num_heads))
        self.qkv1conv = nn.Sequential(nn.Conv2d(num_heads * 1, num_heads * 1, 3, 1, 1), nn.BatchNorm2d(num_heads * 1))
        self.qkv2conv = nn.Sequential(nn.Conv2d(num_heads * 1, num_heads * 1, 3, 1, 1), nn.BatchNorm2d(num_heads * 1))
        self.qkv3conv = ConvBNReLU(dim * 1, dim, 3, 1, 1)
        self.qkv4conv = ConvBNReLU(dim * 1, dim, 3, 1, 1)
        self.qkv5conv = nn.Sequential(nn.Conv2d(num_heads * 1, num_heads * 1, 3, 1, 1), nn.BatchNorm2d(num_heads * 1))
        self.qkv6conv = nn.Sequential(nn.Conv2d(num_heads * 1, num_heads * 1, 3, 1, 1), nn.BatchNorm2d(num_heads * 1))
        self.qkv7conv = ConvBNReLU(dim * 1, dim, 3, 1, 1)
        self.qkv9conv = ConvBNReLU(dim * 1, dim, 3, 1, 1)
        # self.qkv5conv = nn.Sequential(nn.Conv2d(num_heads * 1, num_heads * 2, 3, 1, 1), nn.BatchNorm2d(num_heads * 2))
        self.qkv8conv = ConvBNReLU(dim * 2, dim, 3, 1, 1)

        # self.qkv1conv = GSA(num_heads, num_heads)
        # self.qkv1conv = nn.Sequential(ConvBNReLU(num_heads, num_heads * 4, 3, 1, 1), nn.Conv2d(num_heads * 4, num_heads, 3, 1, 1), nn.BatchNorm2d(num_heads))
        # self.qkv1conv = MBConvBlock(dim, dim)
        # self.qkv2conv = MBConvBlock(dim, dim)
        # self.qkv3conv = MBConvBlock(dim, dim)

        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.project_out1 = nn.Sequential(nn.Conv2d(dim * 1, dim, 3, 1, 1), nn.BatchNorm2d(dim))
        # self.project_out1 = ConvBNReLU(dim * 1, dim, 3, 1, 1)
        # self.project_out1 = dASPP(dim * 1, dim)
        # self.project_out1 = nn.Conv2d(dim * 1, dim, 3, 1, 1)
        # self.project_out2 = ConvBNReLU(dim * 1, dim, 3, 1, 1)
        # self.project_out3 = MBConvBlock(dim * 2, dim)

        # self.focus1 = Focus1(dim, dim)
        # self.msca1 = MMA(dim, dim)
        # self.msca1 = ConvBNReLU(dim * 2, dim, 3, 1, 1)
        # self.focus2 = Focus1(dim, dim)
        # self.focus3 = Focus1(dim, dim)
        # self.focus11 = Focus1(dim, dim)

        # self.softmin = nn.Softmin(dim=-1)
        # self.m1 = MBConvBlock(dim, dim)
        # self.focus2 = Focus2(dim, dim)

    def forward(self, x_0, x_1, x, mask=None, e0=None, e1=None):
        b, c, h, w = x.shape
        # q = x * mask
        # k = x * (1 - mask)
        # q = self.qkv1conv(self.qkv_0(x))
        # k = self.qkv2conv(self.qkv_1(x))
        # v = self.qkv3conv(self.qkv_2(x))

        # q, k, v = self.qkv1conv(x).chunk(3, dim=1)
        # q0, k0, v0 = self.qkv1conv(x).chunk(3, dim=1)
        # q, k, v = x, x, x

        q, k, e = x_0, x_1, x
        # qe, ke = e0, e1
        # v1 = v * mask
        # v2 = v * (1 - mask)
        # q0 = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k0 = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v0 = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        #
        # q0 = torch.nn.functional.normalize(q0, dim=-1)
        # k0 = torch.nn.functional.normalize(k0, dim=-1)
        #
        # attn0 = q0 @ k0.transpose(-2, -1)
        # attn0 = self.qkv0conv(attn0)
        # attn0 = attn0.softmax(dim=-1)
        # out0 = (attn0 @ v0)
        # out0 = rearrange(out0, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # v = out0

        # q, k, v, v0 = self.focus1(q, k, v, mask, edge)
        # v0 = v
        # q = self.qkv_0(rearrange(q, 'b head c h -> b (head c) h', head=self.num_heads))
        # k = self.qkv_1(rearrange(k, 'b head c h -> b (head c) h', head=self.num_heads))

        # q = rearrange(q, 'b c h w -> b c (h w)')
        # k = rearrange(k, 'b c h w -> b c (h w)')
        # v = rearrange(v, 'b c h w -> b c (h w)')
        # q, k = self.qkv_0(torch.cat([q, k], dim=1)).chunk(2, dim=1)
        # v = self.qkv_1(v)

        # q = self.qkv_0(rearrange(q, 'b c h w -> b c (h w)'))
        # k = self.qkv_1(rearrange(k, 'b c h w -> b c (h w)'))
        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        e = rearrange(e, 'b c h w -> b c (h w)')
        # v1 = rearrange(v1, 'b c h w -> b c (h w)')
        # v2 = rearrange(v2, 'b c h w -> b c (h w)')
        # v = self.qkv_2(rearrange(v, 'b c h w -> b c (h w)'))
        # qe = self.qkv_3(rearrange(qe, 'b c h w -> b c (h w)'))
        # ke = self.qkv_4(rearrange(ke, 'b c h w -> b c (h w)'))
        # qe = rearrange(qe, 'b c h w -> b c (h w)')
        # ke = rearrange(ke, 'b c h w -> b c (h w)')

        q = rearrange(q, 'b (head c) hw -> b head c hw', head=self.num_heads)
        k = rearrange(k, 'b (head c) hw -> b head c hw', head=self.num_heads)
        # qe = rearrange(qe, 'b (head c) hw -> b head c hw', head=self.num_heads)
        # ke = rearrange(ke, 'b (head c) hw -> b head c hw', head=self.num_heads)
        e = rearrange(e, 'b (head c) hw -> b head c hw', head=self.num_heads)
        # v1 = rearrange(v1, 'b (head c) hw -> b head c hw', head=self.num_heads)
        # v2 = rearrange(v2, 'b (head c) hw -> b head c hw', head=self.num_heads)


        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        qn = torch.nn.functional.normalize(q, dim=-1)
        kn = torch.nn.functional.normalize(k, dim=-1)
        en = torch.nn.functional.normalize(e, dim=-1)
        # qe = torch.nn.functional.normalize(qe, dim=-1)
        # ke = torch.nn.functional.normalize(ke, dim=-1)
        # attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn1 = qn @ kn.transpose(-2, -1)
        # attn2 = kn @ qn.transpose(-2, -1)
        attn3 = qn @ en.transpose(-2, -1)
        attn4 = kn @ en.transpose(-2, -1)
        # attn2 = qe @ ke.transpose(-2, -1)

        # attn1 = torch.nn.functional.normalize(q @ qe.transpose(-2, -1), dim=1)
        # attn2 = torch.nn.functional.normalize(k @ ke.transpose(-2, -1), dim=1)

        # attn1 = self.qkv1conv(torch.cat([attn1, attn2], dim=1))
        # attn3 = self.qkv1conv(attn1 @ attn2.transpose(-2, -1))
        # attn3 = self.qkv1conv(attn1 + attn2)
        # attn3 = self.qkv1conv(attn1 + attn2)
        # attn = self.qkv1conv(attn)

        # attn1 = self.qkv1conv(attn1)
        # attn2 = self.qkv2conv(attn1)
        # attn2 = self.qkv2conv(attn2)

        # attn0_1 = self.qkv1conv(attn1).softmax(dim=-1)
        # attn0_2 = self.qkv2conv(attn2).softmax(dim=-1)
        attn0_3 = self.qkv5conv(attn3).softmax(dim=-1)
        attn0_4 = self.qkv6conv(attn4).softmax(dim=-1)
        # attn0_1, attn0_2 = self.qkv1conv(attn1).softmax(dim=-1).chunk(2, dim=1)
        # attn0_3, attn0_4 = self.qkv2conv(attn1).softmax(dim=-1).chunk(2, dim=1)
        # attn0_1 = attn1.softmax(dim=-1)
        # attn0_2 = attn2.softmax(dim=-1)
        # attn0_3 = attn3.softmax(dim=-1)
        # v = self.project_out1(out1)
        # v1 = attn0_1 @ v + attn0_2 @ v
        # q1 = q + attn0_1 @ k
        # k1 = k + attn0_2 @ q
        q1 = q + attn0_3 @ e
        k1 = k - attn0_4 @ e
        q1 = rearrange(q1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        k1 = rearrange(k1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        q1 = self.qkv3conv(q1)
        k1 = self.qkv4conv(k1)
        # e1 = e + attn0_3 @ k + attn0_4 @ q

        q1 = rearrange(q1, 'b c h w -> b c (h w)')
        k1 = rearrange(k1, 'b c h w -> b c (h w)')
        q1 = rearrange(q1, 'b (head c) hw -> b head c hw', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) hw -> b head c hw', head=self.num_heads)
        qn = torch.nn.functional.normalize(q1, dim=-1)
        kn = torch.nn.functional.normalize(k1, dim=-1)
        attn1 = qn @ kn.transpose(-2, -1)
        attn2 = kn @ qn.transpose(-2, -1)
        attn0_1 = self.qkv1conv(attn1).softmax(dim=-1)
        attn0_2 = self.qkv2conv(attn2).softmax(dim=-1)
        q2 = q1 + attn0_1 @ k1
        k2 = k1 - attn0_2 @ q1
        # v1 = attn0_1 @ q
        # v2 = attn0_2 @ k
        # v1 = attn0_1 @ v1 + attn0_2 @ v1
        # v2 = attn0_1 @ v2 + attn0_2 @ v2
        # ve0 = attn0_2 @ ke

        # ve1 = ve0 * mask
        # ve2 = ve0 * (1 - mask)
        # out1 = (v + v1 - v2 + ve0)
        # out1 = (v + v1)
        # out1 = (v + v1 + v2)
        # out1 = (attn0_3 @ v)
        # out1 = (attn0_1 @ v + attn0_2 @ v + attn0_3 @ v + v)
        # v = self.qkv_1(v)
        # v1 = rearrange(v1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # v2 = rearrange(v2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # q1 = rearrange(q1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        q2 = rearrange(q2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # q3 = rearrange(q3, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # k1 = rearrange(k1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # e1 = rearrange(e1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        k2 = rearrange(k2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # k3 = rearrange(k3, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # q1 = self.qkv3conv(q1)
        # k1 = self.qkv4conv(k1)
        # e1 = self.qkv7conv(e1)


        q2 = self.qkv7conv(q2)
        k2 = self.qkv9conv(k2)
        # e1 = self.qkv7conv(e1)

        # out1 = q1 + k1 + e1
        # out1 = self.msca1(v1, v2)
        # out1 = self.qkv8conv(torch.cat([q1, k1, e1], dim=1))
        out1 = self.qkv8conv(torch.cat([q2, k2], dim=1))
        # out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # v1 = rearrange(v, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # out1 = self.project_out1(torch.cat([out0, out1], dim=1))
        # out1 = self.project_out1(out1)
        # out1 = self.project_out1(out1 + v0)
        # out1 = out1

        # out = out1
        # out = v0

        # return attn
        return out1




class MBSA(nn.Module):
    def __init__(self, device, channel1, channel2, p=1, d=1, h=64):
        super(MBSA, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2


        # self.upsample = nn.Upsample(scale_factor=8, mode='bilinear')
        # self.downsample = nn.AvgPool2d(8, 8, 0)
        # self.down0 = ConvBNReLU(self.channel1, self.channel1 // 8, 1, 1, 0)
        # self.up0 = nn.Sequential(nn.Conv2d(self.channel1 // 8, self.channel1, 1, 1, 0), nn.BatchNorm2d(self.channel1))
        # self.up2 = ConvBNReLU(self.channel1, self.channel2, 3, 1, 1)
        # self.up2 = nn.Sequential(nn.Conv2d(self.channel1, self.channel1, 3, 2, 1), nn.BatchNorm2d(self.channel1))
        self.mamba2_block = mamba2.Mamba2(
            mamba2.Mamba2Config(d_model=channel1 * 1, d_state=128, expand=2, d_conv=4, headdim=h), device=device)
        # self.mamba2_block2 = mamba2.Mamba2(
        #     mamba2.Mamba2Config(d_model=channel1 // 8, d_state=128, expand=2, d_conv=4, headdim=4), device=device)
        # self.up2 = ConvBNReLU(self.channel1 * 8, self.channel1 * 8, 3, 1, 1, groups=1)
        # self.up2 = ConvBNReLU(self.channel1 // 2, self.channel1 // 2, 3, 1, 1, groups=self.channel1 // 2)
        # self.down1 = ConvBNReLU(self.channel1 // 4, self.channel2, 3, 1, 1)
        # self.down1 = nn.Sequential(nn.Conv2d(self.channel1 // 1, self.channel1, 3, 1, 1), nn.BatchNorm2d(self.channel1), nn.Sigmoid())
        # self.down0 = nn.Sequential(nn.Conv2d(self.channel1, self.channel2, 3, 1, 1), nn.BatchNorm2d(self.channel2))
        self.down1 = ConvBNReLU(self.channel1, self.channel2, 3, 1, p, dilation=d)
        # self.sig = nn.Sigmoid()
        # self.soft = nn.Softmax(dim=-1)
        # # ConvBNReLU(inplanes, outplanes, k_d, s, p_d, dilation=d, groups=1)
        # # self.fuse_s = ConvBNReLU(self.channel1, self.channel1, 3, 1, 1, groups=self.channel1)
        # self.fuse = ConvBNReLU(self.channel1, self.channel1 * 2, 3, 1, 1, groups=1)

    def forward(self, x, z=None, z2=None):
        b, c, h, w = x.shape
        # x0 = self.M_sa1(x) * x
        # x1 = self.fuse(torch.cat([self.M_sa1(x) * x, self.M_ca1(x) * x], dim=1))
        # x1 = (2 * self.M_sa1(x) - 1) * x + (2 * self.M_ca1(x) - 1) * x
        # x1 = (self.M_sa1(x)) * x

        # x1 = (self.M_sa1(x)) * x + (self.M_ca1(x)) * x
        # x1 = (self.M_sa1(x))
        # if z is None:
        #     z = x

        h1 = 1
        w1 = 1
        h0 = h // h1
        w0 = w // w1

        # x0 = rearrange(x, 'b c (h h1) (w w1) -> b (c h1 w1) h w', h1=h1, w1=w1)
        # print(x.shape)
        # x0 = self.downsample(x)
        # print(x0.shape)
        # z0 = self.downsample(z)
        # x0 = self.up1(x0)
        # z0 = self.up2(z0)
        x0 = x
        # x2 = self.down0(x)
        # z0 = z
        # z1 = rearrange(z0, 'b c h w -> b (h w) c', h=h, w=w)
        x1 = rearrange(x0, 'b c h w -> b (h w) c', h=h0, w=w0)
        # z = rearrange(z, 'b c h w -> b (h w) c', h=h0, w=w0)
        # z2 = rearrange(z2, 'b c h w -> b (h w) c', h=h0, w=w0)
        # x2 = rearrange(x2, 'b c h w -> b (h w) c', h=h, w=w)
        x1, mc = self.mamba2_block(x1)
        # x2, mc = self.mamba2_block2(x2)
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=h0, w=w0)
        # x2 = rearrange(x2, 'b (h w) c -> b c h w', h=h, w=w)
        # x1 = rearrange(x0, 'b c (h h1) (w w1) -> b (h w) (c h1 w1)', h1=h1, w1=w1, h=h, w=w)
        # x1, mc = self.mamba2_block(x1)
        # x1 = rearrange(x1, 'b (h w) (c h1 w1) -> b c (h h1) (w w1)', h1=h1, w1=w1, h=h, w=w)
        # x1 = self.upsample(x1)
        x1 = self.down1(x1)
        # x1 = self.soft(self.down1(x1))
        # x2, x3 = self.fuse(x1).chunk(2, dim=1)
        # x1 = (self.M_sa1(x)) + (self.M_ca1(x))

        # x1 = (self.M_ca1(x)) * x
        # x1 = (self.M_sa1(x1)) * x1 + x1
        # return (self.M_sa1(x)) * x, (self.M_ca1(x)) * x
        return x1


class GSA(nn.Module):
    def __init__(self, channel1, channel2):
        super(GSA, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        # self.M_sa1 = nn.Sequential(
        #
        #     # nn.AvgPool2d(2, 2, 0),
        #
        #
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(self.channel1, channel2, 3, 1, 1),
        #     # nn.Conv2d(1, 1, 7, 1, 3),
        #     nn.ReLU(True),
        #     # nn.Upsample(scale_factor=2, mode='bilinear'),
        #     # nn.ConvTranspose2d(1, 1, 2, 2, 0),
        #     # nn.Conv2d(self.channel1 // 2, 1, 3, 1, 1),
        #     nn.Conv2d(channel2, channel2, 3, 2, 1),
        #     # nn.Upsample(scale_factor=2, mode='bilinear'),
        #
        #     # nn.Softmax(dim=1)
        #     nn.Sigmoid(),
        #
        # )

        # self.M_ca1 = nn.Sequential(
        #     # nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(self.channel1, self.channel1 // 2, 3, 1, 1),
        #     # nn.BatchNorm2d(32),
        #     # nn.ReLU(True),
        #     nn.Conv2d(self.channel1 // 2, self.channel1, 1, 1, 0),
        #     # nn.BatchNorm2d(248),
        #     # nn.Softmax(dim=1),
        #     nn.Sigmoid()
        # )
        self.up1 = ConvBNReLU(self.channel1, self.channel1 // 8, 3, 1, 1)
        self.up2 = ConvBNReLU(self.channel1 * 8, self.channel1 * 8, 3, 1, 1, groups=1)
        # self.up2 = ConvBNReLU(self.channel1 // 2, self.channel1 // 2, 3, 1, 1, groups=self.channel1 // 2)
        self.down1 = ConvBNReLU(self.channel1 // 8, self.channel2, 3, 1, 1)
        # self.down1 = nn.Sequential(nn.Conv2d(self.channel1 // 8, self.channel1, 3, 1, 1), nn.BatchNorm2d(self.channel1))
        # self.sig = nn.Sigmoid()
        # self.soft = nn.Softmax(dim=-1)
        # # ConvBNReLU(inplanes, outplanes, k_d, s, p_d, dilation=d, groups=1)
        # # self.fuse_s = ConvBNReLU(self.channel1, self.channel1, 3, 1, 1, groups=self.channel1)
        # self.fuse = ConvBNReLU(self.channel1, self.channel1 * 2, 3, 1, 1, groups=1)

    def forward(self, x):
        # b, c, h, w = x.shape
        # x0 = self.M_sa1(x) * x
        # x1 = self.fuse(torch.cat([self.M_sa1(x) * x, self.M_ca1(x) * x], dim=1))
        # x1 = (2 * self.M_sa1(x) - 1) * x + (2 * self.M_ca1(x) - 1) * x
        # x1 = (self.M_sa1(x)) * x

        # x1 = (self.M_sa1(x)) * x + (self.M_ca1(x)) * x
        # x1 = (self.M_sa1(x))
        h1 = 8
        w1 = 8

        # x0 = rearrange(x, 'b c (h h1) (w w1) -> b (c h1 w1) h w', h1=h1, w1=w1)
        x0 = self.up1(x)
        x1 = rearrange(x0, 'b c (h h1) (w w1) -> b (c h1 w1) h w', h1=h1, w1=w1)
        x1 = self.up2(x1)
        x1 = rearrange(x1, 'b (c h1 w1) h w -> b c (h h1) (w w1)', h1=h1, w1=w1) + x0
        x1 = self.down1(x1)
        # x1 = self.soft(self.down1(x1))
        # x2, x3 = self.fuse(x1).chunk(2, dim=1)
        # x1 = (self.M_sa1(x)) + (self.M_ca1(x))

        # x1 = (self.M_ca1(x)) * x
        # x1 = (self.M_sa1(x1)) * x1 + x1
        # return (self.M_sa1(x)) * x, (self.M_ca1(x)) * x
        return x1



class MMA(nn.Module):
    def __init__(self, channel1, channel2):
        super(MMA, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        # self.M_sa1 = nn.Sequential(
        #
        #     # nn.AvgPool2d(2, 2, 0),
        #
        #
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(self.channel1, channel2, 3, 1, 1),
        #     # nn.Conv2d(1, 1, 7, 1, 3),
        #     nn.ReLU(True),
        #     # nn.Upsample(scale_factor=2, mode='bilinear'),
        #     # nn.ConvTranspose2d(1, 1, 2, 2, 0),
        #     # nn.Conv2d(self.channel1 // 2, 1, 3, 1, 1),
        #     nn.Conv2d(channel2, channel2, 3, 2, 1),
        #     # nn.Upsample(scale_factor=2, mode='bilinear'),
        #
        #     # nn.Softmax(dim=1)
        #     nn.Sigmoid(),
        #
        # )

        self.M_ca1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.channel1, self.channel1 * 2, 1, 1, 0),
            nn.BatchNorm2d(self.channel1 * 2),
            nn.ReLU(True),
            nn.Conv2d(self.channel1 * 2, self.channel1, 1, 1, 0),
            nn.BatchNorm2d(self.channel1),
            # nn.Softmax(dim=1),
            nn.Sigmoid()
        )
        self.M_ca2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.channel1, self.channel1 * 2, 1, 1, 0),
            nn.BatchNorm2d(self.channel1 * 2),
            nn.ReLU(True),
            nn.Conv2d(self.channel1 * 2, self.channel1, 1, 1, 0),
            nn.BatchNorm2d(self.channel1),
            # nn.Softmax(dim=1),
            nn.Sigmoid()
        )

        self.M_ca3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.channel1 * 1, self.channel1 * 2, 1, 1, 0),
            nn.BatchNorm2d(self.channel1 * 2),
            nn.ReLU(True),
            nn.Conv2d(self.channel1 * 2, self.channel1, 1, 1, 0),
            nn.BatchNorm2d(self.channel1),
            # nn.Softmax(dim=1),
            nn.Sigmoid()
        )
        # self.M_ca4 = nn.Sequential(
        #     # nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(self.channel1 * 1, self.channel1 // 2, 3, 1, 1),
        #     nn.BatchNorm2d(self.channel1 // 2),
        #     nn.ReLU(True),
        #     nn.Conv2d(self.channel1 // 2, self.channel1 * 1, 3, 1, 1),
        #     nn.BatchNorm2d(self.channel1),
        #     # nn.Softmax(dim=1),
        #     nn.Sigmoid()
        # )
        self.out1 = ConvBNReLU(self.channel1 * 2, self.channel1, 3, 1, 1)
        self.out2 = ConvBNReLU(self.channel1 * 2, self.channel1, 3, 1, 1)
        self.out3 = ConvBNReLU(self.channel1 * 2, self.channel1, 3, 1, 1)
        self.out4 = ConvBNReLU(self.channel1 * 2, self.channel1, 3, 1, 1)

        # self.up2 = ConvBNReLU(self.channel1 * 8, self.channel1 * 8, 3, 1, 1, groups=1)
        # self.up2 = ConvBNReLU(self.channel1 // 2, self.channel1 // 2, 3, 1, 1, groups=self.channel1 // 2)
        # self.down1 = ConvBNReLU(self.channel1 // 8, self.channel2, 3, 1, 1)
        # self.down1 = nn.Sequential(nn.Conv2d(self.channel1 // 8, self.channel1, 3, 1, 1), nn.BatchNorm2d(self.channel1))
        # self.sig = nn.Sigmoid()
        # self.soft = nn.Softmax(dim=-1)
        # # ConvBNReLU(inplanes, outplanes, k_d, s, p_d, dilation=d, groups=1)
        # # self.fuse_s = ConvBNReLU(self.channel1, self.channel1, 3, 1, 1, groups=self.channel1)
        # self.fuse = ConvBNReLU(self.channel1, self.channel1 * 2, 3, 1, 1, groups=1)

    def forward(self, x, x1):
        # b, c, h, w = x.shape

        x2_1 = x * x1
        x2_2 = x + x1
        x2_3 = self.out1(torch.cat((x2_1, x2_2), dim=1))
        a1 = self.M_ca1(x2_1)
        a2 = self.M_ca2(x2_2)
        x3 = self.out2(torch.cat((a1 * x, a2 * x1), dim=1))
        x31 = self.out3(torch.cat((a2 * x, a1 * x1), dim=1))
        # a3 = self.M_ca3(x3
        # a4 = self.M_ca4(x31)
        # x40 = x3 + x31
        x40 = self.out4(torch.cat((x3, x31), dim=1))
        a4 = self.M_ca3(x40)
        x4 = a4 * x2_3 + x40
        # x1 = self.soft(self.down1(x1))
        # x2, x3 = self.fuse(x1).chunk(2, dim=1)
        # x1 = (self.M_sa1(x)) + (self.M_ca1(x))

        # x1 = (self.M_ca1(x)) * x
        # x1 = (self.M_sa1(x1)) * x1 + x1
        # return (self.M_sa1(x)) * x, (self.M_ca1(x)) * x
        return x4


class MMA2(nn.Module):
    def __init__(self, channel1, channel2):
        super(MMA2, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        # self.M_sa1 = nn.Sequential(
        #
        #     # nn.AvgPool2d(2, 2, 0),
        #
        #
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(self.channel1, channel2, 3, 1, 1),
        #     # nn.Conv2d(1, 1, 7, 1, 3),
        #     nn.ReLU(True),
        #     # nn.Upsample(scale_factor=2, mode='bilinear'),
        #     # nn.ConvTranspose2d(1, 1, 2, 2, 0),
        #     # nn.Conv2d(self.channel1 // 2, 1, 3, 1, 1),
        #     nn.Conv2d(channel2, channel2, 3, 2, 1),
        #     # nn.Upsample(scale_factor=2, mode='bilinear'),
        #
        #     # nn.Softmax(dim=1)
        #     nn.Sigmoid(),
        #
        # )
        self.M_sa1 = nn.Sequential(

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.channel1, 1, 3, 1, 1),
            # nn.Conv2d(1, 1, 7, 1, 3),
            nn.ReLU(True),
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            # nn.ConvTranspose2d(1, 1, 2, 2, 0),
            # nn.Conv2d(self.channel1 // 2, 1, 3, 1, 1),
            nn.Conv2d(1, 1, 3, 2, 1),
            # nn.Upsample(scale_factor=2, mode='bilinear'),

            # nn.Softmax(dim=1)
            nn.Sigmoid(),

        )
        self.M_sa2 = nn.Sequential(

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.channel1, 1, 3, 1, 1),
            # nn.Conv2d(1, 1, 7, 1, 3),
            nn.ReLU(True),
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            # nn.ConvTranspose2d(1, 1, 2, 2, 0),
            # nn.Conv2d(self.channel1 // 2, 1, 3, 1, 1),
            nn.Conv2d(1, 1, 3, 2, 1),
            # nn.Upsample(scale_factor=2, mode='bilinear'),

            # nn.Softmax(dim=1)
            nn.Sigmoid(),

        )
        self.M_sa3 = nn.Sequential(

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.channel1, 1, 3, 1, 1),
            # nn.Conv2d(1, 1, 7, 1, 3),
            nn.ReLU(True),
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            # nn.ConvTranspose2d(1, 1, 2, 2, 0),
            # nn.Conv2d(self.channel1 // 2, 1, 3, 1, 1),
            nn.Conv2d(1, 1, 3, 2, 1),
            # nn.Upsample(scale_factor=2, mode='bilinear'),

            # nn.Softmax(dim=1)
            nn.Sigmoid(),

        )

        # self.M_ca4 = nn.Sequential(
        #     # nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(self.channel1 * 1, self.channel1 // 2, 3, 1, 1),
        #     nn.BatchNorm2d(self.channel1 // 2),
        #     nn.ReLU(True),
        #     nn.Conv2d(self.channel1 // 2, self.channel1 * 1, 3, 1, 1),
        #     nn.BatchNorm2d(self.channel1),
        #     # nn.Softmax(dim=1),
        #     nn.Sigmoid()
        # )
        self.out1 = ConvBNReLU(self.channel1 * 2, self.channel1, 3, 1, 1)
        self.out2 = ConvBNReLU(self.channel1 * 2, self.channel1, 3, 1, 1)
        self.out3 = ConvBNReLU(self.channel1 * 2, self.channel1, 3, 1, 1)
        self.out4 = ConvBNReLU(self.channel1 * 2, self.channel1, 3, 1, 1)

        # self.up2 = ConvBNReLU(self.channel1 * 8, self.channel1 * 8, 3, 1, 1, groups=1)
        # self.up2 = ConvBNReLU(self.channel1 // 2, self.channel1 // 2, 3, 1, 1, groups=self.channel1 // 2)
        # self.down1 = ConvBNReLU(self.channel1 // 8, self.channel2, 3, 1, 1)
        # self.down1 = nn.Sequential(nn.Conv2d(self.channel1 // 8, self.channel1, 3, 1, 1), nn.BatchNorm2d(self.channel1))
        # self.sig = nn.Sigmoid()
        # self.soft = nn.Softmax(dim=-1)
        # # ConvBNReLU(inplanes, outplanes, k_d, s, p_d, dilation=d, groups=1)
        # # self.fuse_s = ConvBNReLU(self.channel1, self.channel1, 3, 1, 1, groups=self.channel1)
        # self.fuse = ConvBNReLU(self.channel1, self.channel1 * 2, 3, 1, 1, groups=1)

    def forward(self, x, x1):
        # b, c, h, w = x.shape

        x2_1 = x * x1
        x2_2 = x + x1
        x2_3 = self.out1(torch.cat((x2_1, x2_2), dim=1))
        a1 = self.M_sa1(x2_1)
        a2 = self.M_sa2(x2_2)
        x3 = self.out2(torch.cat((a1 * x, a2 * x1), dim=1))
        x31 = self.out3(torch.cat((a2 * x, a1 * x1), dim=1))
        # a3 = self.M_ca3(x3
        # a4 = self.M_ca4(x31)
        # x40 = x3 + x31
        x40 = self.out4(torch.cat((x3, x31), dim=1))
        a4 = self.M_sa3(x40)
        x4 = a4 * x2_3 + x40
        # x1 = self.soft(self.down1(x1))
        # x2, x3 = self.fuse(x1).chunk(2, dim=1)
        # x1 = (self.M_sa1(x)) + (self.M_ca1(x))

        # x1 = (self.M_ca1(x)) * x
        # x1 = (self.M_sa1(x1)) * x1 + x1
        # return (self.M_sa1(x)) * x, (self.M_ca1(x)) * x
        return x4

# class NeighborConnectionDecoder(nn.Module):
#     def __init__(self, channel):
#         super(NeighborConnectionDecoder, self).__init__()
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
#         self.downsample = nn.AvgPool2d(2, 2, 0)
#         # self.conv_upsample1 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample2 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample3 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample4 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample5 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample6 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample7 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample8 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample9 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample10 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample11 = ConvBNReLU(channel, channel, 3, 1, 1)
#
#         self.conv_upsample0 = MBConvBlock(channel, channel)
#         self.conv_upsample00 = MBConvBlock(channel, channel)
#         self.conv_upsample000 = MBConvBlock(channel, channel)
#         self.conv_upsample1 = MBConvBlock(channel, channel)
#         self.conv_upsample2 = MBConvBlock(channel, channel)
#         self.conv_upsample3 = MBConvBlock(channel, channel)
#         self.conv_upsample4 = MBConvBlock(channel, channel)
#         self.conv_upsample5 = MBConvBlock(channel, channel)
#         self.conv_upsample6 = MBConvBlock(channel, channel)
#         self.conv_upsample7 = MBConvBlock(channel, channel)
#         self.conv_upsample8 = MBConvBlock(channel, channel)
#         self.conv_upsample9 = MBConvBlock(channel, channel)
#         self.conv_upsample10 = MBConvBlock(channel, channel)
#         self.conv_upsample11 = MBConvBlock(channel, channel)
#         # self.conv_upsample12 = ConvBNReLU(channel, channel, 3, 1, 1)
#
#         # self.conv_upsample5 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1)
#
#         # self.conv_concat2 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
#         # self.conv_concat3 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
#         # self.conv_concat4 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
#         # self.conv_concat5 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
#
#         self.conv_concat2 = MBConvBlock(2 * channel, channel)
#         self.conv_concat3 = MBConvBlock(2 * channel, channel)
#         self.conv_concat4 = MBConvBlock(2 * channel, channel)
#         self.conv_concat5 = MBConvBlock(2 * channel, channel)
#
#         # self.add4 = M_add(64)
#         # self.add3 = M_add(64)
#         # self.add2 = M_add(64)
#         # self.add1 = M_add(64)
#         # self.add0 = M_add(64)
#
#         # self.a5 = Attention2()
#         # self.a4 = Attention2()
#         # self.a3 = Attention2()
#         # self.a2 = Attention2()
#         # self.a1 = Attention2()
#
#         # self.add44 = M_add(64)
#         # self.add33 = M_add(64)
#         # self.add22 = M_add(64)
#         # self.add11 = M_add(64)
#         # self.add00 = M_add(64)
#
#         # self.conv4 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
#         # self.conv5 = nn.Conv2d(3 * channel, 1, 1)
#
#     def forward(self, zt5, zt4, zt3, zt2, zt1):
#         # zt5_1 = self.conv_upsample0(zt5)
#         # zt4_1 = self.add11(self.add1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5))), zt4)
#         # zt3_1 = self.add22(self.add2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4))), zt3)
#         # zt2_1 = self.add33(self.add3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3))), zt2)
#         # zt1_1 = self.add44(self.add4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2))), zt1)
#         # zt5_1 = self.conv_upsample0(zt5)
#         # zt4_1 = self.a1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5)), zt4)
#         # zt3_1 = self.a2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4)), zt3)
#         # zt2_1 = self.a3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3)), zt2)
#         # zt1_1 = self.a4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2)), zt1)
#         # zt5_1 = self.add0(self.conv_upsample0(zt5), e5_1)
#         # zt4_1 = self.add1(self.conv_upsample00(self.upsample(zt5_1)) * self.conv_upsample1(self.upsample(zt5)) + zt4, e4_1)
#         # zt3_1 = self.add2(self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) + zt3, e3_1)
#         # zt2_1 = self.add3(self.conv_upsample4(self.upsample(zt3_1)) * self.conv_upsample5(self.upsample(zt3)) + zt2, e2_1)
#         # zt1_1 = self.add4(self.conv_upsample6(self.upsample(zt2_1)) * self.conv_upsample7(self.upsample(zt2)) + zt1, e1_1)
#
#         zt5_1 = self.conv_upsample0(zt5)
#         zt4_1 = self.conv_upsample00(self.upsample(zt5_1)) * self.conv_upsample1(self.upsample(zt5)) + zt4
#         zt3_1 = self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) + zt3
#         zt2_1 = self.conv_upsample4(self.upsample(zt3_1)) * self.conv_upsample5(self.upsample(zt3)) + zt2
#         zt1_1 = self.conv_upsample6(self.upsample(zt2_1)) * self.conv_upsample7(self.upsample(zt2)) + zt1
#
#         zt4_2 = torch.cat((zt4_1, self.conv_upsample8(self.upsample(zt5_1))),  1)
#         zt4_2 = self.conv_concat2(zt4_2)
#
#         # zt4_2 = self.add4(zt4_1, self.conv_upsample8(self.upsample(zt5_1)))
#
#         zt3_2 = torch.cat((zt3_1, self.conv_upsample9(self.upsample(zt4_2))), 1)
#         zt3_2 = self.conv_concat3(zt3_2)
#
#         # zt3_2 = self.add3(zt3_1, self.conv_upsample9(self.upsample(zt4_2)))
#
#         zt2_2 = torch.cat((zt2_1, self.conv_upsample10(self.upsample(zt3_2))), 1)
#         zt2_2 = self.conv_concat4(zt2_2)
#
#         # zt2_2 = self.add2(zt2_1, self.conv_upsample10(self.upsample(zt3_2)))
#
#         zt1_2 = torch.cat((zt1_1, self.conv_upsample11(self.upsample(zt2_2))), 1)
#         zt1_2 = self.conv_concat5(zt1_2)
#
#         # zt1_2 = self.add1(zt1_1, self.conv_upsample11(self.upsample(zt2_2)))
#
#         # pc = self.conv4(zt1_2)
#         # pc = self.conv5(pc)
#         # pc = zt1_2
#
#         return [zt5_1, zt4_1, zt3_1, zt2_1, zt1_1, zt4_2, zt3_2, zt2_2, zt1_2]
#
# class NeighborConnectionDecoder3(nn.Module):
#     def __init__(self, channel):
#         super(NeighborConnectionDecoder3, self).__init__()
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
#         self.downsample = nn.AvgPool2d(2, 2, 0)
#         # self.conv_upsample1 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample2 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample3 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample4 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample5 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample6 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample7 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample8 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample9 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample10 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample11 = ConvBNReLU(channel, channel, 3, 1, 1)
#
#         self.conv_upsample0 = MBConvBlock(channel, channel)
#         self.conv_upsample00 = MBConvBlock(channel, channel)
#         # self.conv_upsample000 = MBConvBlock(channel, channel)
#         self.conv_upsample1 = MBConvBlock(channel, channel)
#         self.conv_upsample2 = MBConvBlock(channel, channel)
#         self.conv_upsample3 = MBConvBlock(channel, channel)
#         self.conv_upsample4 = MBConvBlock(channel, channel)
#         self.conv_upsample5 = MBConvBlock(channel, channel)
#         self.conv_upsample6 = MBConvBlock(channel, channel)
#         self.conv_upsample7 = MBConvBlock(channel, channel)
#         self.conv_upsample8 = MBConvBlock(channel, channel)
#         self.conv_upsample9 = MBConvBlock(channel, channel)
#         self.conv_upsample10 = MBConvBlock(channel, channel)
#         self.conv_upsample11 = MBConvBlock(channel, channel)
#         # self.conv_upsample12 = ConvBNReLU(channel, channel, 3, 1, 1)
#
#         # self.conv_upsample5 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1)
#
#         # self.conv_concat2 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
#         # self.conv_concat3 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
#         # self.conv_concat4 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
#         # self.conv_concat5 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
#
#         self.conv_concat2 = MBConvBlock(2 * channel, channel)
#         self.conv_concat3 = MBConvBlock(2 * channel, channel)
#         self.conv_concat4 = MBConvBlock(2 * channel, channel)
#         self.conv_concat5 = MBConvBlock(2 * channel, channel)
#
#         # self.add4 = M_add(64)
#         # self.add3 = M_add(64)
#         # self.add2 = M_add(64)
#         # self.add1 = M_add(64)
#         # self.add0 = M_add(64)
#
#         # self.a5 = Attention2()
#         # self.a4 = Attention2()
#         # self.a3 = Attention2()
#         # self.a2 = Attention2()
#         # self.a1 = Attention2()
#
#         # self.add44 = M_add(64)
#         # self.add33 = M_add(64)
#         # self.add22 = M_add(64)
#         # self.add11 = M_add(64)
#         # self.add00 = M_add(64)
#
#         # self.conv4 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
#         # self.conv5 = nn.Conv2d(3 * channel, 1, 1)
#
#     def forward(self, zt5, zt4, zt3, zt2, zt1):
#         # zt5_1 = self.conv_upsample0(zt5)
#         # zt4_1 = self.add11(self.add1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5))), zt4)
#         # zt3_1 = self.add22(self.add2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4))), zt3)
#         # zt2_1 = self.add33(self.add3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3))), zt2)
#         # zt1_1 = self.add44(self.add4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2))), zt1)
#         # zt5_1 = self.conv_upsample0(zt5)
#         # zt4_1 = self.a1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5)), zt4)
#         # zt3_1 = self.a2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4)), zt3)
#         # zt2_1 = self.a3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3)), zt2)
#         # zt1_1 = self.a4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2)), zt1)
#         # zt5_1 = self.add0(self.conv_upsample0(zt5), e5_1)
#         # zt4_1 = self.add1(self.conv_upsample00(self.upsample(zt5_1)) * self.conv_upsample1(self.upsample(zt5)) + zt4, e4_1)
#         # zt3_1 = self.add2(self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) + zt3, e3_1)
#         # zt2_1 = self.add3(self.conv_upsample4(self.upsample(zt3_1)) * self.conv_upsample5(self.upsample(zt3)) + zt2, e2_1)
#         # zt1_1 = self.add4(self.conv_upsample6(self.upsample(zt2_1)) * self.conv_upsample7(self.upsample(zt2)) + zt1, e1_1)
#
#         zt5_1 = self.conv_upsample0(zt5)
#         zt4_1 = self.conv_upsample00(self.upsample(zt5_1)) * self.conv_upsample1(self.upsample(zt5)) + zt4
#         zt3_1 = self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) + zt3
#         zt2_1 = self.conv_upsample4(self.upsample(zt3_1)) * self.conv_upsample5(self.upsample(zt3)) + zt2
#         zt1_1 = self.conv_upsample6(self.upsample(zt2_1)) * self.conv_upsample7(self.upsample(zt2)) + zt1
#
#         zt4_2 = torch.cat((zt4_1, self.conv_upsample8(self.upsample(zt5_1))),  1)
#         zt4_2 = self.conv_concat2(zt4_2)
#
#         # zt4_2 = self.add4(zt4_1, self.conv_upsample8(self.upsample(zt5_1)))
#
#         zt3_2 = torch.cat((zt3_1, self.conv_upsample9(self.upsample(zt4_2))), 1)
#         zt3_2 = self.conv_concat3(zt3_2)
#
#         # zt3_2 = self.add3(zt3_1, self.conv_upsample9(self.upsample(zt4_2)))
#
#         zt2_2 = torch.cat((zt2_1, self.conv_upsample10(self.upsample(zt3_2))), 1)
#         zt2_2 = self.conv_concat4(zt2_2)
#
#         # zt2_2 = self.add2(zt2_1, self.conv_upsample10(self.upsample(zt3_2)))
#
#         zt1_2 = torch.cat((zt1_1, self.conv_upsample11(self.upsample(zt2_2))), 1)
#         zt1_2 = self.conv_concat5(zt1_2)
#
#         # zt1_2 = self.add1(zt1_1, self.conv_upsample11(self.upsample(zt2_2)))
#
#         # pc = self.conv4(zt1_2)
#         # pc = self.conv5(pc)
#         # pc = zt1_2
#
#         return [zt5_1, zt4_2, zt3_2, zt2_2, zt1_2]
#
# class NeighborConnectionDecoder5(nn.Module):
#     def __init__(self, channel):
#         super(NeighborConnectionDecoder5, self).__init__()
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
#         self.downsample = nn.AvgPool2d(2, 2, 0)
#         # self.conv_upsample1 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample2 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample3 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample4 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample5 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample6 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample7 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample8 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample9 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample10 = ConvBNReLU(channel, channel, 3, 1, 1)
#         # self.conv_upsample11 = ConvBNReLU(channel, channel, 3, 1, 1)
#
#         # self.conv_upsample0 = MBConvBlock(channel, channel)
#         # self.conv_upsample00 = MBConvBlock(channel, channel)
#         # self.conv_upsample000 = MBConvBlock(channel, channel)
#         self.conv_f1 = MBConvBlock(3 * channel, channel)
#         self.conv_f2 = MBConvBlock(3 * channel, channel)
#         self.conv_f3 = MBConvBlock(5 * channel, channel)
#         self.conv_f4 = MBConvBlock(3 * channel, channel)
#         self.conv_f5 = MBConvBlock(3 * channel, channel)
#
#
#         self.conv_upsample1 = MBConvBlock(3 * channel, channel)
#         self.conv_upsample2 = MBConvBlock(3 * channel, 2 * channel)
#         # self.conv_upsample3 = MBConvBlock(5 * channel, channel)
#         # self.conv_upsample4 = MBConvBlock(5 * channel, channel)
#         # self.conv_upsample5 = MBConvBlock(5 * channel, channel)
#         # self.conv_upsample6 = MBConvBlock(5 * channel, 1 * channel)
#         # self.conv_upsample7 = MBConvBlock(5 * channel, 2 * channel)
#         # self.conv_upsample8 = MBConvBlock(5 * channel, 3 * channel)
#         # self.conv_upsample9 = MBConvBlock(5 * channel, 4 * channel)
#         # self.conv_upsample10 = MBConvBlock(5 * channel, 5 * channel)
#         # self.conv_upsample11 = MBConvBlock(5 * channel, 5 * channel)
#         # self.conv_upsample12 = MBConvBlock(5 * channel, 5 * channel)
#         # self.conv_upsample13 = MBConvBlock(5 * channel, 5 * channel)
#         # self.conv_upsample10 = MBConvBlock(channel, channel)
#         # self.conv_upsample11 = MBConvBlock(channel, channel)
#
#         self.conv_downsample1 = MBConvBlock(3 * channel, channel)
#         self.conv_downsample2 = MBConvBlock(3 * channel, 2 * channel)
#         # self.conv_downsample3 = MBConvBlock(3 * channel, 3 * channel)
#         # self.conv_downsample4 = MBConvBlock(4 * channel, 4 * channel)
#         # self.conv_downsample5 = MBConvBlock(channel, channel)
#         # self.conv_downsample6 = MBConvBlock(channel, channel)
#         # self.conv_downsample7 = MBConvBlock(channel, channel)
#         # self.conv_downsample8 = MBConvBlock(channel, channel)
#         # self.conv_downsample9 = MBConvBlock(channel, channel)
#         # self.conv_downsample10 = MBConvBlock(channel, channel)
#         # self.conv_downsample11 = MBConvBlock(channel, channel)
#         # self.conv_upsample12 = ConvBNReLU(channel, channel, 3, 1, 1)
#
#         # self.conv_upsample5 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1)
#
#         # self.conv_concat2 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
#         # self.conv_concat3 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
#         # self.conv_concat4 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
#         # self.conv_concat5 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
#
#
#         self.conv_concat2 = MBConvBlock(2 * channel, 2 * channel)
#         self.conv_concat4 = MBConvBlock(2 * channel, 2 * channel)
#         self.conv_concat1 = MBConvBlock(3 * channel, 3 * channel)
#         self.conv_concat5 = MBConvBlock(3 * channel, 3 * channel)
#
#
#
#
#
#     def forward(self, zt5, zt4, zt3, zt2, zt1):
#         # zt5_1 = self.conv_upsample0(zt5)
#         # zt4_1 = self.add11(self.add1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5))), zt4)
#         # zt3_1 = self.add22(self.add2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4))), zt3)
#         # zt2_1 = self.add33(self.add3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3))), zt2)
#         # zt1_1 = self.add44(self.add4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2))), zt1)
#         # zt5_1 = self.conv_upsample0(zt5)
#         # zt4_1 = self.a1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5)), zt4)
#         # zt3_1 = self.a2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4)), zt3)
#         # zt2_1 = self.a3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3)), zt2)
#         # zt1_1 = self.a4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2)), zt1)
#         zt3_1 = zt3
#         zt2_1 = self.conv_concat2(torch.cat((self.upsample(zt3_1), zt2), dim=1))
#         zt4_1 = self.conv_concat4(torch.cat((self.downsample(zt3_1), zt4), dim=1))
#         zt1_1 = self.conv_concat1(torch.cat((self.upsample(zt2_1), zt1), dim=1))
#         zt5_1 = self.conv_concat5(torch.cat((self.downsample(zt4_1), zt5), dim=1))
#
#         # zt5_1 = self.conv_upsample0(zt5) + e5_1
#         # zt4_1 = self.conv_upsample00(self.upsample(zt5_1)) * self.conv_upsample1(self.upsample(zt5)) + zt4 + e4_1
#         # zt3_1 = self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) + zt3 + e3_1
#         # zt2_1 = self.conv_upsample4(self.upsample(zt3_1)) * self.conv_upsample5(self.upsample(zt3)) + zt2 + e2_1
#         # zt1_1 = self.conv_upsample6(self.upsample(zt2_1)) * self.conv_upsample7(self.upsample(zt2)) + zt1 + e1_1
#
#         zt5_2 = self.conv_f5(zt5_1)
#         # zt5_2 = self.conv_upsample1(zt5_1)
#         # zt5_2 = self.add0(zt5_2, e5_1)
#
#         zt4_1 = torch.cat((zt4_1, self.conv_upsample1(self.upsample(zt5_1))),  1)
#         zt4_2 = self.conv_f4(zt4_1)
#         # zt4_2 = self.conv_upsample2(zt4_1)
#         # zt4_2 = self.add4(zt4_2, e4_2)
#
#         zt1_2 = self.conv_f1(zt1_1)
#         # zt1_2 = self.conv_upsample5(zt1_1)
#         # zt1_2 = self.add1(zt1_2, e1_1)
#         # zt1_2 = self.add1(zt1_1, self.conv_upsample11(self.upsample(zt2_2)))
#
#         zt2_1 = torch.cat((zt2_1, self.conv_downsample1(self.downsample(zt1_1))), 1)
#         zt2_2 = self.conv_f2(zt2_1)
#         # zt2_2 = self.conv_upsample4(zt2_1)
#         # zt2_2 = self.add2(zt2_2, e2_2)
#
#         zt3_1 = torch.cat((self.conv_upsample2(self.upsample(zt4_1)), zt3_1, self.conv_downsample2(self.downsample(zt2_1))), 1)
#         zt3_2 = self.conv_f3(zt3_1)
#         # zt3_2 = self.conv_upsample3(zt3_1)
#         # zt3_2 = self.add3(zt3_2, e3_2)
#
#
#
#
#
#         # pc = self.conv4(zt1_2)
#         # pc = self.conv5(pc)
#         # pc = zt1_2
#
#         return [zt5_2, zt4_2, zt3_2, zt2_2, zt1_2, zt4_2, zt3_2, zt2_2]


class NeighborConnectionDecoder4(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder4, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.AvgPool2d(2, 2, 0)
        # self.conv_upsample1 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample2 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample3 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample4 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample5 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample6 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample7 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample8 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample9 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample10 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample11 = ConvBNReLU(channel, channel, 3, 1, 1)

        # self.conv_upsample0 = MBConvBlock(channel, channel)
        # self.conv_upsample00 = MBConvBlock(channel, channel)
        # self.conv_upsample000 = MBConvBlock(channel, channel)
        self.conv_upsample1 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        self.conv_upsample2 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        self.conv_upsample3 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        self.conv_upsample4 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        self.conv_upsample5 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        self.conv_upsample6 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1)
        self.conv_upsample7 = ConvBNReLU(5 * channel, 2 * channel, 3, 1, 1)
        self.conv_upsample8 = ConvBNReLU(5 * channel, 3 * channel, 3, 1, 1)
        self.conv_upsample9 = ConvBNReLU(5 * channel, 4 * channel, 3, 1, 1)
        self.conv_upsample10 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1)
        self.conv_upsample11 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1)
        self.conv_upsample12 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1)
        self.conv_upsample13 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1)
        self.conv_upsample14 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1)
        # self.conv_upsample10 = MBConvBlock(channel, channel)
        # self.conv_upsample11 = MBConvBlock(channel, channel)

        # self.conv_downsample1 = MBConvBlock(channel, channel)
        # self.conv_downsample2 = MBConvBlock(2 * channel, 2 * channel)
        # self.conv_downsample3 = MBConvBlock(3 * channel, 3 * channel)
        # self.conv_downsample4 = MBConvBlock(4 * channel, 4 * channel)
        # self.conv_downsample5 = MBConvBlock(channel, channel)
        # self.conv_downsample6 = MBConvBlock(channel, channel)
        # self.conv_downsample7 = MBConvBlock(channel, channel)
        # self.conv_downsample8 = MBConvBlock(channel, channel)
        # self.conv_downsample9 = MBConvBlock(channel, channel)
        # self.conv_downsample10 = MBConvBlock(channel, channel)
        # self.conv_downsample11 = MBConvBlock(channel, channel)
        # self.conv_upsample12 = ConvBNReLU(channel, channel, 3, 1, 1)

        # self.conv_upsample5 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1)

        # self.conv_concat2 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.conv_concat3 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.conv_concat4 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.conv_concat5 = ConvBNReLU(2 * channel, channel, 3, 1, 1)

        # self.conv_concat1 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1)
        self.conv_concat2 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1)
        self.conv_concat3 = ConvBNReLU(3 * channel, 3 * channel, 3, 1, 1)
        self.conv_concat4 = ConvBNReLU(4 * channel, 4 * channel, 3, 1, 1)
        self.conv_concat5 = ConvBNReLU(5 * channel, 5 * channel, 3, 1, 1)
        # self.relu5 = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU())
        # self.relu4 = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU())
        # self.relu3 = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU())
        # self.relu2 = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU())
        # self.relu1 = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU())

        # self.relu5 = nn.Sequential(nn.Sigmoid())
        # self.relu4 = nn.Sequential(nn.Sigmoid())
        # self.relu3 = nn.Sequential(nn.Sigmoid())
        # self.relu2 = nn.Sequential(nn.Sigmoid())
        # self.relu1 = nn.Sequential(nn.Sigmoid())

        # self.add4 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.add3 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.add2 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.add1 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.add0 = ConvBNReLU(2 * channel, channel, 3, 1, 1)

        # self.a5 = Attention2()
        # self.a4 = Attention2()
        # self.a3 = Attention2()
        # self.a2 = Attention2()
        # self.a1 = Attention2()

        # self.add44 = M_add(64)
        # self.add33 = M_add(64)
        # self.add22 = M_add(64)
        # self.add11 = M_add(64)
        # self.add00 = M_add(64)

        # self.add4 = M_add(64)
        # self.add3 = M_add(64)
        # self.add2 = M_add(64)
        # self.add1 = M_add(64)
        # self.add0 = M_add(64)

        # self.conv4 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        # self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, zt5, zt4, zt3, zt2, zt1, e5_1, e4_2, e3_2, e2_2, e1_1):
        # zt5_1 = self.conv_upsample0(zt5)
        # zt4_1 = self.add11(self.add1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5))), zt4)
        # zt3_1 = self.add22(self.add2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4))), zt3)
        # zt2_1 = self.add33(self.add3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3))), zt2)
        # zt1_1 = self.add44(self.add4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2))), zt1)
        # zt5_1 = self.conv_upsample0(zt5)
        # zt4_1 = self.a1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5)), zt4)
        # zt3_1 = self.a2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4)), zt3)
        # zt2_1 = self.a3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3)), zt2)
        # zt1_1 = self.a4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2)), zt1)
        # zt1_1 = zt1
        zt1_1 = zt1
        zt2_1 = self.conv_concat2(torch.cat((self.downsample(zt1_1), zt2), dim=1))
        zt3_1 = self.conv_concat3(torch.cat((self.downsample(zt2_1), zt3), dim=1))
        zt4_1 = self.conv_concat4(torch.cat((self.downsample(zt3_1), zt4), dim=1))
        zt5_1 = self.conv_concat5(torch.cat((self.downsample(zt4_1), zt5), dim=1))

        # zt5_1 = self.conv_upsample0(zt5) + e5_1
        # zt4_1 = self.conv_upsample00(self.upsample(zt5_1)) * self.conv_upsample1(self.upsample(zt5)) + zt4 + e4_1
        # zt3_1 = self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) + zt3 + e3_1
        # zt2_1 = self.conv_upsample4(self.upsample(zt3_1)) * self.conv_upsample5(self.upsample(zt3)) + zt2 + e2_1
        # zt1_1 = self.conv_upsample6(self.upsample(zt2_1)) * self.conv_upsample7(self.upsample(zt2)) + zt1 + e1_1

        zt5_2 = self.conv_upsample1(zt5_1) + self.conv_upsample10(e5_1)
        # zt5_2 = self.conv_upsample1(zt5_1) + zt5_1
        # zt5_2 = self.conv_upsample1(torch.cat((e5_1, zt5_1), dim=1))

        # zt5_2 = self.conv_upsample1(zt5_1) * (1 + e5_1.sigmoid())
        # zt5_2 = self.add0(torch.cat((zt5_2, e5_1), dim=1))

        zt4_1 = torch.cat((zt4_1, self.conv_upsample6(self.upsample(zt5_1))), 1)
        # zt4_2 = self.conv_upsample2(zt4_1) + e4_2
        zt4_2 = self.conv_upsample2(zt4_1) + self.conv_upsample11(e4_2)
        # zt4_2 = self.conv_upsample2(torch.cat((e4_2, zt4_1), dim=1))

        # zt4_2 = self.conv_upsample2(zt4_1) * (1 + e4_2.sigmoid())
        # zt4_2 = self.add4(torch.cat((zt4_2, e4_2), dim=1))

        zt3_1 = torch.cat((zt3_1, self.conv_upsample7(self.upsample(zt4_1))), 1)
        # zt3_2 = self.conv_upsample3(zt3_1) + e3_2
        zt3_2 = self.conv_upsample3(zt3_1) + self.conv_upsample12(e3_2)
        # zt3_2 = self.conv_upsample3(torch.cat((e3_2, zt3_1), dim=1))
        # zt3_2 = self.conv_upsample3(zt3_1) * (1 + e3_2.sigmoid())
        # zt3_2 = self.add3(torch.cat((zt3_2, e3_2), dim=1))

        zt2_1 = torch.cat((zt2_1, self.conv_upsample8(self.upsample(zt3_1))), 1)
        # zt2_2 = self.conv_upsample4(zt2_1) + e2_2
        zt2_2 = self.conv_upsample4(zt2_1) + self.conv_upsample13(e2_2)
        # zt2_2 = self.conv_upsample4(torch.cat((e2_2, zt2_1), dim=1))
        # zt2_2 = self.conv_upsample4(zt2_1) * (1 + e2_2.sigmoid())
        # zt2_2 = self.add2(torch.cat((zt2_2, e2_2), dim=1))

        zt1_1 = torch.cat((zt1_1, self.conv_upsample9(self.upsample(zt2_1))), 1)
        zt1_2 = self.conv_upsample5(zt1_1) + self.conv_upsample14(e1_1)
        # zt1_2 = self.conv_upsample5(zt1_1) + e1_1
        # zt1_2 = self.conv_upsample5(torch.cat((e1_1, zt1_1), dim=1))
        # print(zt1_2.mean((2, 3), keepdim=True))
        # zt1_2 = zt1_2 + e1_1
        # zt1_2 = self.conv_upsample5(zt1_1)
        # zt1_2 = self.conv_upsample5(zt1_1) * (1 + e1_1.sigmoid())
        # zt1_2 = self.add1(torch.cat((zt1_2, e1_1), dim=1))

        # pc = self.conv4(zt1_2)
        # pc = self.conv5(pc)
        # pc = zt1_2

        return [zt5_2, zt4_2, zt3_2, zt2_2, zt1_2]


class NeighborConnectionDecoder4u(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder4u, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.AvgPool2d(2, 2, 0)
        # self.conv_upsample1 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample2 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample3 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample4 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample5 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample6 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample7 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample8 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample9 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample10 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample11 = ConvBNReLU(channel, channel, 3, 1, 1)

        # self.conv_upsample0 = MBConvBlock(channel, channel)
        # self.conv_upsample00 = MBConvBlock(channel, channel)
        # self.conv_upsample000 = MBConvBlock(channel, channel)
        # self.conv_upsample1 = ConvBNReLU(1 * channel, channel, 3, 1, 1)
        # self.conv_upsample2 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.conv_upsample3 = ConvBNReLU(3 * channel, channel, 3, 1, 1)
        # self.conv_upsample4 = ConvBNReLU(4 * channel, channel, 3, 1, 1)
        # self.conv_upsample5 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        # self.conv_upsample6 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1)
        # self.conv_upsample7 = ConvBNReLU(5 * channel, 2 * channel, 3, 1, 1)
        # self.conv_upsample8 = ConvBNReLU(5 * channel, 3 * channel, 3, 1, 1)
        # self.conv_upsample9 = ConvBNReLU(5 * channel, 4 * channel, 3, 1, 1)
        # self.conv_upsample10 = MBConvBlock(5 * channel, 5 * channel)
        # self.conv_upsample11 = MBConvBlock(5 * channel, 5 * channel)
        # self.conv_upsample12 = MBConvBlock(5 * channel, 5 * channel)
        # self.conv_upsample13 = MBConvBlock(5 * channel, 5 * channel)
        # self.conv_upsample10 = MBConvBlock(channel, channel)
        # self.conv_upsample11 = MBConvBlock(channel, channel)

        # self.conv_downsample1 = MBConvBlock(channel, channel)
        # self.conv_downsample2 = MBConvBlock(2 * channel, 2 * channel)
        # self.conv_downsample3 = MBConvBlock(3 * channel, 3 * channel)
        # self.conv_downsample4 = MBConvBlock(4 * channel, 4 * channel)
        # self.conv_downsample5 = MBConvBlock(channel, channel)
        # self.conv_downsample6 = MBConvBlock(channel, channel)
        # self.conv_downsample7 = MBConvBlock(channel, channel)
        # self.conv_downsample8 = MBConvBlock(channel, channel)
        # self.conv_downsample9 = MBConvBlock(channel, channel)
        # self.conv_downsample10 = MBConvBlock(channel, channel)
        # self.conv_downsample11 = MBConvBlock(channel, channel)
        # self.conv_upsample12 = ConvBNReLU(channel, channel, 3, 1, 1)

        # self.conv_upsample5 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1)

        # self.conv_concat2 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.conv_concat3 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.conv_concat4 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.conv_concat5 = ConvBNReLU(2 * channel, channel, 3, 1, 1)

        self.conv_concat1 = ConvBNReLU_BR(1 * channel, 1 * channel, 3, 1, 1)
        self.conv_concat2 = ConvBNReLU_BR(2 * channel, 1 * channel, 3, 1, 1)
        self.conv_concat3 = ConvBNReLU_BR(2 * channel, 1 * channel, 3, 1, 1)
        self.conv_concat4 = ConvBNReLU_BR(2 * channel, 1 * channel, 3, 1, 1)
        self.conv_concat5 = ConvBNReLU_BR(2 * channel, 1 * channel, 3, 1, 1)
        # self.relu5 = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU())
        # self.relu4 = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU())
        # self.relu3 = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU())
        # self.relu2 = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU())
        # self.relu1 = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU())

        # self.relu5 = nn.Sequential(nn.Sigmoid())
        # self.relu4 = nn.Sequential(nn.Sigmoid())
        # self.relu3 = nn.Sequential(nn.Sigmoid())
        # self.relu2 = nn.Sequential(nn.Sigmoid())
        # self.relu1 = nn.Sequential(nn.Sigmoid())

        # self.add4 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.add3 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.add2 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.add1 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.add0 = ConvBNReLU(2 * channel, channel, 3, 1, 1)

        # self.a5 = Attention2()
        # self.a4 = Attention2()
        # self.a3 = Attention2()
        # self.a2 = Attention2()
        # self.a1 = Attention2()

        # self.add44 = M_add(64)
        # self.add33 = M_add(64)
        # self.add22 = M_add(64)
        # self.add11 = M_add(64)
        # self.add00 = M_add(64)

        # self.add4 = M_add(64)
        # self.add3 = M_add(64)
        # self.add2 = M_add(64)
        # self.add1 = M_add(64)
        # self.add0 = M_add(64)

        # self.conv4 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        # self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, zt5, zt4, zt3, zt2, zt1):
        # zt5_1 = self.conv_upsample0(zt5)
        # zt4_1 = self.add11(self.add1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5))), zt4)
        # zt3_1 = self.add22(self.add2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4))), zt3)
        # zt2_1 = self.add33(self.add3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3))), zt2)
        # zt1_1 = self.add44(self.add4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2))), zt1)
        # zt5_1 = self.conv_upsample0(zt5)
        # zt4_1 = self.a1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5)), zt4)
        # zt3_1 = self.a2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4)), zt3)
        # zt2_1 = self.a3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3)), zt2)
        # zt1_1 = self.a4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2)), zt1)
        # zt1_1 = zt1
        zt5_1 = self.conv_concat1(zt5)
        zt4_1 = self.conv_concat2(torch.cat((self.upsample(zt5_1), zt4), dim=1))
        zt3_1 = self.conv_concat3(torch.cat((self.upsample(zt4_1), zt3), dim=1))
        zt2_1 = self.conv_concat4(torch.cat((self.upsample(zt3_1), zt2), dim=1))
        zt1_1 = self.conv_concat5(torch.cat((self.upsample(zt2_1), zt1), dim=1))

        # zt5_1 = self.conv_upsample0(zt5) + e5_1
        # zt4_1 = self.conv_upsample00(self.upsample(zt5_1)) * self.conv_upsample1(self.upsample(zt5)) + zt4 + e4_1
        # zt3_1 = self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) + zt3 + e3_1
        # zt2_1 = self.conv_upsample4(self.upsample(zt3_1)) * self.conv_upsample5(self.upsample(zt3)) + zt2 + e2_1
        # zt1_1 = self.conv_upsample6(self.upsample(zt2_1)) * self.conv_upsample7(self.upsample(zt2)) + zt1 + e1_1

        # zt5_2 = self.conv_upsample1(zt5_1) + e5_1
        # zt5_2 = self.conv_upsample1(zt5_1)
        # zt5_2 = self.conv_upsample1(torch.cat((e5_1, zt5_1), dim=1))

        # zt5_2 = self.conv_upsample1(zt5_1) * (1 + e5_1.sigmoid())
        # zt5_2 = self.add0(torch.cat((zt5_2, e5_1), dim=1))

        # zt4_1 = torch.cat((zt4_1, self.conv_upsample6(self.upsample(zt5_1))), 1)
        # zt4_2 = self.conv_upsample2(zt4_1) + e4_2
        # zt4_2 = self.conv_upsample2(zt4_1)
        # zt4_2 = self.conv_upsample2(torch.cat((e4_2, zt4_1), dim=1))

        # zt4_2 = self.conv_upsample2(zt4_1) * (1 + e4_2.sigmoid())
        # zt4_2 = self.add4(torch.cat((zt4_2, e4_2), dim=1))

        # zt3_1 = torch.cat((zt3_1, self.conv_upsample7(self.upsample(zt4_1))), 1)
        # zt3_2 = self.conv_upsample3(zt3_1) + e3_2
        # zt3_2 = self.conv_upsample3(zt3_1)
        # zt3_2 = self.conv_upsample3(torch.cat((e3_2, zt3_1), dim=1))
        # zt3_2 = self.conv_upsample3(zt3_1) * (1 + e3_2.sigmoid())
        # zt3_2 = self.add3(torch.cat((zt3_2, e3_2), dim=1))

        # zt2_1 = torch.cat((zt2_1, self.conv_upsample8(self.upsample(zt3_1))), 1)
        # zt2_2 = self.conv_upsample4(zt2_1) + e2_2
        # zt2_2 = self.conv_upsample4(zt2_1)
        # zt2_2 = self.conv_upsample4(torch.cat((e2_2, zt2_1), dim=1))
        # zt2_2 = self.conv_upsample4(zt2_1) * (1 + e2_2.sigmoid())
        # zt2_2 = self.add2(torch.cat((zt2_2, e2_2), dim=1))

        # zt1_1 = torch.cat((zt1_1, self.conv_upsample9(self.upsample(zt2_1))), 1)

        # zt1_2 = self.conv_upsample5(zt1_1)
        # zt1_2 = self.conv_upsample5(torch.cat((e1_1, zt1_1), dim=1))
        # print(zt1_2.mean((2, 3), keepdim=True))
        # zt1_2 = zt1_2 + e1_1
        # zt1_2 = self.conv_upsample5(zt1_1)
        # zt1_2 = self.conv_upsample5(zt1_1) * (1 + e1_1.sigmoid())
        # zt1_2 = self.add1(torch.cat((zt1_2, e1_1), dim=1))

        # pc = self.conv4(zt1_2)
        # pc = self.conv5(pc)
        # pc = zt1_2

        return [zt5_1, zt4_1, zt3_1, zt2_1, zt1_1]

class NeighborConnectionDecoder401(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder401, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.AvgPool2d(2, 2, 0)

        self.conv1_1 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1)
        self.conv1_2 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1, dilation=1)
        self.conv1_3 = ConvBNReLU(3 * channel, 3 * channel, 3, 1, 1, dilation=1)
        self.conv1_4 = ConvBNReLU(4 * channel, 4 * channel, 3, 1, 1, dilation=1)
        self.conv1_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1, dilation=1)

        self.conv2_1 = ConvBNReLU(2 * channel, 1 * channel, 3, 1, 1, dilation=1)
        self.conv2_2 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 2, dilation=2)
        self.conv2_3 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 2, dilation=2)
        self.conv2_4 = ConvBNReLU(3 * channel, 3 * channel, 3, 1, 2, dilation=2)
        self.conv2_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 2, dilation=2)

        self.conv3_1 = ConvBNReLU(3 * channel, 1 * channel, 3, 1, 1, dilation=1)
        self.conv3_2 = ConvBNReLU(3 * channel, 1 * channel, 3, 1, 2, dilation=2)
        self.conv3_3 = ConvBNReLU(3 * channel, 3 * channel, 3, 1, 3, dilation=3)
        self.conv3_4 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 3, dilation=3)
        self.conv3_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 3, dilation=3)


        self.conv4_1 = ConvBNReLU(4 * channel, 1 * channel, 3, 1, 1, dilation=1)
        self.conv4_2 = ConvBNReLU(4 * channel, 1 * channel, 3, 1, 2, dilation=2)
        self.conv4_3 = ConvBNReLU(4 * channel, 1 * channel, 3, 1, 3, dilation=3)
        self.conv4_4 = ConvBNReLU(4 * channel, 4 * channel, 3, 1, 5, dilation=5)
        self.conv4_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 5, dilation=5)

        self.conv5_1 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1, dilation=1)
        self.conv5_2 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 2, dilation=2)
        self.conv5_3 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 3, dilation=3)
        self.conv5_4 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 5, dilation=5)
        self.conv5_5 = ConvBNReLU(5 * channel, 5 * channel, 3, 1, 7, dilation=7)

        self.conv5_5f = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 7, dilation=7)

        # self.conv1_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # self.conv2_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 2, dilation=2)
        # self.conv3_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 3, dilation=3)
        # self.conv4_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 5, dilation=5)





    def forward(self, zt1, zt2, zt3, zt4, zt5):
        # zt5_1 = self.conv_upsample0(zt5)
        # zt4_1 = self.add11(self.add1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5))), zt4)
        # zt3_1 = self.add22(self.add2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4))), zt3)
        # zt2_1 = self.add33(self.add3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3))), zt2)
        # zt1_1 = self.add44(self.add4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2))), zt1)
        # zt5_1 = self.conv_upsample0(zt5)
        # zt4_1 = self.a1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5)), zt4)
        # zt3_1 = self.a2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4)), zt3)
        # zt2_1 = self.a3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3)), zt2)
        # zt1_1 = self.a4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2)), zt1)
        # zt1_1 = zt1

        zt1_1 = self.conv1_1(zt1)

        zt2_1 = self.conv2_2(torch.cat((self.upsample(zt1_1), zt2), dim=1))
        zt1_2 = self.conv2_1(zt2_1)

        zt3_1 = self.conv3_3(torch.cat((self.upsample(zt2_1), zt3), dim=1))
        zt2_2 = self.conv3_2(zt3_1)
        zt1_3 = self.conv3_1(zt3_1)

        zt4_1 = self.conv4_4(torch.cat((self.upsample(zt3_1), zt4), dim=1))
        zt3_2 = self.conv4_3(zt4_1)
        zt2_3 = self.conv4_2(zt4_1)
        zt1_4 = self.conv4_1(zt4_1)

        zt5_1 = self.conv5_5(torch.cat((self.upsample(zt4_1), zt5), dim=1))
        zt4_2 = self.conv5_4(zt5_1)
        zt3_3 = self.conv5_3(zt5_1)
        zt2_4 = self.conv5_2(zt5_1)
        zt1_5 = self.conv5_1(zt5_1)

        zt5_1 = self.conv5_5f(zt5_1)



        zt1_4 = self.conv1_2(torch.cat((self.downsample(zt1_5), zt1_4), dim=1))
        zt1_3 = self.conv1_3(torch.cat((self.downsample(zt1_4), zt1_3), dim=1))
        zt1_2 = self.conv1_4(torch.cat((self.downsample(zt1_3), zt1_2), dim=1))
        zt1_1 = self.conv1_5(torch.cat((self.downsample(zt1_2), zt1_1), dim=1))

        zt2_3 = self.conv2_3(torch.cat((self.downsample(zt2_4), zt2_3), dim=1))
        zt2_2 = self.conv2_4(torch.cat((self.downsample(zt2_3), zt2_2), dim=1))
        zt2_1 = self.conv2_5(torch.cat((self.downsample(zt2_2), zt2_1), dim=1))

        zt3_2 = self.conv3_4(torch.cat((self.downsample(zt3_3), zt3_2), dim=1))
        zt3_1 = self.conv3_5(torch.cat((self.downsample(zt3_2), zt3_1), dim=1))

        zt4_1 = self.conv4_5(torch.cat((self.downsample(zt4_2), zt4_1), dim=1))




        return [zt1_1, zt2_1, zt3_1, zt4_1, zt5_1]


class NeighborConnectionDecoder402(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder402, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.downsample1 = nn.AvgPool2d(2, 2, 0)
        self.downsample2 = nn.AvgPool2d(4, 4, 0)
        self.downsample3 = nn.AvgPool2d(8, 8, 0)
        self.downsample4 = nn.AvgPool2d(16, 16, 0)
        self.conv0_1 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_2 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_3 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_4 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_5 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 2, dilation=2))
        self.conv0_6 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 2, dilation=2))
        self.conv0_7 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 2, dilation=2))
        self.conv0_8 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 2, dilation=2))
        self.conv0_9 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 3, dilation=3))
        self.conv0_10 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 3, dilation=3))
        self.conv0_11 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 3, dilation=3))
        self.conv0_12 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 3, dilation=3))
        self.conv0_13 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 5, dilation=5))
        self.conv0_14 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 5, dilation=5))
        self.conv0_15 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 5, dilation=5))
        self.conv0_16 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 5, dilation=5))
        self.conv0_17 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 7, dilation=7))
        self.conv0_18 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 7, dilation=7))
        self.conv0_19 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 7, dilation=7))
        self.conv0_20 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 7, dilation=7))

        # self.conv1_1 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1)
        self.conv1_1 = nn.Sequential(ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv2_1 = nn.Sequential(ConvBNReLU(5 * channel, 1 * channel, 3, 1, 2, dilation=2))
        self.conv3_1 = nn.Sequential(ConvBNReLU(5 * channel, 1 * channel, 3, 1, 3, dilation=3))
        self.conv4_1 = nn.Sequential(ConvBNReLU(5 * channel, 1 * channel, 3, 1, 5, dilation=5))
        self.conv5_1 = nn.Sequential(ConvBNReLU(5 * channel, 1 * channel, 3, 1, 7, dilation=7))
        # self.conv1_2 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1, dilation=1)
        # self.conv1_3 = ConvBNReLU(3 * channel, 3 * channel, 3, 1, 1, dilation=1)
        # self.conv1_4 = ConvBNReLU(4 * channel, 4 * channel, 3, 1, 1, dilation=1)
        # self.conv1_5 = ConvBNReLU(5 * channel, 5 * channel, 3, 1, 1, dilation=1)


        # self.conv2_1 = ConvBNReLU(2 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # # self.conv2_1 = MBSA('cuda:2', 2 * channel, channel, 2, 2)
        # # self.conv2_2 = MBSA('cuda:7', 2 * channel, 2 * channel, 2, 2)
        # self.conv2_2 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 2, dilation=2)
        # # self.conv2_3 = ConvBNReLU(2 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # # self.conv2_4 = ConvBNReLU(2 * channel, 1 * channel, 3, 1, 2, dilation=2)
        # self.conv2_3 = MBSA('cuda:2', 1 * channel, channel, 2, 2)
        # self.conv2_4 = MBSA('cuda:2', 1 * channel, channel, 2, 2, 32)
        # # self.conv2_5 = ConvBNReLU(5 * channel, 5 * channel, 3, 1, 2, dilation=2)
        #
        # self.conv3_1 = ConvBNReLU(3 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # # self.conv3_1 = MBSA('cuda:2', 3 * channel, channel, 3, 3)
        # # self.conv3_2 = MBSA('cuda:7', 3 * channel, 3 * channel, 3, 3)
        # self.conv3_2 = ConvBNReLU(3 * channel, 3 * channel, 3, 1, 3, dilation=3)
        # # self.conv3_3 = ConvBNReLU(3 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # # self.conv3_4 = ConvBNReLU(3 * channel, 1 * channel, 3, 1, 2, dilation=2)
        # # self.conv3_5 = ConvBNReLU(3 * channel, 1 * channel, 3, 1, 3, dilation=3)
        # self.conv3_3 = MBSA('cuda:2', 1 * channel, channel, 3, 3, 64)
        # self.conv3_4 = MBSA('cuda:2', 1 * channel, channel, 3, 3, 32)
        # self.conv3_5 = MBSA('cuda:2', 1 * channel, channel, 3, 3, 16)
        #
        #
        # self.conv4_1 = ConvBNReLU(4 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # # self.conv4_1 = MBSA('cuda:2', 4 * channel, channel, 5, 5)
        # # self.conv4_2 = MBSA('cuda:7', 4 * channel, 4 * channel, 5, 5)
        # self.conv4_2 = ConvBNReLU(4 * channel, 4 * channel, 3, 1, 5, dilation=5)
        # self.conv4_3 = ConvBNReLU(4 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # self.conv4_4 = ConvBNReLU(4 * channel, 1 * channel, 3, 1, 2, dilation=2)
        # self.conv4_5 = ConvBNReLU(4 * channel, 1 * channel, 3, 1, 3, dilation=3)
        # self.conv4_6 = ConvBNReLU(4 * channel, 1 * channel, 3, 1, 5, dilation=5)
        #
        #
        # self.conv5_1 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # self.conv5_1 = MBSA('cuda:2', 5 * channel, channel, 7, 7)
        # self.conv5_2 = ConvBNReLU(5 * channel, 5 * channel, 3, 1, 7, dilation=7)
        # self.conv5_3 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # self.conv5_4 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 2, dilation=2)
        # self.conv5_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 3, dilation=3)
        # self.conv5_6 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 5, dilation=5)
        # self.conv5_7 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 7, dilation=7)




        # self.conv5_5f = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 7, dilation=7)

        # self.conv1_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # self.conv2_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 2, dilation=2)
        # self.conv3_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 3, dilation=3)
        # self.conv4_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 5, dilation=5)





    def forward(self, zt1, zt2, zt3, zt4, zt5):
        # zt5_1 = self.conv_upsample0(zt5)
        # zt4_1 = self.add11(self.add1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5))), zt4)
        # zt3_1 = self.add22(self.add2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4))), zt3)
        # zt2_1 = self.add33(self.add3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3))), zt2)
        # zt1_1 = self.add44(self.add4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2))), zt1)
        # zt5_1 = self.conv_upsample0(zt5)
        # zt4_1 = self.a1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5)), zt4)
        # zt3_1 = self.a2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4)), zt3)
        # zt2_1 = self.a3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3)), zt2)
        # zt1_1 = self.a4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2)), zt1)
        # zt1_1 = zt1



        # # zt2_1 = self.conv2_2(torch.cat((self.upsample(zt1_1), zt2), dim=1))
        # # zt2_2 = self.conv2_1(zt2_1)
        #
        # # zt2_2 = self.conv2_1(torch.cat((self.conv2_3(zt2_1), self.conv2_4(zt2_1)), dim=1))
        #
        # zt3_1 = self.conv3_2(torch.cat((self.upsample(zt2_1), zt3), dim=1))
        # # zt3_2 = self.conv3_1(zt3_1)
        #
        # zt3_2 = self.conv3_1(torch.cat((self.conv3_3(zt3_1), self.conv3_4(zt3_1), self.conv3_5(zt3_1)), dim=1))
        #
        #
        # zt4_1 = self.conv4_2(torch.cat((self.upsample(zt3_1), zt4), dim=1))
        # # zt4_2 = self.conv4_1(zt4_1)
        #
        # zt4_2 = self.conv4_1(torch.cat((self.conv4_3(zt4_1), self.conv4_4(zt4_1), self.conv4_5(zt4_1), self.conv4_6(zt4_1)), dim=1))
        #
        #
        # zt5_1 = self.conv5_2(torch.cat((self.upsample(zt4_1), zt5), dim=1))
        # # zt5_2 = self.conv5_1(zt5_1)
        #
        # zt5_2 = self.conv5_1(torch.cat((self.conv5_3(zt5_1), self.conv5_4(zt5_1), self.conv5_5(zt5_1), self.conv5_6(zt5_1), self.conv5_7(zt5_1)), dim=1))
        zt1_1 = self.conv1_1(torch.cat((zt1, self.conv0_1(self.downsample1(zt2)), self.conv0_2(self.downsample2(zt3)), self.conv0_3(self.downsample3(zt4)), self.conv0_4(self.downsample4(zt5))), dim=1))
        zt2_2 = self.conv2_1(torch.cat((self.conv0_5(self.upsample1(zt1_1)), zt2, self.conv0_6(self.downsample1(zt3)), self.conv0_7(self.downsample2(zt4)), self.conv0_8(self.downsample3(zt5))), dim=1))
        zt3_2 = self.conv3_1(torch.cat((self.conv0_9(self.upsample2(zt1_1)), self.conv0_10(self.upsample1(zt2_2)), zt3, self.conv0_11(self.downsample1(zt4)), self.conv0_12(self.downsample2(zt5))), dim=1))
        zt4_2 = self.conv4_1(torch.cat((self.conv0_13(self.upsample3(zt1_1)), self.conv0_14(self.upsample2(zt2_2)), self.conv0_15(self.upsample1(zt3_2)), zt4, self.conv0_16(self.downsample1(zt5))), dim=1))
        zt5_2 = self.conv5_1(torch.cat((self.conv0_17(self.upsample4(zt1_1)), self.conv0_18(self.upsample3(zt2_2)), self.conv0_19(self.upsample2(zt3_2)), self.conv0_20(self.upsample1(zt4_2)), zt5), dim=1))

        # zt1_1 = self.conv1_1(zt1 + self.conv0_1(self.downsample1(zt2)) + self.conv0_2(self.downsample2(zt3)) +
        #                                 self.conv0_3(self.downsample3(zt4)) + self.conv0_4(self.downsample4(zt5)))
        # zt2_2 = self.conv2_1(self.conv0_5(self.upsample1(zt1_1)) + zt2 + self.conv0_6(self.downsample1(zt3)) +
        #                                 self.conv0_7(self.downsample2(zt4)) + self.conv0_8(self.downsample3(zt5)))
        # zt3_2 = self.conv3_1(self.conv0_9(self.upsample2(zt1_1)) + self.conv0_10(self.upsample1(zt2_2)) + zt3 +
        #                                 self.conv0_11(self.downsample1(zt4)) + self.conv0_12(self.downsample2(zt5)))
        # zt4_2 = self.conv4_1(self.conv0_13(self.upsample3(zt1_1)) + self.conv0_14(self.upsample2(zt2_2)) +
        #                                 self.conv0_15(self.upsample1(zt3_2)) + zt4 +
        #                                 self.conv0_16(self.downsample1(zt5)))
        # zt5_2 = self.conv5_1(self.conv0_17(self.upsample4(zt1_1)) + self.conv0_18(self.upsample3(zt2_2)) +
        #                                 self.conv0_19(self.upsample2(zt3_2)) + self.conv0_20(self.upsample1(zt4_2)) +
        #                                 zt5)

        # zt1_1 = self.conv1_1(torch.cat((zt1, self.conv0_1(self.downsample1(zt2)), self.conv0_2(self.downsample2(zt3)),
        #                                 self.conv0_3(self.downsample3(zt4)), self.conv0_4(self.downsample4(zt5))),
        #                                dim=1))
        # zt2_2 = self.conv2_1(torch.cat((self.conv0_5(self.upsample1(zt1_1)), zt2, self.conv0_6(self.downsample1(zt3)),
        #                                 self.conv0_7(self.downsample2(zt4)), self.conv0_8(self.downsample3(zt5))),
        #                                dim=1))
        # zt3_2 = self.conv3_1(torch.cat((self.conv0_9(self.upsample2(zt1_1)), self.conv0_10(self.upsample1(zt2_2)), zt3,
        #                                 self.conv0_11(self.downsample1(zt4)), self.conv0_12(self.downsample2(zt5))),
        #                                dim=1))
        # zt4_2 = self.conv4_1(torch.cat((self.conv0_13(self.upsample3(zt1_1)), self.conv0_14(self.upsample2(zt2_2)),
        #                                 self.conv0_15(self.upsample1(zt3_2)), zt4,
        #                                 self.conv0_16(self.downsample1(zt5))), dim=1))
        # zt5_2 = self.conv5_1(torch.cat((self.conv0_17(self.upsample4(zt1_1)), self.conv0_18(self.upsample3(zt2_2)),
        #                                 self.conv0_19(self.upsample2(zt3_2)), self.conv0_20(self.upsample1(zt4_2)),
        #                                 zt5), dim=1))

        # zt1_4 = self.conv1_2(torch.cat((self.downsample(zt1_5), zt1_4), dim=1))
        # zt1_3 = self.conv1_3(torch.cat((self.downsample(zt1_4), zt1_3), dim=1))
        # zt1_2 = self.conv1_4(torch.cat((self.downsample(zt1_3), zt1_2), dim=1))
        # zt1_1 = self.conv1_5(torch.cat((self.downsample(zt1_2), zt1_1), dim=1))
        #
        # zt2_3 = self.conv2_3(torch.cat((self.downsample(zt2_4), zt2_3), dim=1))
        # zt2_2 = self.conv2_4(torch.cat((self.downsample(zt2_3), zt2_2), dim=1))
        # zt2_1 = self.conv2_5(torch.cat((self.downsample(zt2_2), zt2_1), dim=1))
        #
        # zt3_2 = self.conv3_4(torch.cat((self.downsample(zt3_3), zt3_2), dim=1))
        # zt3_1 = self.conv3_5(torch.cat((self.downsample(zt3_2), zt3_1), dim=1))
        #
        # zt4_1 = self.conv4_5(torch.cat((self.downsample(zt4_2), zt4_1), dim=1))


        return [zt1_1, zt2_2, zt3_2, zt4_2, zt5_2]



class NeighborConnectionDecoder403(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder403, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.downsample1 = nn.AvgPool2d(2, 2, 0)
        self.downsample2 = nn.AvgPool2d(4, 4, 0)
        self.downsample3 = nn.AvgPool2d(8, 8, 0)
        self.downsample4 = nn.AvgPool2d(16, 16, 0)
        self.conv0_1 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_2 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_3 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_4 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_5 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_6 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_7 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_8 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_9 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_10 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_11 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_12 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_13 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_14 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_15 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_16 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_17 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_18 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_19 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))
        self.conv0_20 = nn.Sequential(ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1))


        # self.conv1_1 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1, dilation=1)
        self.conv1_1 = nn.Sequential(ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1))
        self.conv2_1 = nn.Sequential(ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1))
        self.conv3_1 = nn.Sequential(ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1))
        self.conv4_1 = nn.Sequential(ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1))
        self.conv5_1 = nn.Sequential(ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1))
        # self.conv1_2 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1, dilation=1)
        # self.conv1_3 = ConvBNReLU(3 * channel, 3 * channel, 3, 1, 1, dilation=1)
        # self.conv1_4 = ConvBNReLU(4 * channel, 4 * channel, 3, 1, 1, dilation=1)
        # self.conv1_5 = ConvBNReLU(5 * channel, 5 * channel, 3, 1, 1, dilation=1)


        # self.conv2_1 = ConvBNReLU(2 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # # self.conv2_1 = MBSA('cuda:2', 2 * channel, channel, 2, 2)
        # # self.conv2_2 = MBSA('cuda:7', 2 * channel, 2 * channel, 2, 2)
        # self.conv2_2 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 2, dilation=2)
        # # self.conv2_3 = ConvBNReLU(2 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # # self.conv2_4 = ConvBNReLU(2 * channel, 1 * channel, 3, 1, 2, dilation=2)
        # self.conv2_3 = MBSA('cuda:2', 1 * channel, channel, 2, 2)
        # self.conv2_4 = MBSA('cuda:2', 1 * channel, channel, 2, 2, 32)
        # # self.conv2_5 = ConvBNReLU(5 * channel, 5 * channel, 3, 1, 2, dilation=2)
        #
        # self.conv3_1 = ConvBNReLU(3 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # # self.conv3_1 = MBSA('cuda:2', 3 * channel, channel, 3, 3)
        # # self.conv3_2 = MBSA('cuda:7', 3 * channel, 3 * channel, 3, 3)
        # self.conv3_2 = ConvBNReLU(3 * channel, 3 * channel, 3, 1, 3, dilation=3)
        # # self.conv3_3 = ConvBNReLU(3 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # # self.conv3_4 = ConvBNReLU(3 * channel, 1 * channel, 3, 1, 2, dilation=2)
        # # self.conv3_5 = ConvBNReLU(3 * channel, 1 * channel, 3, 1, 3, dilation=3)
        # self.conv3_3 = MBSA('cuda:2', 1 * channel, channel, 3, 3, 64)
        # self.conv3_4 = MBSA('cuda:2', 1 * channel, channel, 3, 3, 32)
        # self.conv3_5 = MBSA('cuda:2', 1 * channel, channel, 3, 3, 16)
        #
        #
        # self.conv4_1 = ConvBNReLU(4 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # # self.conv4_1 = MBSA('cuda:2', 4 * channel, channel, 5, 5)
        # # self.conv4_2 = MBSA('cuda:7', 4 * channel, 4 * channel, 5, 5)
        # self.conv4_2 = ConvBNReLU(4 * channel, 4 * channel, 3, 1, 5, dilation=5)
        # self.conv4_3 = ConvBNReLU(4 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # self.conv4_4 = ConvBNReLU(4 * channel, 1 * channel, 3, 1, 2, dilation=2)
        # self.conv4_5 = ConvBNReLU(4 * channel, 1 * channel, 3, 1, 3, dilation=3)
        # self.conv4_6 = ConvBNReLU(4 * channel, 1 * channel, 3, 1, 5, dilation=5)
        #
        #
        # self.conv5_1 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # self.conv5_1 = MBSA('cuda:2', 5 * channel, channel, 7, 7)
        # self.conv5_2 = ConvBNReLU(5 * channel, 5 * channel, 3, 1, 7, dilation=7)
        # self.conv5_3 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # self.conv5_4 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 2, dilation=2)
        # self.conv5_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 3, dilation=3)
        # self.conv5_6 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 5, dilation=5)
        # self.conv5_7 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 7, dilation=7)




        # self.conv5_5f = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 7, dilation=7)

        # self.conv1_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1, dilation=1)
        # self.conv2_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 2, dilation=2)
        # self.conv3_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 3, dilation=3)
        # self.conv4_5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 5, dilation=5)





    def forward(self, zt1, zt2, zt3, zt4, zt5):
        # zt5_1 = self.conv_upsample0(zt5)
        # zt4_1 = self.add11(self.add1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5))), zt4)
        # zt3_1 = self.add22(self.add2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4))), zt3)
        # zt2_1 = self.add33(self.add3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3))), zt2)
        # zt1_1 = self.add44(self.add4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2))), zt1)
        # zt5_1 = self.conv_upsample0(zt5)
        # zt4_1 = self.a1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5)), zt4)
        # zt3_1 = self.a2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4)), zt3)
        # zt2_1 = self.a3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3)), zt2)
        # zt1_1 = self.a4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2)), zt1)
        # zt1_1 = zt1



        # # zt2_1 = self.conv2_2(torch.cat((self.upsample(zt1_1), zt2), dim=1))
        # # zt2_2 = self.conv2_1(zt2_1)
        #
        # # zt2_2 = self.conv2_1(torch.cat((self.conv2_3(zt2_1), self.conv2_4(zt2_1)), dim=1))
        #
        # zt3_1 = self.conv3_2(torch.cat((self.upsample(zt2_1), zt3), dim=1))
        # # zt3_2 = self.conv3_1(zt3_1)
        #
        # zt3_2 = self.conv3_1(torch.cat((self.conv3_3(zt3_1), self.conv3_4(zt3_1), self.conv3_5(zt3_1)), dim=1))
        #
        #
        # zt4_1 = self.conv4_2(torch.cat((self.upsample(zt3_1), zt4), dim=1))
        # # zt4_2 = self.conv4_1(zt4_1)
        #
        # zt4_2 = self.conv4_1(torch.cat((self.conv4_3(zt4_1), self.conv4_4(zt4_1), self.conv4_5(zt4_1), self.conv4_6(zt4_1)), dim=1))
        #
        #
        # zt5_1 = self.conv5_2(torch.cat((self.upsample(zt4_1), zt5), dim=1))
        # # zt5_2 = self.conv5_1(zt5_1)
        #
        # zt5_2 = self.conv5_1(torch.cat((self.conv5_3(zt5_1), self.conv5_4(zt5_1), self.conv5_5(zt5_1), self.conv5_6(zt5_1), self.conv5_7(zt5_1)), dim=1))
        zt1_1 = self.conv1_1(torch.cat((zt1, self.conv0_1(self.downsample1(zt2)), self.conv0_2(self.downsample2(zt3)), self.conv0_3(self.downsample3(zt4)), self.conv0_4(self.downsample4(zt5))), dim=1))
        zt2_2 = self.conv2_1(torch.cat((self.conv0_5(self.upsample1(zt1_1)), zt2, self.conv0_6(self.downsample1(zt3)), self.conv0_7(self.downsample2(zt4)), self.conv0_8(self.downsample3(zt5))), dim=1))
        zt3_2 = self.conv3_1(torch.cat((self.conv0_9(self.upsample2(zt1_1)), self.conv0_10(self.upsample1(zt2_2)), zt3, self.conv0_11(self.downsample1(zt4)), self.conv0_12(self.downsample2(zt5))), dim=1))
        zt4_2 = self.conv4_1(torch.cat((self.conv0_13(self.upsample3(zt1_1)), self.conv0_14(self.upsample2(zt2_2)), self.conv0_15(self.upsample1(zt3_2)), zt4, self.conv0_16(self.downsample1(zt5))), dim=1))
        zt5_2 = self.conv5_1(torch.cat((self.conv0_17(self.upsample4(zt1_1)), self.conv0_18(self.upsample3(zt2_2)), self.conv0_19(self.upsample2(zt3_2)), self.conv0_20(self.upsample1(zt4_2)), zt5), dim=1))

        # zt1_1 = self.conv1_1(zt1 + self.conv0_1(self.downsample1(zt2)) + self.conv0_2(self.downsample2(zt3)) +
        #                                 self.conv0_3(self.downsample3(zt4)) + self.conv0_4(self.downsample4(zt5)))
        # zt2_2 = self.conv2_1(self.conv0_5(self.upsample1(zt1_1)) + zt2 + self.conv0_6(self.downsample1(zt3)) +
        #                                 self.conv0_7(self.downsample2(zt4)) + self.conv0_8(self.downsample3(zt5)))
        # zt3_2 = self.conv3_1(self.conv0_9(self.upsample2(zt1_1)) + self.conv0_10(self.upsample1(zt2_2)) + zt3 +
        #                                 self.conv0_11(self.downsample1(zt4)) + self.conv0_12(self.downsample2(zt5)))
        # zt4_2 = self.conv4_1(self.conv0_13(self.upsample3(zt1_1)) + self.conv0_14(self.upsample2(zt2_2)) +
        #                                 self.conv0_15(self.upsample1(zt3_2)) + zt4 +
        #                                 self.conv0_16(self.downsample1(zt5)))
        # zt5_2 = self.conv5_1(self.conv0_17(self.upsample4(zt1_1)) + self.conv0_18(self.upsample3(zt2_2)) +
        #                                 self.conv0_19(self.upsample2(zt3_2)) + self.conv0_20(self.upsample1(zt4_2)) +
        #                                 zt5)

        # zt1_1 = self.conv1_1(torch.cat((zt1, self.conv0_1(self.downsample1(zt2)), self.conv0_2(self.downsample2(zt3)),
        #                                 self.conv0_3(self.downsample3(zt4)), self.conv0_4(self.downsample4(zt5))),
        #                                dim=1))
        # zt2_2 = self.conv2_1(torch.cat((self.conv0_5(self.upsample1(zt1_1)), zt2, self.conv0_6(self.downsample1(zt3)),
        #                                 self.conv0_7(self.downsample2(zt4)), self.conv0_8(self.downsample3(zt5))),
        #                                dim=1))
        # zt3_2 = self.conv3_1(torch.cat((self.conv0_9(self.upsample2(zt1_1)), self.conv0_10(self.upsample1(zt2_2)), zt3,
        #                                 self.conv0_11(self.downsample1(zt4)), self.conv0_12(self.downsample2(zt5))),
        #                                dim=1))
        # zt4_2 = self.conv4_1(torch.cat((self.conv0_13(self.upsample3(zt1_1)), self.conv0_14(self.upsample2(zt2_2)),
        #                                 self.conv0_15(self.upsample1(zt3_2)), zt4,
        #                                 self.conv0_16(self.downsample1(zt5))), dim=1))
        # zt5_2 = self.conv5_1(torch.cat((self.conv0_17(self.upsample4(zt1_1)), self.conv0_18(self.upsample3(zt2_2)),
        #                                 self.conv0_19(self.upsample2(zt3_2)), self.conv0_20(self.upsample1(zt4_2)),
        #                                 zt5), dim=1))

        # zt1_4 = self.conv1_2(torch.cat((self.downsample(zt1_5), zt1_4), dim=1))
        # zt1_3 = self.conv1_3(torch.cat((self.downsample(zt1_4), zt1_3), dim=1))
        # zt1_2 = self.conv1_4(torch.cat((self.downsample(zt1_3), zt1_2), dim=1))
        # zt1_1 = self.conv1_5(torch.cat((self.downsample(zt1_2), zt1_1), dim=1))
        #
        # zt2_3 = self.conv2_3(torch.cat((self.downsample(zt2_4), zt2_3), dim=1))
        # zt2_2 = self.conv2_4(torch.cat((self.downsample(zt2_3), zt2_2), dim=1))
        # zt2_1 = self.conv2_5(torch.cat((self.downsample(zt2_2), zt2_1), dim=1))
        #
        # zt3_2 = self.conv3_4(torch.cat((self.downsample(zt3_3), zt3_2), dim=1))
        # zt3_1 = self.conv3_5(torch.cat((self.downsample(zt3_2), zt3_1), dim=1))
        #
        # zt4_1 = self.conv4_5(torch.cat((self.downsample(zt4_2), zt4_1), dim=1))


        return [zt1_1, zt2_2, zt3_2, zt4_2, zt5_2]

class NeighborConnectionDecoder400(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder400, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.AvgPool2d(2, 2, 0)

        self.conv_upsample1 = ConvBNReLU(1 * channel, channel, 3, 1, 1)
        self.conv_upsample2 = ConvBNReLU(1 * channel, channel, 3, 1, 1)
        self.conv_upsample3 = ConvBNReLU(1 * channel, channel, 3, 1, 1)
        self.conv_upsample4 = ConvBNReLU(1 * channel, channel, 3, 1, 1)
        self.conv_upsample5 = ConvBNReLU(1 * channel, channel, 3, 1, 1)

    def forward(self, zt5, zt4, zt3, zt2, zt1):

        zt1_1 = self.conv_upsample1(zt1)
        zt2_1 = self.conv_upsample2(zt2)
        zt3_1 = self.conv_upsample3(zt3)
        zt4_1 = self.conv_upsample4(zt4)
        zt5_1 = self.conv_upsample5(zt5)

        return [zt5_1, zt4_1, zt3_1, zt2_1, zt1_1]

class NeighborConnectionDecoder400m(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder400m, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.AvgPool2d(2, 2, 0)

        self.conv_upsample1 = MBSA('cuda:7', 1 * channel, channel)
        self.conv_upsample2 = MBSA('cuda:7', 1 * channel, channel)
        self.conv_upsample3 = MBSA('cuda:7', 1 * channel, channel)
        self.conv_upsample4 = MBSA('cuda:7', 1 * channel, channel)
        self.conv_upsample5 = MBSA('cuda:7', 1 * channel, channel)

    def forward(self, zt5, zt4, zt3, zt2, zt1):

        zt1_1 = self.conv_upsample1(zt1)
        zt2_1 = self.conv_upsample2(zt2)
        zt3_1 = self.conv_upsample3(zt3)
        zt4_1 = self.conv_upsample4(zt4)
        zt5_1 = self.conv_upsample5(zt5)

        return [zt5_1, zt4_1, zt3_1, zt2_1, zt1_1]


class NeighborConnectionDecoder40(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder40, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.AvgPool2d(2, 2, 0)


        self.conv_concat1 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1)
        self.conv_concat2 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1)
        self.conv_concat3 = ConvBNReLU(3 * channel, 3 * channel, 3, 1, 1)
        self.conv_concat4 = ConvBNReLU(4 * channel, 4 * channel, 3, 1, 1)
        self.conv_concat5 = ConvBNReLU(5 * channel, 5 * channel, 3, 1, 1)


        self.conv_concat7 = MMA(4 * channel, 4 * channel)
        self.conv_concat8 = MMA(3 * channel, 3 * channel)
        self.conv_concat9 = MMA(2 * channel, 2 * channel)
        self.conv_concat10 = MMA(1 * channel, 1 * channel)


        self.conv_upsample1 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        self.conv_upsample2 = ConvBNReLU(4 * channel, channel, 3, 1, 1)
        self.conv_upsample3 = ConvBNReLU(3 * channel, channel, 3, 1, 1)
        self.conv_upsample4 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        self.conv_upsample5 = ConvBNReLU(1 * channel, channel, 3, 1, 1)
        self.conv_upsample6 = ConvBNReLU(5 * channel, 4 * channel, 3, 1, 1)
        self.conv_upsample7 = ConvBNReLU(4 * channel, 3 * channel, 3, 1, 1)
        self.conv_upsample8 = ConvBNReLU(3 * channel, 2 * channel, 3, 1, 1)
        self.conv_upsample9 = ConvBNReLU(2 * channel, 1 * channel, 3, 1, 1)


    def forward(self, zt5, zt4, zt3, zt2, zt1, depth=None):
        '''
        zt1:(b,64,192,192)
        zt2:(b,64,96,96)
        zt3:(b,64,48,48)
        zt4:(b,64,24,24)
        zt5:(b,64,12,12)
        '''
        zt1_1 = self.conv_concat1(zt1) # CBR, (b,64,192,192)
        zt2_1 = self.conv_concat2(torch.cat((self.downsample(zt1_1), zt2), dim=1)) # (b,128,96,96)
        zt3_1 = self.conv_concat3(torch.cat((self.downsample(zt2_1), zt3), dim=1)) # (b,192,48,48)
        zt4_1 = self.conv_concat4(torch.cat((self.downsample(zt3_1), zt4), dim=1)) # (b,256,24,24)
        zt5_1 = self.conv_concat5(torch.cat((self.downsample(zt4_1), zt5), dim=1)) # (b,320,12,12)

        zt5_2 = self.conv_upsample1(zt5_1)

        zt4_1 = self.conv_concat7(zt4_1, self.conv_upsample6(self.upsample(zt5_1)))
        zt4_2 = self.conv_upsample2(zt4_1)

        zt3_1 = self.conv_concat8(zt3_1, self.conv_upsample7(self.upsample(zt4_1)))
        zt3_2 = self.conv_upsample3(zt3_1)

        zt2_1 = self.conv_concat9(zt2_1, self.conv_upsample8(self.upsample(zt3_1)))
        zt2_2 = self.conv_upsample4(zt2_1)

        zt1_1 = self.conv_concat10(zt1_1, self.conv_upsample9(self.upsample(zt2_1)))
        zt1_2 = self.conv_upsample5(zt1_1)

        '''
        zt5_2: (b,64,12,12)
        zt4_2: (b,64,24,24)
        zt3_2: (b,64,48,48)
        zt2_2: (b,64,96,96)
        zt1_2: (b,64,192,192)
        '''
        return [zt5_2, zt4_2, zt3_2, zt2_2, zt1_2]





class NeighborConnectionDecoder40_e(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder40_e, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.AvgPool2d(2, 2, 0)

        # self.conv_upsample1 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        # self.conv_upsample2 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        # self.conv_upsample3 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        # self.conv_upsample4 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        # self.conv_upsample5 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        # self.conv_upsample6 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1)
        # self.conv_upsample7 = ConvBNReLU(5 * channel, 2 * channel, 3, 1, 1)
        # self.conv_upsample8 = ConvBNReLU(5 * channel, 3 * channel, 3, 1, 1)
        # self.conv_upsample9 = ConvBNReLU(5 * channel, 4 * channel, 3, 1, 1)



        # self.conv_concat1 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1)
        # self.conv_concat2 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1)
        # self.conv_concat3 = ConvBNReLU(3 * channel, 3 * channel, 3, 1, 1)
        # self.conv_concat4 = ConvBNReLU(4 * channel, 4 * channel, 3, 1, 1)
        # self.conv_concat5 = ConvBNReLU(5 * channel, 5 * channel, 3, 1, 1)

        # self.back_layer_01d = nn.Sequential(ConvBNReLU(1, 32, 3, 1, 1), ConvBNReLU(32, 64, 3, 1, 1))
        # self.back_layer_02d = nn.Sequential(ConvBNReLU(1, 32, 3, 1, 1), ConvBNReLU(32, 64, 3, 1, 1))
        # self.back_layer_03d = nn.Sequential(ConvBNReLU(1, 32, 3, 1, 1), ConvBNReLU(32, 64, 3, 1, 1))
        # self.back_layer_04d = nn.Sequential(ConvBNReLU(1, 32, 3, 1, 1), ConvBNReLU(32, 64, 3, 1, 1))
        # self.back_layer_05d = nn.Sequential(ConvBNReLU(1, 32, 3, 1, 1), ConvBNReLU(32, 64, 3, 1, 1))
        # self.back_layer_06d = nn.Sequential(ConvBNReLU(1, 32, 3, 1, 1), ConvBNReLU(32, 64, 3, 1, 1))


        # self.conv_concat6 = MMA(5 * channel, 5 * channel)
        self.conv_concat7 = MMA2(1 * channel, 1 * channel)
        self.conv_concat8 = MMA2(1 * channel, 1 * channel)
        self.conv_concat9 = MMA2(1 * channel, 1 * channel)
        self.conv_concat10 = MMA2(1 * channel, 1 * channel)

        # self.conv_upsample0 = ConvBNReLU_BR(5 * channel, 5 * channel, 1, 1, 0)
        # self.conv_upsample1 = ConvBNReLU_BR(5 * channel, channel, 3, 1, 1)
        # self.conv_upsample2 = ConvBNReLU_BR(4 * channel, channel, 3, 1, 1)
        # self.conv_upsample3 = ConvBNReLU_BR(3 * channel, channel, 3, 1, 1)
        # self.conv_upsample4 = ConvBNReLU_BR(2 * channel, channel, 3, 1, 1)
        # self.conv_upsample5 = ConvBNReLU_BR(1 * channel, channel, 3, 1, 1)
        # self.conv_upsample6 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1)
        # self.conv_upsample7 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1)
        # self.conv_upsample8 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1)
        # self.conv_upsample9 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1)

        self.d1 = dASPP(64, 64)
        self.d2 = dASPP(64, 64)
        self.d3 = dASPP(64, 64)
        self.d4 = dASPP(64, 64)
        self.d5 = dASPP(64, 64)

        # self.conv4 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        # self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, zt1, zt2, zt3, zt4, zt5, depth=None):
        # zt5_1 = self.conv_upsample0(zt5)
        # zt4_1 = self.add11(self.add1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5))), zt4)
        # zt3_1 = self.add22(self.add2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4))), zt3)
        # zt2_1 = self.add33(self.add3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3))), zt2)
        # zt1_1 = self.add44(self.add4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2))), zt1)
        # zt5_1 = self.conv_upsample0(zt5)
        # zt4_1 = self.a1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5)), zt4)
        # zt3_1 = self.a2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4)), zt3)
        # zt2_1 = self.a3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3)), zt2)
        # zt1_1 = self.a4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2)), zt1)
        # zt1_1 = zt1
        # d1 = self.back_layer_01d(depth)
        zt1_1 = self.d1(zt1)
        zt2_1 = self.d2(zt2)
        zt3_1 = self.d3(zt3)
        zt4_1 = self.d4(zt4)
        zt5_1 = self.d5(zt5)
        # zt1_1 = self.conv_concat1(zt1)
        # zt2_1 = self.conv_concat2(torch.cat((self.downsample(zt1_1), zt2), dim=1))
        # zt3_1 = self.conv_concat3(torch.cat((self.downsample(zt2_1), zt3), dim=1))
        # zt4_1 = self.conv_concat4(torch.cat((self.downsample(zt3_1), zt4), dim=1))
        # zt5_1 = self.conv_concat5(torch.cat((self.downsample(zt4_1), zt5), dim=1))

        # zt5_1 = self.conv_upsample0(zt5) + e5_1
        # zt4_1 = self.conv_upsample00(self.upsample(zt5_1)) * self.conv_upsample1(self.upsample(zt5)) + zt4 + e4_1
        # zt3_1 = self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) + zt3 + e3_1
        # zt2_1 = self.conv_upsample4(self.upsample(zt3_1)) * self.conv_upsample5(self.upsample(zt3)) + zt2 + e2_1
        # zt1_1 = self.conv_upsample6(self.upsample(zt2_1)) * self.conv_upsample7(self.upsample(zt2)) + zt1 + e1_1

        # zt5_2 = self.conv_upsample1(zt5_1) + e5_1
        # zt5_1 = self.conv_concat6(zt5_1, self.conv_upsample0(zt5_1))
        zt1_2 = zt1_1
        # zt5_2 = self.conv_upsample1(torch.cat((e5_1, zt5_1), dim=1))

        # zt5_2 = self.conv_upsample1(zt5_1) * (1 + e5_1.sigmoid())
        # zt5_2 = self.add0(torch.cat((zt5_2, e5_1), dim=1))

        # zt4_1 = torch.cat((zt4_1, self.conv_upsample6(self.upsample(zt5_1))), 1)
        # zt2_2 = self.conv_concat7(zt2_1, self.conv_upsample6(self.upsample(zt1_1)))
        zt2_2 = self.conv_concat7(zt2_1, self.upsample(zt1_1))
        # zt4_1 = self.conv_concat6(torch.cat((zt4_1, self.conv_upsample6(self.upsample(zt5_1))), 1))
        # zt4_2 = self.conv_upsample2(zt4_1) + e4_2
        # zt2_2 = self.conv_upsample2(zt2_1)
        # zt4_2 = self.conv_concat6(zt4_1)
        # zt4_2 = self.conv_upsample2(torch.cat((e4_2, zt4_1), dim=1))

        # zt4_2 = self.conv_upsample2(zt4_1) * (1 + e4_2.sigmoid())
        # zt4_2 = self.add4(torch.cat((zt4_2, e4_2), dim=1))

        # zt3_1 = torch.cat((zt3_1, self.conv_upsample7(self.upsample(zt4_1))), 1)
        # zt3_2 = self.conv_concat8(zt3_1, self.conv_upsample7(self.upsample(zt2_1)))
        zt3_2 = self.conv_concat8(zt3_1, self.upsample(zt2_1))
        # zt3_1 = self.conv_concat7(torch.cat((zt3_1, self.conv_upsample7(self.upsample(zt4_1))), 1))
        # zt3_2 = self.conv_upsample3(zt3_1) + e3_2
        # zt3_2 = self.conv_upsample3(zt3_1)
        # zt3_2 = self.conv_concat7(zt3_1)
        # zt3_2 = self.conv_upsample3(torch.cat((e3_2, zt3_1), dim=1))
        # zt3_2 = self.conv_upsample3(zt3_1) * (1 + e3_2.sigmoid())
        # zt3_2 = self.add3(torch.cat((zt3_2, e3_2), dim=1))

        # zt2_1 = torch.cat((zt2_1, self.conv_upsample8(self.upsample(zt3_1))), 1)
        # zt4_2 = self.conv_concat9(zt4_1, self.conv_upsample8(self.upsample(zt3_1)))
        zt4_2 = self.conv_concat9(zt4_1, self.upsample(zt3_1))
        # zt2_1 = self.conv_concat8(torch.cat((zt2_1, self.conv_upsample8(self.upsample(zt3_1))), 1))
        # zt2_2 = self.conv_upsample4(zt2_1) + e2_2
        # zt2_2 = self.conv_upsample4(zt2_1)
        # zt2_2 = self.conv_concat8(zt2_1)
        # zt2_2 = self.conv_upsample4(torch.cat((e2_2, zt2_1), dim=1))
        # zt2_2 = self.conv_upsample4(zt2_1) * (1 + e2_2.sigmoid())
        # zt2_2 = self.add2(torch.cat((zt2_2, e2_2), dim=1))

        # zt1_1 = torch.cat((zt1_1, self.conv_upsample9(self.upsample(zt2_1))), 1)
        # zt5_2 = self.conv_concat10(zt5_1, self.conv_upsample9(self.upsample(zt4_1)))
        zt5_2 = self.conv_concat10(zt5_1, self.upsample(zt4_1))
        # zt1_1 = self.conv_concat9(torch.cat((zt1_1, self.conv_upsample9(self.upsample(zt2_1))), 1))
        # zt1_2 = self.conv_upsample5(zt1_1)
        # zt1_2 = self.conv_concat9(zt1_1)
        # zt1_2 = self.conv_upsample5(torch.cat((e1_1, zt1_1), dim=1))
        # print(zt1_2.mean((2, 3), keepdim=True))
        # zt1_2 = zt1_2 + e1_1
        # zt1_2 = self.conv_upsample5(zt1_1)
        # zt1_2 = self.conv_upsample5(zt1_1) * (1 + e1_1.sigmoid())
        # zt1_2 = self.add1(torch.cat((zt1_2, e1_1), dim=1))

        # pc = self.conv4(zt1_2)
        # pc = self.conv5(pc)
        # pc = zt1_2

        return [zt1_2, zt2_2, zt3_2, zt4_2, zt5_2]

class NeighborConnectionDecoder40_m(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder40_m, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.AvgPool2d(2, 2, 0)

        # self.conv_upsample1 = MBSA('cuda:1', 5 * channel, 1 * channel)
        # self.conv_upsample2 = MBSA('cuda:1', 5 * channel, 1 * channel)
        # self.conv_upsample3 = MBSA('cuda:1', 5 * channel, 1 * channel)
        # self.conv_upsample4 = MBSA('cuda:1', 5 * channel, 1 * channel)
        # self.conv_upsample5 = MBSA('cuda:1', 5 * channel, 1 * channel)

        self.conv_upsample1 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1)
        self.conv_upsample2 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1)
        self.conv_upsample3 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1)
        self.conv_upsample4 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1)
        self.conv_upsample5 = ConvBNReLU(5 * channel, 1 * channel, 3, 1, 1)

        self.conv_upsample6 = MBSA('cuda:1', 5 * channel, 1 * channel)
        self.conv_upsample7 = MBSA('cuda:1', 5 * channel, 2 * channel)
        self.conv_upsample8 = MBSA('cuda:1', 5 * channel, 3 * channel)
        self.conv_upsample9 = MBSA('cuda:1', 5 * channel, 4 * channel)

        # self.conv_concat1 = ConvBNReLU(1 * channel, 1 * channel, 3, 1, 1)
        self.conv_concat2 = MBSA('cuda:1', 2 * channel, 2 * channel)
        self.conv_concat3 = MBSA('cuda:1', 3 * channel, 3 * channel)
        self.conv_concat4 = MBSA('cuda:1', 4 * channel, 4 * channel)
        self.conv_concat5 = MBSA('cuda:1', 5 * channel, 5 * channel)
        # self.conv_concat2 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1)
        # self.conv_concat3 = ConvBNReLU(3 * channel, 3 * channel, 3, 1, 1)
        # self.conv_concat4 = ConvBNReLU(4 * channel, 4 * channel, 3, 1, 1)
        # self.conv_concat5 = ConvBNReLU(5 * channel, 5 * channel, 3, 1, 1)
        # self.relu5 = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU())
        # self.relu4 = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU())
        # self.relu3 = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU())
        # self.relu2 = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU())
        self.relu = nn.ReLU()

        # self.relu5 = nn.Sequential(nn.Sigmoid())
        # self.relu4 = nn.Sequential(nn.Sigmoid())
        # self.relu3 = nn.Sequential(nn.Sigmoid())
        # self.relu2 = nn.Sequential(nn.Sigmoid())
        # self.relu1 = nn.Sequential(nn.Sigmoid())

        # self.add4 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.add3 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.add2 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.add1 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.add0 = ConvBNReLU(2 * channel, channel, 3, 1, 1)

        # self.a5 = Attention2()
        # self.a4 = Attention2()
        # self.a3 = Attention2()
        # self.a2 = Attention2()
        # self.a1 = Attention2()

        # self.add44 = M_add(64)
        # self.add33 = M_add(64)
        # self.add22 = M_add(64)
        # self.add11 = M_add(64)
        # self.add00 = M_add(64)

        # self.add4 = M_add(64)
        # self.add3 = M_add(64)
        # self.add2 = M_add(64)
        # self.add1 = M_add(64)
        # self.add0 = M_add(64)

        # self.conv4 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        # self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, zt5, zt4, zt3, zt2, zt1):
        # zt5_1 = self.conv_upsample0(zt5)
        # zt4_1 = self.add11(self.add1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5))), zt4)
        # zt3_1 = self.add22(self.add2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4))), zt3)
        # zt2_1 = self.add33(self.add3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3))), zt2)
        # zt1_1 = self.add44(self.add4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2))), zt1)
        # zt5_1 = self.conv_upsample0(zt5)
        # zt4_1 = self.a1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5)), zt4)
        # zt3_1 = self.a2(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.upsample(zt4)), zt3)
        # zt2_1 = self.a3(self.conv_upsample4(self.upsample(zt3_1)), self.conv_upsample5(self.upsample(zt3)), zt2)
        # zt1_1 = self.a4(self.conv_upsample6(self.upsample(zt2_1)), self.conv_upsample7(self.upsample(zt2)), zt1)
        # zt1_1 = zt1
        zt1_1 = zt1
        zt2_1 = self.relu(self.conv_concat2(torch.cat((self.downsample(zt1_1), zt2), dim=1),
                                            torch.cat((self.downsample(zt1_1), zt2), dim=1)))
        zt3_1 = self.relu(self.conv_concat3(torch.cat((self.downsample(zt2_1), zt3), dim=1),
                                            torch.cat((self.downsample(zt2_1), zt3), dim=1)))
        zt4_1 = self.relu(self.conv_concat4(torch.cat((self.downsample(zt3_1), zt4), dim=1),
                                            torch.cat((self.downsample(zt3_1), zt4), dim=1)))
        zt5_1 = self.relu(self.conv_concat5(torch.cat((self.downsample(zt4_1), zt5), dim=1),
                                            torch.cat((self.downsample(zt4_1), zt5), dim=1)))

        # zt5_1 = self.conv_upsample0(zt5) + e5_1
        # zt4_1 = self.conv_upsample00(self.upsample(zt5_1)) * self.conv_upsample1(self.upsample(zt5)) + zt4 + e4_1
        # zt3_1 = self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) + zt3 + e3_1
        # zt2_1 = self.conv_upsample4(self.upsample(zt3_1)) * self.conv_upsample5(self.upsample(zt3)) + zt2 + e2_1
        # zt1_1 = self.conv_upsample6(self.upsample(zt2_1)) * self.conv_upsample7(self.upsample(zt2)) + zt1 + e1_1

        # zt5_2 = self.conv_upsample1(zt5_1) + e5_1
        zt5_2 = self.conv_upsample1(zt5_1)
        # zt5_2 = self.conv_upsample1(torch.cat((e5_1, zt5_1), dim=1))

        # zt5_2 = self.conv_upsample1(zt5_1) * (1 + e5_1.sigmoid())
        # zt5_2 = self.add0(torch.cat((zt5_2, e5_1), dim=1))

        zt4_1 = torch.cat((zt4_1, self.relu(self.conv_upsample6(self.upsample(zt5_1), self.upsample(zt5_1)))), 1)
        # zt4_2 = self.conv_upsample2(zt4_1) + e4_2
        zt4_2 = self.conv_upsample2(zt4_1)
        # zt4_2 = self.conv_upsample2(torch.cat((e4_2, zt4_1), dim=1))

        # zt4_2 = self.conv_upsample2(zt4_1) * (1 + e4_2.sigmoid())
        # zt4_2 = self.add4(torch.cat((zt4_2, e4_2), dim=1))

        zt3_1 = torch.cat((zt3_1, self.relu(self.conv_upsample7(self.upsample(zt4_1), self.upsample(zt4_1)))), 1)
        # zt3_2 = self.conv_upsample3(zt3_1) + e3_2
        zt3_2 = self.conv_upsample3(zt3_1)
        # zt3_2 = self.conv_upsample3(torch.cat((e3_2, zt3_1), dim=1))
        # zt3_2 = self.conv_upsample3(zt3_1) * (1 + e3_2.sigmoid())
        # zt3_2 = self.add3(torch.cat((zt3_2, e3_2), dim=1))

        zt2_1 = torch.cat((zt2_1, self.relu(self.conv_upsample8(self.upsample(zt3_1), self.upsample(zt3_1)))), 1)
        # zt2_2 = self.conv_upsample4(zt2_1) + e2_2
        zt2_2 = self.conv_upsample4(zt2_1)
        # zt2_2 = self.conv_upsample4(torch.cat((e2_2, zt2_1), dim=1))
        # zt2_2 = self.conv_upsample4(zt2_1) * (1 + e2_2.sigmoid())
        # zt2_2 = self.add2(torch.cat((zt2_2, e2_2), dim=1))

        zt1_1 = torch.cat((zt1_1, self.relu(self.conv_upsample9(self.upsample(zt2_1), self.upsample(zt2_1)))), 1)

        zt1_2 = self.conv_upsample5(zt1_1)
        # zt1_2 = self.conv_upsample5(torch.cat((e1_1, zt1_1), dim=1))
        # print(zt1_2.mean((2, 3), keepdim=True))
        # zt1_2 = zt1_2 + e1_1
        # zt1_2 = self.conv_upsample5(zt1_1)
        # zt1_2 = self.conv_upsample5(zt1_1) * (1 + e1_1.sigmoid())
        # zt1_2 = self.add1(torch.cat((zt1_2, e1_1), dim=1))

        # pc = self.conv4(zt1_2)
        # pc = self.conv5(pc)
        # pc = zt1_2

        return [zt5_2, zt4_2, zt3_2, zt2_2, zt1_2]


class NeighborConnectionDecoder1(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder1, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.AvgPool2d(2, 2, 0)
        # self.conv_upsample1 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample2 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample3 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample4 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample5 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample6 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample7 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample8 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample9 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample10 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample11 = ConvBNReLU(channel, channel, 3, 1, 1)

        self.conv_upsample0 = ConvBNReLU(channel, channel, 3, 1, 1)
        self.conv_upsample00 = ConvBNReLU(channel, channel, 3, 1, 1)
        self.conv_upsample000 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample1 = ConvBNReLU(channel, channel, 3, 1, 1)
        self.conv_upsample2 = ConvBNReLU(channel, channel, 3, 1, 1)
        self.conv_upsample3 = ConvBNReLU(channel, channel, 3, 1, 1)
        self.conv_upsample4 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample5 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample6 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample6 = MBConvBlock(channel, channel)
        # self.conv_upsample7 = MBConvBlock(channel, channel)
        # self.conv_upsample8 = ConvBNReLU(channel, channel, 3, 1, 1)
        self.conv_upsample9 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample10 = ConvBNReLU(channel, channel, 3, 1, 1)
        self.conv_upsample11 = ConvBNReLU(channel, channel, 3, 1, 1)
        self.conv_upsample12 = ConvBNReLU(channel, channel, 3, 1, 1)
        self.conv_upsample13 = ConvBNReLU(channel, channel, 3, 1, 1)

        # self.conv_upsample5 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1)

        # self.conv_concat2 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.conv_concat3 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.conv_concat4 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.conv_concat5 = ConvBNReLU(2 * channel, channel, 3, 1, 1)

        self.conv_concat2 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        self.conv_concat3 = ConvBNReLU(3 * channel, channel, 3, 1, 1)
        self.conv_concat4 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.d1 = dASPP(1 * channel, 1 * channel)
        # self.conv_concat5 = MBConvBlock(2 * channel, 1 * channel)

        # self.fuse = nn.Conv2d(1 * channel, 1, 1)

        # self.add5 = MMA(64, 64)
        # self.add4 = MMA(64, 64)
        # self.add3 = MMA(64, 64)
        # self.add2 = MMA(64, 64)
        # self.add1 = MMA(64, 64)
        # self.add31 = MMA(64, 64)
        # self.add32 = MMA(64, 64)

        # self.a5 = Attention2()
        # self.a4 = Attention2()
        # self.a3 = Attention2()
        # self.a2 = Attention2()
        # self.a1 = Attention2()

        # self.conv4 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        # self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, zt5, zt4, zt3, zt2, zt1):



        # zt5_1 = self.conv_upsample0(self.downsample(zt4)) * zt5
        # zt1_1 = self.conv_upsample00(self.upsample(zt2)) * zt1
        # zt5_1 = self.conv_upsample0(zt5)
        # zt1_1 = self.conv_upsample00(zt1)

        # zt4_1 = (self.conv_upsample000(self.upsample(zt5_1)) * self.conv_upsample1(self.downsample(zt3)) * zt4)
        # zt2_1 = (self.conv_upsample4(self.downsample(zt1_1)) * self.conv_upsample5(self.upsample(zt3)) * zt2)
        # zt3_1 = (self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.downsample(zt2_1)) * zt3)

        # zt4_1 = (self.conv_upsample000(self.upsample(zt5_1)) * self.conv_upsample1(self.upsample(zt5)) * zt4)
        # zt2_1 = (self.conv_upsample4(self.downsample(zt1_1)) * self.conv_upsample5(self.downsample(zt1)) * zt2)
        # zt3_1 = (self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.downsample(zt2_1)) * zt3)


        zt5_1 = self.conv_upsample0(zt5) * zt5
        zt1_1 = self.conv_upsample00(zt1) * zt1
        zt4_1 = (self.conv_upsample000(self.upsample(zt5_1)) * zt4)
        zt2_1 = (self.conv_upsample4(self.downsample(zt1_1)) * zt2)
        zt3_1 = (self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.downsample(zt2_1)) * zt3)

        # zt5_1 = self.add5(self.conv_upsample0(zt5), zt5)
        # zt1_1 = self.add1(self.conv_upsample00(zt1), zt1)
        # zt4_1 = self.add4(self.conv_upsample000(self.upsample(zt5_1)), zt4)
        # zt2_1 = self.add2(self.conv_upsample4(self.downsample(zt1_1)), zt2)
        #
        # zt3 = self.add31(self.conv_upsample2(self.upsample(zt4_1)), zt3)
        # zt3_1 = self.add3(self.conv_upsample3(self.downsample(zt2_1)), zt3)


        zt4_2 = torch.cat((zt4_1, self.conv_upsample9(self.upsample(zt5_1))), 1)
        zt4_2 = self.conv_concat2(zt4_2)

        zt2_2 = torch.cat((zt2_1, self.conv_upsample11(self.downsample(zt1_1))), 1)
        zt2_2 = self.conv_concat4(zt2_2)

        zt3_2 = torch.cat(
            (self.conv_upsample12(self.upsample(zt4_2)), zt3_1, self.conv_upsample13(self.downsample(zt2_2))), 1)
        zt3_2 = self.conv_concat3(zt3_2)



        # zt4_2 = torch.cat((self.conv_upsample8(self.downsample(zt3_1)), zt4_1, self.conv_upsample9(self.upsample(zt5_1))), 1)
        # zt4_2 = self.conv_concat2(zt4_2)
        #
        # zt2_2 = torch.cat((self.conv_upsample10(self.upsample(zt3_1)), zt2_1, self.conv_upsample11(self.downsample(zt1_1))), 1)
        # zt2_2 = self.conv_concat4(zt2_2)
        #
        # zt3_2 = torch.cat(
        #       (self.conv_upsample12(self.upsample(zt4_2)), zt3_1, self.conv_upsample13(self.downsample(zt2_2))), 1)
        # zt3_2 = self.conv_concat3(zt3_2)

        return zt3_2


class NeighborConnectionDecoder_m(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder_m, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.AvgPool2d(2, 2, 0)
        # self.conv_upsample1 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample2 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample3 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample4 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample5 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample6 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample7 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample8 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample9 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample10 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample11 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.m0 = MBSA('cuda:1', channel * 1, channel * 1)
        # self.m1 = MBSA('cuda:1', channel * 1, channel * 1)
        self.m2 = MBSA('cuda:1', channel * 1, channel * 1)
        self.m3 = MBSA('cuda:1', channel * 1, channel * 1)
        self.m4 = MBSA('cuda:1', channel * 1, channel * 1)
        # self.m5 = MBSA('cuda:1', channel * 1, channel * 1)
        # self.m4 = MBSA('cuda:1', self.channel * 1, self.channel * 1)
        # self.m5 = MBSA('cuda:1', self.channel * 1, self.channel * 1)
        # self.m6 = MBSA('cuda:1', self.channel * 1, self.channel * 1)
        # self.relu = nn.ReLU()

        # self.conv_upsample0 = MBSA('cuda:1', channel * 1, channel * 1)
        # self.conv_upsample00 = MBSA('cuda:1', channel * 1, channel * 1)
        self.conv_upsample000 = MBSA('cuda:1', channel * 1, channel * 1)
        # # self.conv_upsample1 = ConvBNReLU(channel, channel, 3, 1, 1)
        self.conv_upsample2 = MBSA('cuda:1', channel * 1, channel * 1)
        self.conv_upsample3 = MBSA('cuda:1', channel * 1, channel * 1)
        self.conv_upsample4 = MBSA('cuda:1', channel * 1, channel * 1)
        # # self.conv_upsample5 = ConvBNReLU(channel, channel, 3, 1, 1)
        # # self.conv_upsample6 = ConvBNReLU(channel, channel, 3, 1, 1)
        # # self.conv_upsample6 = MBConvBlock(channel, channel)
        # # self.conv_upsample7 = MBConvBlock(channel, channel)
        # # self.conv_upsample8 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample9 = MBSA('cuda:1', channel * 1, channel * 1)
        # # self.conv_upsample10 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample11 = MBSA('cuda:1', channel * 1, channel * 1)
        # self.conv_upsample12 = MBSA('cuda:1', channel * 1, channel * 1)
        # self.conv_upsample13 = MBSA('cuda:1', channel * 1, channel * 1)

        # # self.conv_upsample5 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1)
        #
        # # self.conv_concat2 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # # self.conv_concat3 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # # self.conv_concat4 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # # self.conv_concat5 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        self.conv_concat2 = MBSA('cuda:1', channel * 2, channel * 1)
        self.conv_concat3 = MBSA('cuda:1', channel * 3, channel * 1)
        self.conv_concat4 = MBSA('cuda:1', channel * 2, channel * 1)

        # self.conv_concat2 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.conv_concat3 = ConvBNReLU(3 * channel, channel, 3, 1, 1)
        # self.conv_concat4 = ConvBNReLU(2 * channel, channel, 3, 1, 1)

    def forward(self, zt5, zt4, zt3, zt2, zt1):
        zt5_1 = zt5
        zt1_1 = zt1
        # zt5_1 = self.conv_upsample0(self.downsample(zt4)) * zt5
        # zt1_1 = self.conv_upsample00(self.upsample(zt2)) * zt1
        # zt5_1 = self.conv_upsample0(zt5)
        # zt1_1 = self.conv_upsample00(zt1)

        # zt4_1 = (self.conv_upsample000(self.upsample(zt5_1)) * self.conv_upsample1(self.downsample(zt3)) * zt4)
        # zt2_1 = (self.conv_upsample4(self.downsample(zt1_1)) * self.conv_upsample5(self.upsample(zt3)) * zt2)
        # zt3_1 = (self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.downsample(zt2_1)) * zt3)

        # zt4_1 = (self.conv_upsample000(self.upsample(zt5_1)) * self.conv_upsample1(self.upsample(zt5)) * zt4)
        # zt2_1 = (self.conv_upsample4(self.downsample(zt1_1)) * self.conv_upsample5(self.downsample(zt1)) * zt2)
        # zt3_1 = (self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.downsample(zt2_1)) * zt3)

        zt4_1 = (self.conv_upsample000(self.upsample(zt5_1)) * zt4)
        zt2_1 = (self.conv_upsample4(self.downsample(zt1_1)) * zt2)
        zt3_1 = (self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.downsample(zt2_1)) * zt3)

        zt4_2 = torch.cat((zt4_1, self.upsample(zt5_1)), 1)
        zt4_2 = self.conv_concat2(zt4_2)

        zt2_2 = torch.cat((zt2_1, self.downsample(zt1_1)), 1)
        zt2_2 = self.conv_concat4(zt2_2)

        zt3_2 = torch.cat(
            (self.upsample(zt4_2), zt3_1, self.downsample(zt2_2)), 1)
        zt3_2 = self.conv_concat3(zt3_2)

        # zt4_2 = torch.cat((self.conv_upsample8(self.downsample(zt3_1)), zt4_1, self.conv_upsample9(self.upsample(zt5_1))), 1)
        # zt4_2 = self.conv_concat2(zt4_2)
        #
        # zt2_2 = torch.cat((self.conv_upsample10(self.upsample(zt3_1)), zt2_1, self.conv_upsample11(self.downsample(zt1_1))), 1)
        # zt2_2 = self.conv_concat4(zt2_2)
        #
        # zt3_2 = torch.cat(
        #       (self.conv_upsample12(self.upsample(zt4_2)), zt3_1, self.conv_upsample13(self.downsample(zt2_2))), 1)
        # zt3_2 = self.conv_concat3(zt3_2)

        return zt3_2


class NeighborConnectionDecoder11(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder11, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.AvgPool2d(2, 2, 0)
        # self.conv_upsample1 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample2 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample3 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample4 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample5 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample6 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample7 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample8 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample9 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample10 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample11 = ConvBNReLU(channel, channel, 3, 1, 1)

        self.conv_upsample0 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample00 = ConvBNReLU(channel, channel, 3, 1, 1)

        self.conv_upsample1 = ConvBNReLU(channel, channel, 3, 1, 1)
        self.conv_upsample2 = ConvBNReLU(channel, channel, 3, 1, 1)
        self.conv_upsample3 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample4 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample5 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample6 = ConvBNReLU(channel, channel, 3, 1, 1)
        # self.conv_upsample7 = ConvBNReLU(channel, channel, 3, 1, 1)

        self.conv_upsample8 = ConvBNReLU(channel, channel, 3, 1, 1)
        self.conv_upsample9 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1)
        # self.conv_upsample10 = ConvBNReLU(3 * channel, 3 * channel, 3, 1, 1)
        # self.conv_upsample11 = ConvBNReLU(4 * channel, 4 * channel, 3, 1, 1)
        # self.conv_upsample12 = ConvBNReLU(channel, channel, 3, 1, 1)

        # self.conv_upsample5 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1)

        # self.conv_concat2 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.conv_concat3 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.conv_concat4 = ConvBNReLU(2 * channel, channel, 3, 1, 1)
        # self.conv_concat5 = ConvBNReLU(2 * channel, channel, 3, 1, 1)

        self.conv_concat1 = ConvBNReLU(2 * channel, 2 * channel, 3, 1, 1)
        self.conv_concat2 = ConvBNReLU(3 * channel, 3 * channel, 3, 1, 1)
        # self.conv_concat3 = ConvBNReLU(4 * channel, 4 * channel, 3, 1, 1)
        # self.conv_concat4 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        # self.d1 = dASPP(1 * channel, 1 * channel)
        # self.conv_concat5 = MBConvBlock(2 * channel, 1 * channel)

        self.fuse = ConvBNReLU(3 * channel, channel, 3, 1, 1)

        # self.add4 = M_add(64)
        # self.add3 = M_add(64)
        # self.add2 = M_add(64)
        # self.add1 = M_add(64)
        # self.add0 = M_add(64)

        # self.a5 = Attention2()
        # self.a4 = Attention2()
        # self.a3 = Attention2()
        # self.a2 = Attention2()
        # self.a1 = Attention2()

        # self.conv4 = ConvBNReLU(5 * channel, channel, 3, 1, 1)
        # self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, zt5, zt4, zt3, zt2, zt1):
        # zt5_1 = self.conv_upsample00(zt5) * zt5
        zt5_1 = zt5
        # zt1_1 = self.conv_upsample00(zt1)

        # zt4_1 = self.add0((self.conv_upsample00(self.upsample(zt5_1)) + self.conv_upsample1(self.upsample(zt5))), zt4)
        # zt2_1 = self.add1((self.conv_upsample4(self.downsample(zt1_1)) + self.conv_upsample5(self.downsample(zt1))), zt2)
        # zt3_1 = self.add0((self.conv_upsample2(self.upsample(zt4_1)) + self.conv_upsample3(self.downsample(zt2_1))), zt3)

        # zt4_1 = self.a1(self.conv_upsample00(self.upsample(zt5_1)), self.conv_upsample1(self.upsample(zt5)), zt4)
        # zt2_1 = self.a2(self.conv_upsample4(self.downsample(zt1_1)), self.conv_upsample5(self.downsample(zt1)), zt2)
        # zt3_1 = self.a3(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.downsample(zt2_1)), zt3)

        zt4_1 = (self.conv_upsample0(self.upsample(zt5_1)) * zt4)
        zt3_1 = (self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) * zt3)
        # zt2_1 = (self.conv_upsample4(self.upsample(zt3_1)) * self.conv_upsample5(self.upsample(zt3)) * zt2)
        # zt1_1 = (self.conv_upsample6(self.upsample(zt2_1)) * self.conv_upsample7(self.upsample(zt2)) * zt1)
        # zt3_1 = self.a1(self.conv_upsample2(self.upsample(zt4_1)), self.conv_upsample3(self.downsample(zt2_1)), zt3)

        # zt5_1_1 = self.conv_upsample00(self.upsample(zt5_1))
        # zt5_1_2 = self.conv_upsample1(self.upsample(zt5))
        # zt4_1 = (zt5_1_1 * zt5_1_2) + (zt4 * zt5_1_1) + (zt5_1_2 * zt4)
        #
        # zt1_1_1 = self.conv_upsample4(self.downsample(zt1_1))
        # zt1_1_2 = self.conv_upsample5(self.downsample(zt1))
        # zt2_1 = (zt1_1_1 * zt1_1_2) + (zt2 * zt1_1_1) + (zt1_1_2 * zt2)
        #
        # zt4_1_1 = self.conv_upsample2(self.upsample(zt4_1))
        # zt2_1_1 = self.conv_upsample3(self.downsample(zt2_1))
        # zt3_1 = (zt4_1_1 * zt2_1_1) + (zt3 * zt4_1_1) + (zt2_1_1 * zt3)

        zt4_2 = torch.cat((zt4_1, self.conv_upsample8(self.upsample(zt5_1))), 1)
        zt4_2 = self.conv_concat1(zt4_2)

        zt3_2 = torch.cat((zt3_1, self.conv_upsample9(self.upsample(zt4_2))), 1)
        zt3_2 = self.conv_concat2(zt3_2)
        zt3_2 = self.fuse(zt3_2)
        # zt2_2 = torch.cat((zt2_1, self.conv_upsample10(self.upsample(zt3_2))), 1)
        # zt2_2 = self.conv_concat3(zt2_2)

        # zt1_2 = torch.cat((zt1_1, self.conv_upsample11(self.upsample(zt2_2))), 1)
        # zt1_2 = self.conv_concat4(zt1_2)
        # zt3_2 = self.d1(zt3_2)
        # zt3_2 = self.a2(self.conv_upsample9(self.upsample(zt4_2)), self.conv_upsample11(self.downsample(zt2_2)), zt3_1)
        # return self.fuse(zt3_2)

        return zt3_2


def get_coef(iter_percentage, method):
    if method == "linear":
        milestones = (0.3, 0.7)
        coef_range = (0, 1)
        min_point, max_point = min(milestones), max(milestones)
        min_dimoef, max_coef = min(coef_range), max(coef_range)
        if iter_percentage < min_point:
            ual_coef = min_dimoef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            ratio = (max_coef - min_dimoef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
    elif method == "cos":
        coef_range = (0, 1)
        min_dimoef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
        ual_coef = normalized_coef * (max_coef - min_dimoef) + min_dimoef
    else:
        ual_coef = 1.0
    return ual_coef


def sdice_loss(predict, target):
    smooth = 1
    p = 2
    # valid_mask = torch.ones_like(target)
    # predict2 = predict
    # predict = torch.sigmoid(predict)
    # print("d\n", target.shape)

    predict = torch.sigmoid(predict)
    # print("d\n", target.shape)
    weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    # predict2 = predict2.contiguous().view(predict2.shape[0], -1)
    # print(weit)

    # valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    # num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    # den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth

    weit = weit.contiguous().view(weit.shape[0], -1)
    # weit = 1 + 5 * torch.abs(F.avg_pool1d(target, kernel_size=31, stride=1, padding=15) - target)
    num = torch.sum(torch.mul(predict, target) * weit, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * weit, dim=1) + smooth

    loss = 1 - num / den

    # num = torch.sum(torch.mul(predict2, target) * weit, dim=1) * 2 + smooth
    # den = torch.sum((predict2.pow(p) + target.pow(p)) * weit, dim=1) + smooth
    #
    # loss2 = 1 - num / den
    # loss2 = loss2.mean()

    # print(str(loss2) + " " + str(loss.mean()))
    return loss.mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    # predict2 = predict
    # predict = torch.sigmoid(predict)
    # print("d\n", target.shape)

    predict = torch.sigmoid(predict)
    # print("d\n", target.shape)
    # weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    # predict2 = predict2.contiguous().view(predict2.shape[0], -1)
    # print(weit)

    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth

    # weit = weit.contiguous().view(weit.shape[0], -1)
    # num = torch.sum(torch.mul(predict, target) * weit, dim=1) * 2 + smooth
    # den = torch.sum((predict.pow(p) + target.pow(p)) * weit, dim=1) + smooth

    loss = 1 - num / den

    # num = torch.sum(torch.mul(predict2, target) * weit, dim=1) * 2 + smooth
    # den = torch.sum((predict2.pow(p) + target.pow(p)) * weit, dim=1) + smooth
    #
    # loss2 = 1 - num / den
    # loss2 = loss2.mean()

    # print(str(loss2) + " " + str(loss.mean()))
    return loss.mean()


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2,
        reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()


def structure_loss(pred, mask, iter_percentage, method):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # print("s\n", mask.shape)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    # wdice = dice_loss(pred, mask)

    # ual = 1 - (2 * pred - 1).abs().pow(2)
    # ual_coef = get_coef(iter_percentage, method)
    # ual *= ual_coef
    # wual = (weit * ual).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    return (wbce + wiou).mean()


@MODELS.register()
class ZoomNet(BasicModelClass):
    def __init__(self):
        super().__init__()
        # self.shared_encoder = timm.create_model(
        #     model_name="res2net50_26w_4s", pretrained=True, in_chans=3, features_only=True
        # )
        # self.shared_encoder = timm.create_model(
        #     model_name="resnet50", pretrained=True, in_chans=3, features_only=True
        # )
        # self.shared_encoder = timm.create_model(
        #     model_name="convnext_tiny", pretrained=True, in_chans=3, features_only=True
        # )
        # self.shared_encoder = timm.create_model(
        #     model_name="convnext_small", pretrained=True, in_chans=3, features_only=True
        # )

        ''' pvt_v2_b4 '''
        state_dict = torch.load("/data1/wuhu/ckpts/pvt_v2_b4/pvt_v2_b4.pth")
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        self.shared_encoder = pvtv2_encoder.pvt_v2_b4()
        self.shared_encoder.load_state_dict(state_dict)

        ''' pvt_v2_b2 '''
        # state_dict = torch.load("/data1/wuhu/ckpts/pvt_v2_b2/pvt_v2_b2.pth")
        # state_dict.pop('head.weight')
        # state_dict.pop('head.bias')
        # self.shared_encoder = pvtv2_encoder.pvt_v2_b2()
        # self.shared_encoder.load_state_dict(state_dict)

        ''' pvt_v1_large '''
        # state_dict = torch.load("/data1/wuhu/ckpts/pvt_v1_large/pvt_large.pth")
        # self.shared_encoder = pvt.pvt_large()
        # self.shared_encoder.load_state_dict(state_dict)

        ''' pvt_v1_small '''
        # state_dict = torch.load("/data1/wuhu/ckpts/pvt_v1_small/pvt_small.pth")
        # self.shared_encoder = pvt.pvt_small()
        # self.shared_encoder.load_state_dict(state_dict)

        ''' swin-base-384-in22K '''
        # self.shared_encoder = SwinModel.from_pretrained(
        #     "/data1/wuhu/ckpts/swin-base-384-in22K/",
        #     output_hidden_states=True,
        # )

        ''' ResNet50 '''
        # self.shared_encoder = ResNetModel.from_pretrained(
        #     "/data1/wuhu/ckpts/resnet50",
        #     output_hidden_states=True,
        # )



        self.translayer = TransLayer(out_c=64)  # [c5, c4, c3, c2, c1]

        self.d5 = dASPP(64, 64)
        self.d4 = dASPP(64, 64)
        self.d3 = dASPP(64, 64)
        self.d2 = dASPP(64, 64)
        self.d1 = dASPP(64, 64)
        # self.d0 = dASPP(64, 64)
        # self.m0 = MBSA('cuda:1', 64, 64)
        # self.mc = ConvBNReLU(64, 64, 3, 1, 1)

        self.ncd1 = NeighborConnectionDecoder40_e(64)
        # self.ncd1 = NeighborConnectionDecoder_m(64)
        self.ncd2 = NeighborConnectionDecoder40(64)
        # self.ncd2 = NeighborConnectionDecoder40_d(64)
        # self.ncd2 = NeighborConnectionDecoder403(64)
        # self.ncd2 = NeighborConnectionDecoder4u(64)
        # self.ncd401 = NeighborConnectionDecoder401(64)

        # self.br1 = NeighborConnectionDecoder402(64)
        # self.br2 = NeighborConnectionDecoder402(64)

        self.a5 = Focus1(64, 64)
        self.a4 = Focus1(64, 64)
        self.a3 = Focus1(64, 64)
        self.a2 = Focus1(64, 64)
        self.a1 = Focus1(64, 64)
        self.a0 = Focus1(64, 64)

        # self.m6 = MBSA('cuda:7', 2 * 64, 64)
        # self.m5 = MBSA('cuda:7', 3 * 64, 64)
        # self.m4 = MBSA('cuda:7', 3 * 64, 64)
        # self.m3 = MBSA('cuda:7', 3 * 64, 64)
        # self.m2 = MBSA('cuda:7', 3 * 64, 64)
        # self.m1 = MBSA('cuda:3', 2 * 64, 64)

        # self.mm6 = MBSA('cuda:2', 64, 64)
        # self.mm5 = MBSA('cuda:2', 64, 64)
        # self.mm4 = MBSA('cuda:2', 64, 64)
        # self.mm3 = MBSA('cuda:2', 64, 64)
        # self.mm2 = MBSA('cuda:2', 64, 64)
        # self.mm1 = MBSA('cuda:2', 2, 1)

        # self.m6 = ConvBNReLU(128, 64, 3, 1, 1)
        # self.m5 = ConvBNReLU(128, 64, 3, 1, 1)
        # self.m4 = ConvBNReLU(128, 64, 3, 1, 1)
        # self.m3 = ConvBNReLU(128, 64, 3, 1, 1)
        # self.m2 = ConvBNReLU(2 * 64, 64, 3, 1, 1)
        # self.m1 = ConvBNReLU(128, 64, 3, 1, 1)

        # self.m6 = Attention1(64)
        # self.m5 = Attention1(64)
        # self.m4 = Attention1(64)
        # self.m3 = Attention1(64)
        # self.m2 = Attention1(64)
        # self.m1 = Attention1(64)


        # self.add5 = ConvBNReLU(128, 64, 3, 1, 1)
        self.add4 = ConvBNReLU(128, 64, 3, 1, 1)
        self.add3 = ConvBNReLU(128, 64, 3, 1, 1)
        self.add2 = ConvBNReLU(128, 64, 3, 1, 1)
        self.add1 = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add0 = ConvBNReLU(64, 64, 3, 1, 1)

        self.cat5 = ConvBNReLU(128, 64, 3, 1, 1)
        self.cat4 = ConvBNReLU(128, 64, 3, 1, 1)
        self.cat3 = ConvBNReLU(128, 64, 3, 1, 1)
        self.cat2 = ConvBNReLU(128, 64, 3, 1, 1)
        self.cat1 = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add0 = ConvBNReLU(64, 64, 3, 1, 1)

        # self.add5 = ConvBNReLU(64, 64, 3, 1, 1)
        # self.add4 = ConvBNReLU(64, 64, 3, 1, 1)
        # self.add3 = ConvBNReLU(64, 64, 3, 1, 1)
        # self.add2 = ConvBNReLU(64, 64, 3, 1, 1)
        # self.add1 = ConvBNReLU(64, 64, 3, 1, 1)
        # self.add0 = ConvBNReLU(64, 64, 3, 1, 1)

        # self.add5 = ConvBNReLU_BR(64, 64, 3, 1, 1)
        # self.add4 = ConvBNReLU_BR(64, 64, 3, 1, 1)
        # self.add3 = ConvBNReLU_BR(64, 64, 3, 1, 1)
        # self.add2 = ConvBNReLU_BR(64, 64, 3, 1, 1)
        # self.add1 = ConvBNReLU_BR(64, 64, 3, 1, 1)
        # self.add0 = ConvBNReLU_BR(64, 64, 3, 1, 1)

        # self.add5 = ConvBNReLU_BR(128, 64, 3, 1, 1)
        # self.add4 = ConvBNReLU_BR(128, 64, 3, 1, 1)
        # self.add3 = ConvBNReLU_BR(128, 64, 3, 1, 1)
        # self.add2 = ConvBNReLU_BR(128, 64, 3, 1, 1)
        # self.add1 = ConvBNReLU_BR(128, 64, 3, 1, 1)
        # self.add0 = ConvBNReLU_BR(64, 64, 3, 1, 1)

        # self.add6d = ConvBNReLU(64, 64, 3, 1, 1)
        # self.add5d = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add4d = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add3d = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add2d = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add1d = ConvBNReLU(128, 64, 3, 1, 1)

        # self.add6db = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add5db = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add4db = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add3db = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add2db = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add1db = ConvBNReLU(128, 64, 3, 1, 1)

        # self.add6b = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add5b = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add4b = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add3b = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add2b = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add1b = ConvBNReLU(64, 64, 3, 1, 1)
        #
        # self.add6f = ConvBNReLU(64, 64, 3, 1, 1)
        # self.add5f = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add4f = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add3f = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add2f = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add1f = ConvBNReLU(64, 64, 3, 1, 1)
        # #
        # self.add6f0 = ConvBNReLU(64, 64, 3, 1, 1)
        # self.add5f0 = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add4f0 = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add3f0 = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add2f0 = ConvBNReLU(128, 64, 3, 1, 1)
        # # # self.add1f0 = ConvBNReLU(64, 64, 3, 1, 1)
        # #
        # self.add6f1 = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add5f1 = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add4f1 = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add3f1 = ConvBNReLU(128, 64, 3, 1, 1)
        # self.add2f1 = ConvBNReLU(128, 64, 3, 1, 1)
        #
        #
        #
        # self.out_layer_01b = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # self.out_layer_02b = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # self.out_layer_03b = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # self.out_layer_04b = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # self.out_layer_05b = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # self.out_layer_06b = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        #
        # self.out_layer_01f = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # # self.out_layer_02f = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # # self.out_layer_03f = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # # self.out_layer_04f = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # # self.out_layer_05f = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # # self.out_layer_06f = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        #
        # self.out_layer_01f0 = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # self.out_layer_02d = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # self.out_layer_03d = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # self.out_layer_04d = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # self.out_layer_05d = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        # self.out_layer_06d = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))

        # self.back_layer_02d = nn.Sequential(ConvBNReLU(1, 32, 3, 1, 1), ConvBNReLU(32, 64, 3, 1, 1))
        # self.back_layer_03d = nn.Sequential(ConvBNReLU(1, 32, 3, 1, 1), ConvBNReLU(32, 64, 3, 1, 1))
        # self.back_layer_04d = nn.Sequential(ConvBNReLU(1, 32, 3, 1, 1), ConvBNReLU(32, 64, 3, 1, 1))
        # self.back_layer_05d = nn.Sequential(ConvBNReLU(1, 32, 3, 1, 1), ConvBNReLU(32, 64, 3, 1, 1))
        # self.back_layer_06d = nn.Sequential(ConvBNReLU(1, 32, 3, 1, 1), ConvBNReLU(32, 64, 3, 1, 1))

        # self.in_layer_01 = nn.Sequential(ConvBNReLU(1, 3, 3, 1, 1))
        # self.in_layer_02 = nn.Sequential(ConvBNReLU(6, 3, 3, 1, 1))

        self.out_layer_01e = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        self.out_layer_02e = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        self.out_layer_03e = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        self.out_layer_04e = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        self.out_layer_05e = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        self.out_layer_06e = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))

        self.out_layer_01 = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        self.out_layer_02 = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        self.out_layer_03 = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        self.out_layer_04 = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        self.out_layer_05 = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        self.out_layer_06 = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))

        # self.bout_layer_00 = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))





        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')

    def encoder_translayer(self, x):
        en_feats = self.shared_encoder(x) # pvt、pvt_v2
        # en_feats = list(self.shared_encoder(x)['reshaped_hidden_states'][:4]) # swin-base
        # en_feats = list(self.shared_encoder(x)['hidden_states'][1:]) # ResNet


        # en_feats1 = self.shared_encoder1(x)
        # en_feats2 = self.shared_encoder2(x)
        '''
        en_feats: list_4=[
            (b,64,96,96),
            (b,128,48,48),
            (b,320,24,24),
            (b,512,12,12),
        ]
        '''

        a1 = self.up1(en_feats[0]) # (b,64,192,192)
        en_feats.insert(0, a1)
        '''
        en_feats: list_5=[
            (b,64,192,192),
            (b,64,96,96),
            (b,128,48,48),
            (b,320,24,24),
            (b,512,12,12),
        ]
        '''

        # a2 = self.up1(en_feats1[0])
        # en_feats1.insert(0, a2)

        # print(type(en_feats))
        # en_feats = reversed(list(en_feats))
        # for i in range(len(en_feats)):
        #     en_feats[i] = torch.cat([en_feats[i], en_feats1[i]], dim=1)

        # for i in range(len(en_feats)):
        #     print(en_feats[i].shape)
        # trans_feats = en_feats
        trans_feats = self.translayer(en_feats)
        '''
        trans_feats: tuple_5=[
            (b,64,12,12),
            (b,64,24,24),
            (b,64,48,48),
            (b,64,96,96),
            (b,64,192,192),
        ]
        '''
        # trans_feats1 = self.translayer1(en_feats1)
        # return trans_feats, trans_feats1
        return trans_feats
        # return en_feats

    def body(self, m_scale, depth=None):
        # m_trans_feats, m_trans_feats1 = self.encoder_translayer(m_scale)
        # m_scale = self.in_layer_02(torch.cat((self.in_layer_01(depth), m_scale), dim=1))
        m_trans_feats = self.encoder_translayer(m_scale)
        m_trans_feats = list(m_trans_feats) # Encoder中，经过 5个独立 CBR统一通道为 64维之后的输出


        edges = self.ncd1(m_trans_feats[0], m_trans_feats[1], m_trans_feats[2], m_trans_feats[3], m_trans_feats[4]) # BSM+MDASPP的输出

        e6 = self.out_layer_06e(edges[0]) # Boundary Decoder中的 CBR，生成的是边界mask
        e5 = self.out_layer_05e(edges[1])
        e4 = self.out_layer_04e(edges[2])
        e3 = self.out_layer_03e(edges[3])
        e2 = self.out_layer_02e(edges[4])

        # import matplotlib.pyplot as plt
        # plt.imshow(e6[0, 0, :, :].detach().cpu().numpy())
        # plt.show()

        # 指定保存路径
        # img = e2[0, 0, :, :].detach().cpu().numpy()
        # import os
        # save_dir = "/data1/wuhu/Projects/CAMO_xiao/tmp/"
        # os.makedirs(save_dir, exist_ok=True) # 如果文件夹不存在就创建
        # save_path = os.path.join(save_dir, "image.png")
        # # 保存图像
        # plt.imsave(save_path, img, cmap="gray") # cmap="gray" 如果是灰度图
        # print(f"图片已保存到 {save_path}")
        e60 = edges[0] # 论文中，Boundary Decoder的 MDASPP的输出
        e50 = edges[1]
        e40 = edges[2]
        e30 = edges[3]
        e20 = edges[4]

        '''
        zt5_2: (b,64,12,12)
        zt4_2: (b,64,24,24)
        zt3_2: (b,64,48,48)
        zt2_2: (b,64,96,96)
        zt1_2: (b,64,192,192)
        '''
        feats = self.ncd2(m_trans_feats[0], m_trans_feats[1], m_trans_feats[2], m_trans_feats[3], m_trans_feats[4]) # MBBFB+MDASPP
        # feats = m_trans_feats # 去掉 MBBFB的消融

        # feats[0] = self.d5(feats[0]) # MDASPP?
        # feats[1] = self.d4(feats[1])
        # feats[2] = self.d3(feats[2])
        # feats[3] = self.d2(feats[3])
        # feats[4] = self.d1(feats[4])

        feats[0] = self.cat5(torch.cat((feats[0], edges[0]), dim=1))
        feats[1] = self.cat4(torch.cat((feats[1], edges[1]), dim=1))
        feats[2] = self.cat3(torch.cat((feats[2], edges[2]), dim=1))
        feats[3] = self.cat2(torch.cat((feats[3], edges[3]), dim=1))
        feats[4] = self.cat1(torch.cat((feats[4], edges[4]), dim=1))


        x6 = self.out_layer_06(feats[0]) # CBR, 输出是单通道的 (b,1,12,12), 这一步是使用MBBFB下方最右侧的MDASPP的输出，预测一个二值分割掩码

        x = self.a5(feats[0], x6.sigmoid(), e6.sigmoid(), e60) # BADA

        x = cus_sample(x, mode="scale", factors=2) # (b,64,24,24)
        x50 = cus_sample(x6, mode="scale", factors=2) # (b,1,24,24)


        x = self.add4(torch.cat((x, self.a4(feats[1], x50.sigmoid(), e5.sigmoid(), e50)), 1))
        x5 = self.out_layer_05(x)

        x = cus_sample(x, mode="scale", factors=2)

        x40 = cus_sample(x5, mode="scale", factors=2)


        x = self.add3(torch.cat((x, self.a3(feats[2], x40.sigmoid(), e4.sigmoid(), e40)), 1))

        x4 = self.out_layer_04(x)

        x = cus_sample(x, mode="scale", factors=2)

        x30 = cus_sample(x4, mode="scale", factors=2)


        x = self.add2(torch.cat((x, self.a2(feats[3], x30.sigmoid(), e3.sigmoid(), e30)), 1))

        x3 = self.out_layer_03(x)

        x = cus_sample(x, mode="scale", factors=2)
        x20 = cus_sample(x3, mode="scale", factors=2)

        x = self.add1(torch.cat((x, self.a1(feats[4], x20.sigmoid(), e2.sigmoid(), e20)), 1))
        x2 = self.out_layer_02(x)


        x = cus_sample(x, mode="scale", factors=2)
        e1 = cus_sample(e2, mode="scale", factors=2)
        x10 = cus_sample(x2, mode="scale", factors=2)

        x = self.a0(x, x10.sigmoid(), e1.sigmoid())
        x1 = self.out_layer_01(x)

        return dict(seg=x1, seg2=x2, seg3=x3, seg4=x4, seg5=x5, seg6=x6), dict(seg2=e2, seg3=e3, seg4=e4, seg5=e5, seg6=e6)


    def train_forward(self, data, **kwargs):
        # assert not {"image1.5", "image1.0", "image0.5", "mask"}.difference(set(data)), set(data)

        output, output2 = self.body(
            m_scale=data["image1.0"],
            # depth=data["depth"],
        )
        # output = self.body(
        #     m_scale=data["image1.0"],
        # )
        loss, loss_str = self.cal_loss(
            all_preds=output,
            all_preds2=output2,
            # all_preds3=output3,
            gts=data["mask"],
            # gts2=data["depth"],
            gts2=data["edge"],
            # gts3=1 - data["mask"],
            iter_percentage=kwargs["curr"]["iter_percentage"],
        )
        # return dict(sal=(1 - output2["seg"].sigmoid())), loss, loss_str
        return dict(sal=output["seg"].sigmoid()), loss, loss_str
        # return dict(sal=output["seg"].sigmoid(), sed=output2["edge"].sigmoid()), loss, loss_str

    def test_forward(self, data, **kwargs):
        output, output2 = self.body(
            m_scale=data["image1.0"],
            # depth=data["depth"],
            # m_scale=data,
        )
        # output = self.body(
        #     m_scale=data["image1.0"],
        # )
        # return [output["seg"], output["seg2"], output["seg3"], output["seg4"], output["seg5"], output["seg6"]], output2["edge"]
        # return 1 - output2["seg"].sigmoid()
        return output["seg"].sigmoid()

    def cal_loss(self, all_preds: dict, gts: torch.Tensor, all_preds2: dict = None, gts2: torch.Tensor = None, all_preds3: dict = None, gts3: torch.Tensor = None,
                 method="cos",
                 iter_percentage: float = 0):
        # ual_coef = get_coef(iter_percentage, method)

        losses = []
        loss_str = []
        # for main


        for name, preds in all_preds.items():
            # resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:], interpolation="bilinear")
            preds = cus_sample(preds, mode="size", factors=gts.shape[2:], interpolation="bilinear")
            # resized_gts = gts
            # print(gts.max())
            # sod_loss = structure_loss(pred=preds, mask=resized_gts, iter_percentage=iter_percentage, method=method)
            sod_loss = structure_loss(pred=preds, mask=gts, iter_percentage=iter_percentage, method=method)

            # sod_loss = F.binary_cross_entropy(input=preds, target=resized_gts, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"{name}_IBCE: {sod_loss.item():.5f}")

            # ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
            # ual_loss *= ual_coef
            # losses.append(ual_loss)
            # loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")

            # iou_loss = IoULoss(preds, resized_gts)
            # iou_loss *= (1 - ual_coef)
            # losses.append(iou_loss)
            # loss_str.append(f"{name}_UIOU_{(1 - ual_coef):.5f}: {iou_loss.item():.5f}")
        # for name, preds in all_preds3.items():
        #     resized_gts = cus_sample(gts3, mode="size", factors=preds.shape[2:], interpolation="bilinear")
        #     # resized_gts = gts
        #
        #     # edge_loss = structure_loss(pred=preds, mask=resized_gts, iter_percentage=iter_percentage, method=method)
        #     # preds = preds.sigmoid()
        #
        #     back_loss = structure_loss(pred=preds, mask=resized_gts, iter_percentage=iter_percentage, method=method)
        #
        #     # sod_loss = F.binary_cross_entropy(input=preds, target=resized_gts, reduction="mean")
        #     losses.append(back_loss)
        #     loss_str.append(f"{name}_IBCE: {back_loss.item():.5f}")
        #
        for name, preds in all_preds2.items():
            # resized_gts = cus_sample(gts2, mode="size", factors=preds.shape[2:], interpolation="bilinear")
            # resized_gts = gts
            preds = cus_sample(preds, mode="size", factors=gts.shape[2:], interpolation="bilinear")
            # edge_loss = structure_loss(pred=preds, mask=resized_gts, iter_percentage=iter_percentage, method=method)
            # preds = preds.sigmoid()
            # edge_loss = sdice_loss(preds, resized_gts)
            # edge_loss = dice_loss(preds, resized_gts)
            # edge_loss = sdice_loss(preds, gts2)
            edge_loss = dice_loss(preds, gts2)

            # sod_loss = F.binary_cross_entropy(input=preds, target=resized_gts, reduction="mean")
            losses.append(edge_loss)
            loss_str.append(f"{name}_DICE: {edge_loss.item():.5f}")

        return sum(losses), " ".join(loss_str)

    def get_grouped_params(self):
        param_groups = {}
        for name, param in self.named_parameters():
            print(name)
            # print(param)
            if name.startswith("shared_encoder.layer"):
            # if name.startswith("shared_encoder.patch_embed"):
                param_groups.setdefault("pretrained", []).append(param)
            elif name.startswith("shared_encoder."):
                param_groups.setdefault("fixed", []).append(param)
            else:
                param_groups.setdefault("retrained", []).append(param)
        # print("111111111")
        print(param_groups.keys())
        return param_groups


# @MODELS.register()
# class ZoomNet_CK(ZoomNet):
#     def __init__(self):
#         super().__init__()
#         self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)
#
#     def encoder(self, x, dummy_arg=None):
#         assert dummy_arg is not None
#         x0, x1, x2, x3, x4 = self.shared_encoder(x)
#         return x0, x1, x2, x3, x4
#
#     def trans(self, x0, x1, x2, x3, x4):
#         x5, x4, x3, x2, x1 = self.translayer([x0, x1, x2, x3, x4])
#         return x5, x4, x3, x2, x1
#
#     def decoder(self, x5, x4, x3, x2, x1):
#         x = self.d5(x5)
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d4(x + x4)
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d3(x + x3)
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d2(x + x2)
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d1(x + x1)
#         x = cus_sample(x, mode="scale", factors=2)
#         logits = self.out_layer_01(self.out_layer_00(x))
#         return logits
#
#     def body(self, l_scale, m_scale, s_scale):
#         l_trans_feats = checkpoint(self.encoder, l_scale, self.dummy)
#         m_trans_feats = checkpoint(self.encoder, m_scale, self.dummy)
#         s_trans_feats = checkpoint(self.encoder, s_scale, self.dummy)
#         l_trans_feats = checkpoint(self.trans, *l_trans_feats)
#         m_trans_feats = checkpoint(self.trans, *m_trans_feats)
#         s_trans_feats = checkpoint(self.trans, *s_trans_feats)
#
#         feats = []
#         for layer_idx, (l, m, s) in enumerate(zip(l_trans_feats, m_trans_feats, s_trans_feats)):
#             siu_outs = checkpoint(self.merge_layers[layer_idx], l, m, s)
#             feats.append(siu_outs)
#
#         logits = checkpoint(self.decoder, *feats)
#         return dict(seg=logits)

@MODELS.register()
class ZoomNet_CK(ZoomNet):
    def __init__(self):
        super().__init__()
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def encoder(self, x, dummy_arg=None):
        assert dummy_arg is not None
        x0, x1, x2, x3, x4 = self.shared_encoder(x)
        return x0, x1, x2, x3, x4

    def trans(self, x0, x1, x2, x3, x4):
        x5, x4, x3, x2, x1 = self.translayer([x0, x1, x2, x3, x4])
        return x5, x4, x3, x2, x1

    def decoder(self, x5, x4, x3, x2, x1):
        x = self.d5(x5)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d4(x + x4)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d3(x + x3)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d2(x + x2)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d1(x + x1)
        x = cus_sample(x, mode="scale", factors=2)
        logits = self.out_layer_01(self.out_layer_00(x))
        return logits

    def body(self, m_scale):
        # l_trans_feats = checkpoint(self.encoder, l_scale, self.dummy)
        m_trans_feats = checkpoint(self.encoder, m_scale, self.dummy)
        # s_trans_feats = checkpoint(self.encoder, s_scale, self.dummy)
        # l_trans_feats = checkpoint(self.trans, *l_trans_feats)
        m_trans_feats = checkpoint(self.trans, *m_trans_feats)
        # s_trans_feats = checkpoint(self.trans, *s_trans_feats)

        feats = []
        for layer_idx, (m) in enumerate(zip(m_trans_feats)):
            siu_outs = checkpoint(self.merge_layers[layer_idx], m)
            feats.append(siu_outs)

        logits = checkpoint(self.decoder, *feats)
        return dict(seg=logits)
