from .layer import *

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F

import math

from pdb import set_trace

# class MeanShift(nn.Conv2d):
#     def __init__(self, rgb_mean, sign):
#         super(MeanShift, self).__init__(3, 3, kernel_size=1)
#         self.weight.data = torch.eye(3).view(3, 3, 1, 1)
#         self.bias.data = float(sign) * torch.Tensor(rgb_mean)
#
#         # Freeze the MeanShift layer
#         for params in self.parameters():
#             params.requires_grad = False

class ResidualBlock(nn.Module):
    def __init__(self, num_channels=256):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output, identity_data)
        return output


class EDSR(nn.Module):
    def __init__(self, num_block=32, num_channels=256, num_upscale=2, factor_upscale=1):
        super(EDSR, self).__init__()

        self.num_block = num_block
        self.num_channels = num_channels

        self.num_upscale = num_upscale
        self.factor_upscale = factor_upscale

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(ResidualBlock, self.num_block, self.num_channels)

        self.conv_mid = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.upscale_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels * self.factor_upscale, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.PixelShuffle(self.factor_upscale),
            PixelShuffle(ry=self.factor_upscale, rx=1),
        )
        self.upscale_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels * self.factor_upscale, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.PixelShuffle(self.factor_upscale),
            PixelShuffle(ry=self.factor_upscale, rx=1),
        )
        self.upscale_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels * self.factor_upscale, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.PixelShuffle(self.factor_upscale),
            PixelShuffle(ry=self.factor_upscale, rx=1),
        )
        self.upscale_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels * self.factor_upscale, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.PixelShuffle(self.factor_upscale),
            PixelShuffle(ry=self.factor_upscale, rx=1),
        )
        self.upscale_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels * self.factor_upscale, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.PixelShuffle(self.factor_upscale),
            PixelShuffle(ry=self.factor_upscale, rx=1),
        )

        self.conv_output = nn.Conv2d(in_channels=self.num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_block, num_channels):
        layers = []
        for _ in range(num_block):
            layers.append(block(num_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = self.sub_mean(x)
        # out = self.conv_input(out)

        out = self.conv_input(x)
        # out = self.conv_input(out)
        residual = out
        out = self.conv_mid(self.residual(out))
        out = torch.add(out, residual)
        # out = self.upscale4x(out)

        if self.num_upscale == 1:
            out = self.upscale_1(out)
        elif self.num_upscale == 2:
            out = self.upscale_1(out)
            out = self.upscale_2(out)
        elif self.num_upscale == 3:
            out = self.upscale_1(out)
            out = self.upscale_2(out)
            out = self.upscale_3(out)
        elif self.num_upscale == 4:
            out = self.upscale_1(out)
            out = self.upscale_2(out)
            out = self.upscale_3(out)
            out = self.upscale_4(out)
        elif self.num_upscale == 5:
            out = self.upscale_1(out)
            out = self.upscale_2(out)
            out = self.upscale_3(out)
            out = self.upscale_4(out)
            out = self.upscale_5(out)

        out = self.conv_output(out)

        # out = self.add_mean(out)
        return out


class UNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(UNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        """
        Encoder part
        """

        self.enc1_1 = CNR2d(1 * self.nch_in, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc1_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2_1 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc2_2 = CNR2d(2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3_1 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc3_2 = CNR2d(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool3 = Pooling2d(pool=2, type='avg')

        self.enc4_1 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc4_2 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool4 = Pooling2d(pool=2, type='avg')

        self.enc5_1 = CNR2d(8 * self.nch_ker, 2 * 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        """
        Decoder part
        """
        self.dec5_1 = CNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='nearest')
        #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='conv')

        self.dec4_2 = CNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec4_1 = CNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='nearest')
        #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='conv')

        self.dec3_2 = CNR2d(2 * 4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec3_1 = CNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='nearest')
        #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='conv')

        self.dec2_2 = CNR2d(2 * 2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec2_1 = CNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='nearest')
        #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='conv')

        self.dec1_2 = CNR2d(2 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec1_1 = CNR2d(1 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.fc = Conv2d(1 * self.nch_ker,        1 * self.nch_out, kernel_size=1, padding=0)

    def forward(self, x):

        """
        Encoder part
        """

        enc1 = self.enc1_2(self.enc1_1(x))
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(pool4)

        """
        Encoder part
        """
        dec5 = self.dec5_1(enc5)

        unpool4 = self.unpool4(dec5)
        cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4_1(self.dec4_2(cat4))

        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3_1(self.dec3_2(cat3))

        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2_1(self.dec2_2(cat2))

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1_1(self.dec1_2(cat1))

        x = self.fc(dec1)
        # x = torch.sigmoid(x)

        return x

class UNet_img(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(UNet_img, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        # if norm == 'bnorm':
        #     self.bias = False
        # else:
        #     self.bias = True

        """
        Encoder part
        """

        self.enc1_1 = CNR2d(1 * self.nch_in, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc1_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2_1 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc2_2 = CNR2d(2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3_1 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc3_2 = CNR2d(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool3 = Pooling2d(pool=2, type='avg')

        self.enc4_1 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc4_2 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool4 = Pooling2d(pool=2, type='avg')

        self.enc5_1 = CNR2d(8 * self.nch_ker, 2 * 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        """
        Decoder part
        """
        self.dec5_1 = CNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='nearest')
        #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='conv')

        self.dec4_2 = CNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec4_1 = CNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='nearest')
        #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='conv')

        self.dec3_2 = CNR2d(2 * 4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec3_1 = CNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='nearest')
        #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='conv')

        self.dec2_2 = CNR2d(2 * 2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec2_1 = CNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='nearest')
        #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='conv')

        self.dec1_2 = CNR2d(2 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec1_1 = CNR2d(1 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.dec_img1 = CNR2d(1 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec_img2 = CNR2d(1 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.fc_img = Conv2d(1 * self.nch_ker,        1 * self.nch_out, kernel_size=1, padding=0)

    def forward(self, x):

        """
        Encoder part
        """

        enc1 = self.enc1_2(self.enc1_1(x))
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(pool4)

        """
        Encoder part
        """
        dec5 = self.dec5_1(enc5)

        unpool4 = self.unpool4(dec5)
        cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4_1(self.dec4_2(cat4))

        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3_1(self.dec3_2(cat3))

        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2_1(self.dec2_2(cat2))

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1_1(self.dec1_2(cat1))

        img1 = self.dec_img1(dec1)
        img2 = self.dec_img1(img1)

        x_img = self.fc_img(img2)
        # x = torch.sigmoid(x)

        # return x_img
        return x_img, dec1

class UNet_prj(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(UNet_prj, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        # if norm == 'bnorm':
        #     self.bias = False
        # else:
        #     self.bias = True

        """
        Encoder part
        """

        self.enc1_1 = CNR2d(1 * self.nch_in, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc1_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2_1 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc2_2 = CNR2d(2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3_1 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc3_2 = CNR2d(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool3 = Pooling2d(pool=2, type='avg')

        self.enc4_1 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc4_2 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool4 = Pooling2d(pool=2, type='avg')

        self.enc5_1 = CNR2d(8 * self.nch_ker, 2 * 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        """
        Decoder part
        """
        self.dec5_1 = CNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='nearest')
        #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='conv')

        self.dec4_2 = CNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec4_1 = CNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='nearest')
        #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='conv')

        self.dec3_2 = CNR2d(2 * 4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec3_1 = CNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='nearest')
        #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='conv')

        self.dec2_2 = CNR2d(2 * 2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec2_1 = CNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='nearest')
        #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='conv')


        self.dec1_2 = CNR2d(2 * 1 * self.nch_ker,   1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec1_1 = CNR2d(1 * self.nch_ker,       1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.dec_prj1 = CNR2d(1 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec_prj2 = CNR2d(1 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.fc_prj = Conv2d(1 * self.nch_ker,      1 * self.nch_out, kernel_size=1, padding=0)


        self.dec_noise1 = CNR2d(1 * self.nch_ker,   1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec_noise2 = CNR2d(1 * self.nch_ker,   1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        # self.dec_noise1 = CNR2d(1 * self.nch_ker,   1 * self.nch_ker, kernel_size=3, stride=1, norm='inorm', relu=0.0)
        # self.dec_noise2 = CNR2d(1 * self.nch_ker,   1 * self.nch_ker, kernel_size=3, stride=1, norm='inorm', relu=0.0)

        self.fc_noise = Conv2d(1 * self.nch_ker,    1 * self.nch_out, kernel_size=1, padding=0)

    def forward(self, x):

        """
        Encoder part
        """

        enc1 = self.enc1_2(self.enc1_1(x))
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(pool4)

        """
        Encoder part
        """
        dec5 = self.dec5_1(enc5)

        unpool4 = self.unpool4(dec5)
        cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4_1(self.dec4_2(cat4))

        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3_1(self.dec3_2(cat3))

        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2_1(self.dec2_2(cat2))

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1_1(self.dec1_2(cat1))

        prj1 = self.dec_prj1(dec1)
        prj2 = self.dec_prj2(prj1)
        x_prj = self.fc_prj(prj2)

        noise1 = self.dec_noise1(dec1)
        noise2 = self.dec_noise2(noise1)
        x_noise = self.fc_noise(noise2)

        # x = torch.sigmoid(x)

        return x_prj, x_noise, dec1




class UNet_res_dom2dom(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='inorm', params=None):
        super(UNet_res_dom2dom, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        # self.nbatch = nbatch
        self.nbatch = params['nDctY']
        self.norm = norm
        self.params = params

        # if norm == 'bnorm':
        #     self.bias = False
        # else:
        #     self.bias = True

        """
        Encoder part
        """
        self.enc1 = ConvBlock(1 * self.nch_in, 1 * self.nch_ker, drop_prob=0.0)
        # self.img2prj1 = Img2Prj(params=self.params.copy(), nstage=self.params['nStage'] + 0, nslice=self.nbatch * 1 * self.nch_ker,
        #                         nview=self.params['nView'] // 1)
        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2 = ConvBlock(1 * self.nch_ker, 2 * self.nch_ker, drop_prob=0.0)
        # self.img2prj2 = Img2Prj(params=self.params.copy(), nstage=self.params['nStage'] + 1, nslice=self.nbatch * 2 * self.nch_ker,
        #                         nview=self.params['nView'] // 2)
        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3 = ConvBlock(2 * self.nch_ker, 4 * self.nch_ker, drop_prob=0.0)
        # self.img2prj3 = Img2Prj(params=self.params.copy(), nstage=self.params['nStage'] + 2, nslice=self.nbatch * 4 * self.nch_ker,
        #                         nview=self.params['nView'] // 4)
        self.pool3 = Pooling2d(pool=2, type='avg')

        self.enc4 = ConvBlock(4 * self.nch_ker, 8 * self.nch_ker, drop_prob=0.0)
        # self.img2prj4 = Img2Prj(params=self.params.copy(), nstage=self.params['nStage'] + 3, nslice=self.nbatch * 8 * self.nch_ker,
        #                         nview=self.params['nView'] // 8)
        self.pool4 = Pooling2d(pool=2, type='avg')

        self.enc5 = CNR2d(8 * self.nch_ker, 16 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.2)
        # self.img2prj5 = Img2Prj(params=self.params.copy(), nstage=self.params['nStage'] + 4, nslice=self.nbatch * 16 * self.nch_ker,
        #                         nview=self.params['nView'] // 16)

        """
        Decoder part
        """
        self.dec5 = CNR2d(16 * self.nch_ker, 16 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.2)

        self.unpool4 = DECNR2d(16 * self.nch_ker, 8 * self.nch_ker, kernel_size=2, stride=2, norm=self.norm, relu=0.2)
        self.dec4 = ConvBlock(16 * self.nch_ker, 8 * self.nch_ker, drop_prob=0.0)

        self.unpool3 = DECNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=2, stride=2, norm=self.norm, relu=0.2)
        self.dec3 = ConvBlock(8 * self.nch_ker, 4 * self.nch_ker, drop_prob=0.0)

        self.unpool2 = DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=2, stride=2, norm=self.norm, relu=0.2)
        self.dec2 = ConvBlock(4 * self.nch_ker, 2 * self.nch_ker, drop_prob=0.0)

        self.unpool1 = DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=2, stride=2, norm=self.norm, relu=0.2)
        self.dec1 = ConvBlock(2 * self.nch_ker, 1 * self.nch_ker, drop_prob=0.0)

        self.fc_img = nn.Sequential(
            ConvBlock(1 * self.nch_ker, self.nch_ker // 2, drop_prob=0.0),
            nn.Conv2d(self.nch_ker // 2, self.nch_out, kernel_size=1, stride=1),
        )

    def forward(self, x):
        """
        Encoder part
        """

        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        # enc1 = self.img2prj1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        # enc2 = self.img2prj2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        # enc3 = self.img2prj3(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)
        # enc4 = self.img2prj4(enc4)

        enc5 = self.enc5(pool4)
        # enc5 = self.img2prj5(enc5)

        """
        Encoder part
        """
        dec5 = self.dec5(enc5)

        unpool4 = self.unpool4(dec5)
        cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4(cat4)

        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3(cat3)

        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2(cat2)

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1(cat1)

        # img1 = self.dec_img1(dec1)
        # img2 = self.dec_img1(img1)

        x_img = self.fc_img(dec1)
        # x = torch.sigmoid(x)

        return x_img
        # return x_img, dec1



class UNet_res_img2prj(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='inorm', params=None):
        super(UNet_res_img2prj, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        # self.nbatch = nbatch
        self.nbatch = params['nDctY']
        self.norm = norm
        self.params = params

        # if norm == 'bnorm':
        #     self.bias = False
        # else:
        #     self.bias = True

        """
        Encoder part
        """
        self.enc1 = ConvBlock(1 * self.nch_in, 1 * self.nch_ker, drop_prob=0.0)
        self.img2prj1 = Img2Prj(params=self.params.copy(), nstage=self.params['nStage'] + 0, nslice=self.nbatch * 1 * self.nch_ker,
                                nview=self.params['nView'] // 1)
        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2 = ConvBlock(1 * self.nch_ker, 2 * self.nch_ker, drop_prob=0.0)
        self.img2prj2 = Img2Prj(params=self.params.copy(), nstage=self.params['nStage'] + 1, nslice=self.nbatch * 2 * self.nch_ker,
                                nview=self.params['nView'] // 2)
        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3 = ConvBlock(2 * self.nch_ker, 4 * self.nch_ker, drop_prob=0.0)
        self.img2prj3 = Img2Prj(params=self.params.copy(), nstage=self.params['nStage'] + 2, nslice=self.nbatch * 4 * self.nch_ker,
                                nview=self.params['nView'] // 4)
        self.pool3 = Pooling2d(pool=2, type='avg')

        self.enc4 = ConvBlock(4 * self.nch_ker, 8 * self.nch_ker, drop_prob=0.0)
        self.img2prj4 = Img2Prj(params=self.params.copy(), nstage=self.params['nStage'] + 3, nslice=self.nbatch * 8 * self.nch_ker,
                                nview=self.params['nView'] // 8)
        self.pool4 = Pooling2d(pool=2, type='avg')

        self.enc5 = CNR2d(8 * self.nch_ker, 16 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.2)
        self.img2prj5 = Img2Prj(params=self.params.copy(), nstage=self.params['nStage'] + 4, nslice=self.nbatch * 16 * self.nch_ker,
                                nview=self.params['nView'] // 16)

        """
        Decoder part
        """
        self.dec5 = CNR2d(16 * self.nch_ker, 16 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.2)

        self.unpool4 = DECNR2d(16 * self.nch_ker, 8 * self.nch_ker, kernel_size=2, stride=2, norm=self.norm, relu=0.2)
        self.dec4 = ConvBlock(16 * self.nch_ker, 8 * self.nch_ker, drop_prob=0.0)

        self.unpool3 = DECNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=2, stride=2, norm=self.norm, relu=0.2)
        self.dec3 = ConvBlock(8 * self.nch_ker, 4 * self.nch_ker, drop_prob=0.0)

        self.unpool2 = DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=2, stride=2, norm=self.norm, relu=0.2)
        self.dec2 = ConvBlock(4 * self.nch_ker, 2 * self.nch_ker, drop_prob=0.0)

        self.unpool1 = DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=2, stride=2, norm=self.norm, relu=0.2)
        self.dec1 = ConvBlock(2 * self.nch_ker, 1 * self.nch_ker, drop_prob=0.0)

        self.fc_img = nn.Sequential(
            ConvBlock(1 * self.nch_ker, self.nch_ker // 2, drop_prob=0.0),
            nn.Conv2d(self.nch_ker // 2, self.nch_out, kernel_size=1, stride=1),
        )

    def forward(self, x):
        """
        Encoder part
        """

        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        enc1 = self.img2prj1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        enc2 = self.img2prj2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        enc3 = self.img2prj3(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)
        enc4 = self.img2prj4(enc4)

        enc5 = self.enc5(pool4)
        enc5 = self.img2prj5(enc5)

        """
        Encoder part
        """
        dec5 = self.dec5(enc5)

        unpool4 = self.unpool4(dec5)
        cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4(cat4)

        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3(cat3)

        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2(cat2)

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1(cat1)

        # img1 = self.dec_img1(dec1)
        # img2 = self.dec_img1(img1)

        x_img = self.fc_img(dec1)
        # x = torch.sigmoid(x)

        return x_img
        # return x_img, dec1


class UNet_res_prj2img(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='inorm', params=None):
        super(UNet_res_prj2img, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        # self.nbatch = nbatch
        self.nbatch = params['nDctY']
        self.norm = norm
        self.params = params

        # if norm == 'bnorm':
        #     self.bias = False
        # else:
        #     self.bias = True

        """
        Encoder part
        """
        self.enc1 = ConvBlock(1 * self.nch_in, 1 * self.nch_ker, drop_prob=0.0)
        self.prj2img1 = Prj2Img(params=self.params.copy(), nstage=self.params['nStage'] + 0, nslice=self.nbatch * 1 * self.nch_ker, nview=self.params['nView'] // 1)
        self.pool1 = Pooling2d(pool=2, type='avg')


        self.enc2 = ConvBlock(1 * self.nch_ker, 2 * self.nch_ker, drop_prob=0.0)
        self.prj2img2 = Prj2Img(params=self.params.copy(), nstage=self.params['nStage'] + 1, nslice=self.nbatch * 2 * self.nch_ker, nview=self.params['nView'] // 2)
        self.pool2 = Pooling2d(pool=2, type='avg')
        

        self.enc3 = ConvBlock(2 * self.nch_ker, 4 * self.nch_ker, drop_prob=0.0)
        self.prj2img3 = Prj2Img(params=self.params.copy(), nstage=self.params['nStage'] + 2, nslice=self.nbatch * 4 * self.nch_ker, nview=self.params['nView'] // 4)
        self.pool3 = Pooling2d(pool=2, type='avg')
        

        self.enc4 = ConvBlock(4 * self.nch_ker, 8 * self.nch_ker, drop_prob=0.0)
        self.prj2img4 = Prj2Img(params=self.params.copy(), nstage=self.params['nStage'] + 3, nslice=self.nbatch * 8 * self.nch_ker, nview=self.params['nView'] // 8)
        self.pool4 = Pooling2d(pool=2, type='avg')
        

        self.enc5 = CNR2d(8 * self.nch_ker, 16 * self.nch_ker, kernel_size=3, stride=1, norm='inorm', relu=0.2)
        self.prj2img5 = Prj2Img(params=self.params.copy(), nstage=self.params['nStage'] + 4, nslice=self.nbatch * 16 * self.nch_ker, nview=self.params['nView'] // 16)
        
        """
        Decoder part
        """
        self.dec5 = CNR2d(16 * self.nch_ker, 16 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.2)

        self.unpool4 = DECNR2d(16 * self.nch_ker, 8 * self.nch_ker, kernel_size=2, stride=2, norm=self.norm, relu=0.2)
        self.dec4 = ConvBlock(16 * self.nch_ker, 8 * self.nch_ker, drop_prob=0.0)

        self.unpool3 = DECNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=2, stride=2, norm=self.norm, relu=0.2)
        self.dec3 = ConvBlock(8 * self.nch_ker, 4 * self.nch_ker, drop_prob=0.0)

        self.unpool2 = DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=2, stride=2, norm=self.norm, relu=0.2)
        self.dec2 = ConvBlock(4 * self.nch_ker, 2 * self.nch_ker, drop_prob=0.0)

        self.unpool1 = DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=2, stride=2, norm=self.norm, relu=0.2)
        self.dec1 = ConvBlock(2 * self.nch_ker, 1 * self.nch_ker, drop_prob=0.0)

        self.fc_img = nn.Sequential(
            ConvBlock(1 * self.nch_ker, self.nch_ker // 2, drop_prob=0.0),
            nn.Conv2d(self.nch_ker // 2, self.nch_out, kernel_size=1, stride=1),
        )

    def forward(self, x):

        """
        Encoder part
        """

        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        enc1 = self.prj2img1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        enc2 = self.prj2img2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        enc3 = self.prj2img3(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)
        enc4 = self.prj2img4(enc4)

        enc5 = self.enc5(pool4)
        enc5 = self.prj2img5(enc5)

        """
        Encoder part
        """
        dec5 = self.dec5(enc5)


        unpool4 = self.unpool4(dec5)
        cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4(cat4)

        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3(cat3)

        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2(cat2)

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1(cat1)

        # img1 = self.dec_img1(dec1)
        # img2 = self.dec_img1(img1)

        x_img = self.fc_img(dec1)
        # x = torch.sigmoid(x)

        return x_img
        # return x_img, dec1


"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

class UNet_res(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        factor_upscale: list = [0, ],
        num_upscale: int = 0,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.factor_upscale = factor_upscale
        self.num_upscale = num_upscale
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                # nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
                nn.Conv2d(ch, self.chans, kernel_size=1, stride=1),
            )
        )


        if self.num_upscale:
            up_scale = []
            for iup in range(self.num_upscale):
                up_scale += [nn.Conv2d(in_channels=self.chans, out_channels=self.chans * self.factor_upscale[iup], kernel_size=3, stride=1, padding=1, bias=False)]
                up_scale += [PixelShuffle(ry=self.factor_upscale[iup], rx=1)]

            self.up_scale = nn.Sequential(*up_scale)

        # self.upscale_1 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.chans, out_channels=self.chans * self.factor_upscale, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.PixelShuffle(self.factor_upscale),
        #     PixelShuffle(ry=self.factor_upscale, rx=1),
        # )
        # self.upscale_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.chans, out_channels=self.chans * self.factor_upscale, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.PixelShuffle(self.factor_upscale),
        #     PixelShuffle(ry=self.factor_upscale, rx=1),
        # )
        # self.upscale_3 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.chans, out_channels=self.chans * self.factor_upscale, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.PixelShuffle(self.factor_upscale),
        #     PixelShuffle(ry=self.factor_upscale, rx=1),
        # )
        # self.upscale_4 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.chans, out_channels=self.chans * self.factor_upscale, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.PixelShuffle(self.factor_upscale),
        #     PixelShuffle(ry=self.factor_upscale, rx=1),
        # )
        # self.upscale_5 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.chans, out_channels=self.chans * self.factor_upscale, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.PixelShuffle(self.factor_upscale),
        #     PixelShuffle(ry=self.factor_upscale, rx=1),
        # )

        self.fc = nn.Sequential(
            ConvBlock(self.chans, self.chans // 2, drop_prob),
            # nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            nn.Conv2d(self.chans // 2, self.out_chans, kernel_size=1, stride=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        if self.num_upscale:
            output = self.up_scale(output)

        output = self.fc(output)

        return output




class UNet_res_trans(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        # factor_upscale: list = [0, ],
        # num_upscale: int = 0,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        params: None = None,
        dir_trans: int = 0  # 0: dom2dom / 1: prj2img / 2: img2prj
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        # self.factor_upscale = factor_upscale
        # self.num_upscale = num_upscale
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.params = params
        self.dir_trans = dir_trans

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                # nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
                nn.Conv2d(ch, self.chans, kernel_size=1, stride=1),
            )
        )

        self.up_transform_i2p = nn.ModuleList()
        slice = self.params['nDctY'] * chans
        view = self.params['nView']
        stage = self.params['nStage']
        for _ in range(num_pool_layers):
            self.up_transform_i2p.insert(0, Img2Prj(params=self.params.copy(), nstage=stage, nslice=slice, nview=view))
            slice *= 2
            view //= 2
            stage += 1
        self.transform_i2p = Img2Prj(params=self.params.copy(), nstage=stage, nslice=slice, nview=view)

        self.up_transform_p2i = nn.ModuleList()
        slice = self.params['nDctY'] * chans
        view = self.params['nView']
        stage = self.params['nStage']
        for _ in range(num_pool_layers):
            self.up_transform_p2i.insert(0, Prj2Img(params=self.params.copy(), nstage=stage, nslice=slice, nview=view))
            slice *= 2
            view //= 2
            stage += 1
        self.transform_p2i = Prj2Img(params=self.params.copy(), nstage=stage, nslice=slice, nview=view)
        
        self.fc = nn.Sequential(
            ConvBlock(self.chans, self.chans // 2, drop_prob),
            # nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            nn.Conv2d(self.chans // 2, self.out_chans, kernel_size=1, stride=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        if self.dir_trans == 1:
            output = self.transform_p2i(output)
        elif self.dir_trans == 2:
            output = self.transform_i2p(output)

        # apply up-sampling layers
        for transpose_conv, conv, transform_p2i, transform_i2p in zip(self.up_transpose_conv, self.up_conv, self.up_transform_p2i, self.up_transform_i2p):
        # for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            if self.dir_trans == 1:
                downsample_layer = transform_p2i(downsample_layer)
            elif self.dir_trans == 2:
                downsample_layer = transform_i2p(downsample_layer)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        # if self.num_upscale:
        #     output = self.up_scale(output)

        output = self.fc(output)

        return output


# """
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# """
#
# class UNet_res(nn.Module):
#     """
#     PyTorch implementation of a U-Net model.
#
#     O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
#     for biomedical image segmentation. In International Conference on Medical
#     image computing and computer-assisted intervention, pages 234–241.
#     Springer, 2015.
#     """
#
#     def __init__(
#         self,
#         in_chans: int,
#         out_chans: int,
#         chans: int = 32,
#         num_pool_layers: int = 4,
#         drop_prob: float = 0.0,
#     ):
#         """
#         Args:
#             in_chans: Number of channels in the input to the U-Net model.
#             out_chans: Number of channels in the output to the U-Net model.
#             chans: Number of output channels of the first convolution layer.
#             num_pool_layers: Number of down-sampling and up-sampling layers.
#             drop_prob: Dropout probability.
#         """
#         super().__init__()
#
#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.chans = chans
#         self.num_pool_layers = num_pool_layers
#         self.drop_prob = drop_prob
#
#         self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
#         ch = chans
#         for _ in range(num_pool_layers - 1):
#             self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
#             ch *= 2
#         self.conv = ConvBlock(ch, ch * 2, drop_prob)
#
#         self.up_conv = nn.ModuleList()
#         self.up_transpose_conv = nn.ModuleList()
#         for _ in range(num_pool_layers - 1):
#             self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
#             self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
#             ch //= 2
#
#         self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
#         self.up_conv.append(
#             nn.Sequential(
#                 ConvBlock(ch * 2, ch, drop_prob),
#                 nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
#             )
#         )
#
#     def forward(self, image: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             image: Input 4D tensor of shape `(N, in_chans, H, W)`.
#
#         Returns:
#             Output tensor of shape `(N, out_chans, H, W)`.
#         """
#         stack = []
#         output = image
#
#         # apply down-sampling layers
#         for layer in self.down_sample_layers:
#             output = layer(output)
#             stack.append(output)
#             output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
#
#         output = self.conv(output)
#
#         # apply up-sampling layers
#         for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
#             downsample_layer = stack.pop()
#             output = transpose_conv(output)
#
#             # reflect pad on the right/botton if needed to handle odd input dimensions
#             padding = [0, 0, 0, 0]
#             if output.shape[-1] != downsample_layer.shape[-1]:
#                 padding[1] = 1  # padding right
#             if output.shape[-2] != downsample_layer.shape[-2]:
#                 padding[3] = 1  # padding bottom
#             if torch.sum(torch.tensor(padding)) != 0:
#                 output = F.pad(output, padding, "reflect")
#
#             output = torch.cat([output, downsample_layer], dim=1)
#             output = conv(output)
#
#         return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)

# class UNet_img(nn.Module):
#     def __init__(self, nch_in, nch_out, nch_ker=64, norm_enc='bnorm', norm_dec='bnorm'):
#         super(UNet_img, self).__init__()
#
#         self.nch_in = nch_in
#         self.nch_out = nch_out
#         self.nch_ker = nch_ker
#         self.norm_enc = norm_enc
#         self.norm_dec = norm_dec
#         self.is_snorm = is_snorm
#
#         # if norm == 'bnorm':
#         #     self.bias = False
#         # else:
#         #     self.bias = True
#
#         """
#         Encoder part
#         """
#
#         self.enc1_1 = CNR2d(1 * self.nch_in, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#         self.enc1_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#
#         self.pool1 = Pooling2d(pool=2, type='avg')
#
#         self.enc2_1 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#         self.enc2_2 = CNR2d(2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#
#         self.pool2 = Pooling2d(pool=2, type='avg')
#
#         self.enc3_1 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#         self.enc3_2 = CNR2d(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#
#         self.pool3 = Pooling2d(pool=2, type='avg')
#
#         self.enc4_1 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#         self.enc4_2 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#
#         self.pool4 = Pooling2d(pool=2, type='avg')
#
#         self.enc5_1 = CNR2d(8 * self.nch_ker, 2 * 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#
#         """
#         Decoder part
#         """
#         self.dec5_1 = CNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#
#         self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='nearest')
#         #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='bilinear')
#         #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='conv')
#
#         self.dec4_2 = CNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#         self.dec4_1 = CNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#
#         self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='nearest')
#         #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='bilinear')
#         #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='conv')
#
#         self.dec3_2 = CNR2d(2 * 4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#         self.dec3_1 = CNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#
#         self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='nearest')
#         #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='bilinear')
#         #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='conv')
#
#         self.dec2_2 = CNR2d(2 * 2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#         self.dec2_1 = CNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#
#         self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='nearest')
#         #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='bilinear')
#         #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='conv')
#
#         self.dec1_2 = CNR2d(2 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#         self.dec1_1 = CNR2d(1 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#
#         self.dec_img1 = CNR2d(1 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#         self.dec_img2 = CNR2d(1 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#
#         self.fc_img = Conv2d(1 * self.nch_ker,        1 * self.nch_out, kernel_size=1, padding=0)
#
#     def forward(self, x):
#
#         """
#         Encoder part
#         """
#
#         enc1 = self.enc1_2(self.enc1_1(x))
#         pool1 = self.pool1(enc1)
#
#         enc2 = self.enc2_2(self.enc2_1(pool1))
#         pool2 = self.pool2(enc2)
#
#         enc3 = self.enc3_2(self.enc3_1(pool2))
#         pool3 = self.pool3(enc3)
#
#         enc4 = self.enc4_2(self.enc4_1(pool3))
#         pool4 = self.pool4(enc4)
#
#         enc5 = self.enc5_1(pool4)
#
#         """
#         Encoder part
#         """
#         dec5 = self.dec5_1(enc5)
#
#         unpool4 = self.unpool4(dec5)
#         cat4 = torch.cat([enc4, unpool4], dim=1)
#         dec4 = self.dec4_1(self.dec4_2(cat4))
#
#         unpool3 = self.unpool3(dec4)
#         cat3 = torch.cat([enc3, unpool3], dim=1)
#         dec3 = self.dec3_1(self.dec3_2(cat3))
#
#         unpool2 = self.unpool2(dec3)
#         cat2 = torch.cat([enc2, unpool2], dim=1)
#         dec2 = self.dec2_1(self.dec2_2(cat2))
#
#         unpool1 = self.unpool1(dec2)
#         cat1 = torch.cat([enc1, unpool1], dim=1)
#         dec1 = self.dec1_1(self.dec1_2(cat1))
#
#         img1 = self.dec_img1(dec1)
#         img2 = self.dec_img1(img1)
#
#         x_img = self.fc_img(img2)
#         # x = torch.sigmoid(x)
#
#         return x_img
#
# class UNet_prj(nn.Module):
#     def __init__(self, nch_in, nch_out, nch_ker=64, norm_enc='bnorm', norm_dec='bnorm'):
#         super(UNet_prj, self).__init__()
#
#         self.nch_in = nch_in
#         self.nch_out = nch_out
#         self.nch_ker = nch_ker
#         self.norm_enc = norm_enc
#         self.norm_dec = norm_dec
#         self.is_snorm = is_snorm
#
#         # if norm == 'bnorm':
#         #     self.bias = False
#         # else:
#         #     self.bias = True
#
#         """
#         Encoder part
#         """
#
#         self.enc1_1 = CNR2d(1 * self.nch_in, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#         self.enc1_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#
#         self.pool1 = Pooling2d(pool=2, type='avg')
#
#         self.enc2_1 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#         self.enc2_2 = CNR2d(2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#
#         self.pool2 = Pooling2d(pool=2, type='avg')
#
#         self.enc3_1 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#         self.enc3_2 = CNR2d(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#
#         self.pool3 = Pooling2d(pool=2, type='avg')
#
#         self.enc4_1 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#         self.enc4_2 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#
#         self.pool4 = Pooling2d(pool=2, type='avg')
#
#         self.enc5_1 = CNR2d(8 * self.nch_ker, 2 * 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_enc, relu=0.0)
#
#         """
#         Decoder part
#         """
#         self.dec5_1 = CNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#
#         self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='nearest')
#         #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='bilinear')
#         #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='conv')
#
#         self.dec4_2 = CNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#         self.dec4_1 = CNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#
#         self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='nearest')
#         #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='bilinear')
#         #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='conv')
#
#         self.dec3_2 = CNR2d(2 * 4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#         self.dec3_1 = CNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#
#         self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='nearest')
#         #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='bilinear')
#         #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='conv')
#
#         self.dec2_2 = CNR2d(2 * 2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#         self.dec2_1 = CNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#
#         self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='nearest')
#         #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='bilinear')
#         #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='conv')
#
#
#         self.dec1_2 = CNR2d(2 * 1 * self.nch_ker,   1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#         self.dec1_1 = CNR2d(1 * self.nch_ker,       1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#
#         self.dec_prj1 = CNR2d(1 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#         self.dec_prj2 = CNR2d(1 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#
#         self.fc_prj = Conv2d(1 * self.nch_ker,      1 * self.nch_out, kernel_size=1, padding=0)
#
#
#         self.dec_noise1 = CNR2d(1 * self.nch_ker,   1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#         self.dec_noise2 = CNR2d(1 * self.nch_ker,   1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm_dec, relu=0.0)
#
#         self.fc_noise = Conv2d(1 * self.nch_ker,    1 * self.nch_out, kernel_size=1, padding=0)
#
#     def forward(self, x):
#
#         """
#         Encoder part
#         """
#
#         enc1 = self.enc1_2(self.enc1_1(x))
#         pool1 = self.pool1(enc1)
#
#         enc2 = self.enc2_2(self.enc2_1(pool1))
#         pool2 = self.pool2(enc2)
#
#         enc3 = self.enc3_2(self.enc3_1(pool2))
#         pool3 = self.pool3(enc3)
#
#         enc4 = self.enc4_2(self.enc4_1(pool3))
#         pool4 = self.pool4(enc4)
#
#         enc5 = self.enc5_1(pool4)
#
#         """
#         Encoder part
#         """
#         dec5 = self.dec5_1(enc5)
#
#         unpool4 = self.unpool4(dec5)
#         cat4 = torch.cat([enc4, unpool4], dim=1)
#         dec4 = self.dec4_1(self.dec4_2(cat4))
#
#         unpool3 = self.unpool3(dec4)
#         cat3 = torch.cat([enc3, unpool3], dim=1)
#         dec3 = self.dec3_1(self.dec3_2(cat3))
#
#         unpool2 = self.unpool2(dec3)
#         cat2 = torch.cat([enc2, unpool2], dim=1)
#         dec2 = self.dec2_1(self.dec2_2(cat2))
#
#         unpool1 = self.unpool1(dec2)
#         cat1 = torch.cat([enc1, unpool1], dim=1)
#         dec1 = self.dec1_1(self.dec1_2(cat1))
#
#         prj1 = self.dec_prj1(dec1)
#         prj2 = self.dec_prj2(prj1)
#         x_prj = self.fc_prj(prj2)
#
#         noise1 = self.dec_noise1(dec1)
#         noise2 = self.dec_noise2(noise1)
#         x_noise = self.fc_noise(noise2)
#
#         # x = torch.sigmoid(x)
#
#         return x_prj, x_noise


class ResNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', nblk=6):
        super(ResNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CNR2d(self.nch_in,      1 * self.nch_ker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)

        self.enc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.enc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        if self.nblk:
            res = []

            for i in range(self.nblk):
                res += [ResBlock(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')]

            self.res = nn.Sequential(*res)

        self.dec3 = DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec2 = DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec1 = CNR2d(1 * self.nch_ker, self.nch_out, kernel_size=7, stride=1, padding=3, norm=[], relu=[], bias=False)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        if self.nblk:
            x = self.res(x)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, nch_in, nch_ker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        # dsc1 : 256 x 256 x 3 -> 128 x 128 x 64
        # dsc2 : 128 x 128 x 64 -> 64 x 64 x 128
        # dsc3 : 64 x 64 x 128 -> 32 x 32 x 256
        # dsc4 : 32 x 32 x 256 -> 32 x 32 x 512
        # dsc5 : 32 x 32 x 512 -> 32 x 32 x 1

        self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[],        relu=[], bias=False)

        # self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=1, padding=1, norm=[], relu=0.2)
        # self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[], relu=[], bias=False)

    def forward(self, x):

        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.dsc3(x)
        x = self.dsc4(x)
        x = self.dsc5(x)

        # x = torch.sigmoid(x)

        return x


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)

    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)  # multi-GPUs
    return net

