# -*- coding: utf-8 -*-

import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as tF

def convblock(inc, ouc, kernel_size, bn=True):
    padding = kernel_size // 2
    module = nn.Sequential(
        nn.Conv2d(inc, ouc, kernel_size=kernel_size, stride=1, padding=padding, bias=not bn),
        nn.BatchNorm2d(ouc) if bn else nn.Identity(),
        nn.ReLU(inplace=True)
    )
    if not bn: 
    #     init.kaiming_uniform_(module[0].weight, mode='fan_in', nonlinearity='relu')
        init.constant_(module[0].bias, 0.)

    return module


def conv_3x3(inc, ouc, bn=True):
    return convblock(inc, ouc, kernel_size=3, bn=bn)

class UpSample_P2P(nn.Module):
    def __init__(self, incs, ouc, bn=True, relu=True):
        super().__init__()
        self.align_layers = nn.ModuleList([
            nn.Conv2d(inc, ouc, kernel_size=1, stride=1, padding=0, bias=not bn) for inc in incs
        ])
        if not bn:
            for layer in self.align_layers:
                init.constant_(layer.bias, 0.)

        self.fuse = nn.Sequential(
            nn.Conv2d(ouc, ouc, kernel_size=3, stride=1, padding=1, bias= not bn),
            nn.BatchNorm2d(ouc) if bn else nn.Identity(),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        self.fuse_channel = ouc
    
    def forward(self, xs):
        x0 = self.align_layers[0](xs[0])
        out_shape = x0.shape[-2:]
        for x, layer in zip(xs[1:], self.align_layers[1:]):
            x = tF.interpolate(layer(x), out_shape, mode='bilinear', align_corners=False)
            x0 = x0 + x
        x = self.fuse(x0)
        return x



class SimpleDecoder(nn.Sequential):
    def __init__(self, in_channel = 128, fea_channel=64, up_scale=1, out_channel=1):
        super().__init__(
            conv_3x3(in_channel, fea_channel, bn=False),
            conv_3x3(fea_channel, fea_channel, bn=False),
            nn.Conv2d(fea_channel, out_channel * (up_scale ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale
        init.constant_(self[-2].bias, 0.)