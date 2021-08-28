import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class ImplicitConv(nn.Module):
    """docstring for AE"""
    def __init__(self, in_channels, out_channels, dims=64, n_downsample=4):
        super(ImplicitConv, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, 512, 1))
        layers.append(nn.InstanceNorm2d(512))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        for _ in range(8):
            layers.append(nn.Conv2d(512, 512, 1))
            layers.append(nn.InstanceNorm2d(512))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(512, out_channels, 1))
        self.model = nn.Sequential(*layers, nn.Tanh())


    def forward(self, x):
        out = self.model(x)
        return out

