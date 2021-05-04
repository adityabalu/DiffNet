from torch import nn
from torch.autograd import grad
import torch
import numpy as np



from torch.nn.modules.utils import _pair

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GoodGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(GoodGenerator, self).__init__()

        self.down1 = UNetDown(in_channels, 16)
        self.down2 = UNetDown(16, 32)
        self.down3 = UNetDown(32, 64)
        self.down4 = UNetDown(64, 128, dropout=0.5)
        self.down5 = UNetDown(128, 128, normalize=False)
        # self.down6 = UNetDown(128, 128, dropout=0.5)
        # self.down7 = UNetDown(128, 128, normalize=False, dropout=0.5)

        # self.up1 = UNetUp(128, 128, dropout=0.5)
        # self.up2 = UNetUp(128, 128, dropout=0.5)
        self.up3 = UNetUp(128, 128, dropout=0.5)
        self.up4 = UNetUp(256, 64, dropout=0.5)
        self.up5 = UNetUp(128, 32)
        self.up6 = UNetUp(64, 16)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(32, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        # d6 = self.down6(d5)
        # d7 = self.down7(d6)

        # u1 = self.up1(d7, d6)
        # u2 = self.up2(d6, d5)
        u3 = self.up3(d5, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        return self.final(u6)