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
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
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
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
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
    def __init__(self, in_channels=1, out_channels=1, num_outputs=3):
        super(GoodGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outputs = num_outputs
        self.down1 = UNetDown(in_channels, 32)
        self.down2 = UNetDown(32, 64)
        self.down3 = UNetDown(64, 128)
        self.down4 = UNetDown(128, 256, dropout=0.5)
        self.down5 = UNetDown(256, 256, dropout=0.5)

        # self.up3 = []
        # self.up4 = []
        # self.up5 = []
        # self.up6 = []
        # self.final = []

        self.up3 = nn.ModuleList()
        self.up4 = nn.ModuleList()
        self.up5 = nn.ModuleList()
        self.up6 = nn.ModuleList()
        self.final = nn.ModuleList()
        
        for _ in range(self.num_outputs):
            self.up3.append(UNetUp(256, 256, dropout=0.5))
            self.up4.append(UNetUp(512, 128, dropout=0.5))
            self.up5.append(UNetUp(256, 64))
            self.up6.append(UNetUp(128, 32))
            self.final.append(nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.ZeroPad2d((1, 0, 1, 0)),
                    nn.Conv2d(64, out_channels, 4, padding=1),
                    nn.Sigmoid(),
                    ))

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
        outs = []
        print("self.num_outputs = ", self.num_outputs)
        for idx in range(self.num_outputs):
            u3 = self.up3[idx](d5, d4)
            u4 = self.up4[idx](u3, d3)
            u5 = self.up5[idx](u4, d2)
            u6 = self.up6[idx](u5, d1)
            outs.append(self.final[idx](u6))
        return outs