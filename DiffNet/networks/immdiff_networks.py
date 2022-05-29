import torch
import torch.nn as nn
import torch.nn.functional as F

from DiffNet.networks.dgcnn import DGCNN2D

class ConvNet(nn.Module):
    def __init__(self, inchannels, outchannels, hchannels, kernel=2, nonlin=nn.ReLU(), final_nonlin=nn.Identity()):
        super(ConvNet, self).__init__()
        
        self.in_channels, self.out_channels = inchannels, outchannels
        self.nhidden = len(hchannels)
        channels = [inchannels] + hchannels + [outchannels]
        self.nonlin = [nonlin for k in range(self.nhidden)] + [final_nonlin]
        self.conv = nn.ModuleList(
            [
                nn.ConvTranspose1d(channels[k], channels[k+1], kernel, stride=2) for k in range(self.nhidden + 1)
            ]
        )
    def forward(self, x):
        for conv, nlin in zip(self.conv, self.nonlin):
            x = nlin(conv(x))
        return x



class LinearNet(nn.Module):
    def __init__(self, insize, outsize, hsizes, nonlin=nn.LeakyReLU(), final_nonlin=nn.Identity()):
        super(LinearNet, self).__init__()
        #### pulled from neuromancer MLP class
        
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        sizes = [insize] + hsizes + [outsize]
        self.nonlin = [nonlin for k in range(self.nhidden)] + [final_nonlin]
        self.linear = nn.ModuleList(
            [
                nn.Linear(sizes[k], sizes[k+1]) for k in range(self.nhidden + 1)
            ]
        )
    def forward(self, x):
        for lin, nlin in zip(self.linear, self.nonlin):
            x = nlin(lin(x))
        return x




# class ImmDiff(nn.Module):
#     def __init__(self, out_channels):
#         super(ImmDiff, self).__init__()

#         self.nurbs_to_img = ConvNet(1000, 32, [500 for i in range(3)], nonlin=torch.sin)
# 		# self.linear_net = LinearNet(2000, 1024, [1500 for i in range(3)])

#         self.up_conv_1 = nn.ConvTranspose2d(1, 2, kernel_size=2, stride=2)
#         self.up_conv_2 = nn.ConvTranspose2d(2, out_channels, kernel_size=2, stride=2)
    
#     def forward(self, x):
#         x = torch.tanh(self.nurbs_to_img(x)).unsqueeze(1)
#         x = torch.tanh(self.up_conv_1(x))
#         return self.up_conv_2(x)


# class ImmDiff(nn.Module):
#     def __init__(self, out_channels):
#         super(ImmDiff, self).__init__()

#         # self.nurbs_to_img = ConvNet(1000, 32, [500 for i in range(3)], nonlin=torch.sin)
# 		self.linear_net = LinearNet(2000, 1024, [1500 for i in range(3)])

#         self.up_conv_1 = nn.ConvTranspose2d(1, 2, kernel_size=2, stride=2)
#         self.up_conv_2 = nn.ConvTranspose2d(2, out_channels, kernel_size=2, stride=2)
    
#     def forward(self, x):
#         x = torch.tanh(self.linear_net(x.flatten()))
#         x = torch.tanh(self.up_conv_1(x))
#         return self.up_conv_2(x)



class ImmDiff(nn.Module):
    def __init__(self, out_channels):
        super(ImmDiff, self).__init__()

        self.out_channels = out_channels

        self.linear_net = LinearNet(2000, 1024, [1500 for i in range(6)], final_nonlin=nn.LeakyReLU())

        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, padding=1)
        self.conv1_up = nn.ConvTranspose2d(16,32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        self.conv2_up = nn.ConvTranspose2d(64,128, kernel_size=4)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=5, padding=1)
        self.conv3_up = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=6, padding=1)
        self.conv4_up = nn.ConvTranspose2d(16,self.out_channels, kernel_size=4)

    def forward(self, x):
        x = self.linear_net(x.flatten(1))
        x = torch.reshape(x, (x.shape[0], 1, 32, 32))
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv1_up(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2_up(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv3_up(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.conv4_up(x)
        # print('* '*10)
        # print(x.shape)
        # print('* '*10)
        # exit()
        return x




class IBN_DGCNN2d(nn.Module):
    def __init__(self):
        super(IBN_DGCNN2d, self).__init__()

        self.conv2d = nn.Conv2d(1,1, kernel_size=(5,3), stride=(5,1), padding=(0,1))

        self.dgcnn = DGCNN2D(domain_size=128, num_points=40, lowest_size=16)

    def forward(self, x):
        x = self.conv2d(x.unsqueeze(1))#.squeeze(1)
        # print(x.shape)
        # exit()
        x = F.leaky_relu(x)
        x = self.dgcnn(x)
        return x



class ImmDiff_VAE(nn.Module):
    def __init__(self, out_channels):
        super(ImmDiff_VAE, self).__init__()

        self.out_channels = out_channels

        self.linear_net_mu = LinearNet(2000, 256, [1024 for i in range(6)], final_nonlin=nn.LeakyReLU())
        self.linear_net_logvar = LinearNet(2000, 256, [1024 for i in range(6)], final_nonlin=nn.LeakyReLU())

        self.conv_up = nn.ConvTranspose2d(1, 16, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(16, 32, kernel_size=4, padding=1)
        self.conv1_up = nn.ConvTranspose2d(32,64, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
        self.conv2_up = nn.ConvTranspose2d(128,128, kernel_size=4)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=5, padding=1)
        self.conv3_up = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=6, padding=1)
        self.conv4_up = nn.ConvTranspose2d(16,self.out_channels, kernel_size=4)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu = self.linear_net_mu(x.flatten(1))
        logvar = self.linear_net_logvar(x.flatten(1))
        z = self.reparametrize(mu, logvar)

        x = torch.reshape(z, (x.shape[0], 1, 16, 16))

        x = F.leaky_relu(self.conv_up(x))
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv1_up(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2_up(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv3_up(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.conv4_up(x)
        # print('* '*10)
        # print(x.shape)
        # print('* '*10)
        # exit()
        return x, mu, logvar





class ImmDiff_Large(nn.Module):
    def __init__(self, out_channels):
        super(ImmDiff_Large, self).__init__()

        self.out_channels = out_channels

        self.linear_net = LinearNet(2000, 256, [1024 for i in range(7)], final_nonlin=nn.LeakyReLU())
        self.resnet = LinearNet(2000, 256, [1024 for i in range(7)], nonlin=torch.tanh, final_nonlin=nn.LeakyReLU())
        self.linear_net_sin = LinearNet(2000, 256, [1024 for i in range(7)], nonlin=torch.sin, final_nonlin=nn.LeakyReLU())
        self.pc_sparse = nn.Sequential(nn.Conv2d(1,1, kernel_size=(5,2), stride=5),
                                        nn.LeakyReLU())
        self.pc_sparse_up = nn.Sequential(nn.Linear(200, 256),
                                        nn.LeakyReLU())

        self.conv_up_1 = nn.ConvTranspose2d(4, 16, kernel_size=2, stride=2)
        self.conv_up_2 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.conv_up_3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(16, 32, kernel_size=4, padding=1)
        self.conv1_up = nn.ConvTranspose2d(32,64, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(64+32, 128, kernel_size=5, padding=1)
        self.conv2_up = nn.ConvTranspose2d(128,128, kernel_size=4)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, padding=1)
        self.conv3_up = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(32+32, 16, kernel_size=6, padding=1)
        self.conv4_up = nn.ConvTranspose2d(16,self.out_channels, kernel_size=4)

    def forward(self, x):
        x_lin = self.linear_net(x.flatten(1))
        x_res = self.resnet(x.flatten(1))
        x_sin = self.linear_net_sin(x.flatten(1))
        x_sparse = self.pc_sparse(x.unsqueeze(1))
        x_sparse = x_sparse.squeeze(1).squeeze(-1)
        x_sparse = self.pc_sparse_up(x_sparse)

        x_lin = torch.reshape(x_lin, (x.shape[0], 1, 16, 16))
        x_res = torch.reshape(x_res, (x.shape[0], 1, 16, 16))
        x_sin = torch.reshape(x_sin, (x.shape[0], 1, 16, 16))
        x_sparse = torch.reshape(x_sparse, (x.shape[0], 1, 16, 16))
        x = torch.cat((x_lin, x_res, x_sin, x_sparse),1)

        x_1 = F.leaky_relu(self.conv_up_1(x))
        x_2 = F.leaky_relu(self.conv_up_2(x_1))
        x_4 = F.leaky_relu(self.conv_up_3(x_2))

        x = F.leaky_relu(self.conv1(x_1))
        x = F.leaky_relu(self.conv1_up(x))
        x = F.leaky_relu(self.conv2(torch.cat((x, x_2),1)))
        x = F.leaky_relu(self.conv2_up(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv3_up(x))
        x = F.leaky_relu(self.conv4(torch.cat((x, x_4),1)))
        x = self.conv4_up(x)
        return x









class ImmDiff_Large_normals(nn.Module):
    def __init__(self, out_channels):
        super(ImmDiff_Large_normals, self).__init__()

        self.out_channels = out_channels

        self.lin_pc = LinearNet(2000, 256, [1024 for i in range(7)], final_nonlin=nn.LeakyReLU())
        self.lin_pc_skip = LinearNet(2000, 256, [1024], final_nonlin=nn.LeakyReLU())
        self.lin_norm = LinearNet(2000, 256, [1024 for i in range(7)], final_nonlin=nn.LeakyReLU())
        self.lin_norm_skip = LinearNet(2000, 256, [1024], final_nonlin=nn.LeakyReLU())
        

        self.conv_up_1 = nn.ConvTranspose2d(4, 16, kernel_size=2, stride=2)
        self.conv_up_2 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.conv_up_3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(16, 32, kernel_size=4, padding=1)
        self.conv1_up = nn.ConvTranspose2d(32,64, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(64+32, 128, kernel_size=5, padding=1)
        self.conv2_up = nn.ConvTranspose2d(128,128, kernel_size=4)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, padding=1)
        self.conv3_up = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(32+32, 16, kernel_size=6, padding=1)
        self.conv4_up = nn.ConvTranspose2d(16,self.out_channels, kernel_size=4)

    def forward(self, x, y):
        lin_pc = self.lin_pc(x.flatten(1))
        lin_pc_skip = self.lin_pc_skip(x.flatten(1))
        x_nrm = self.lin_norm(y.flatten(1))
        x_nrm_skip = self.lin_norm_skip(y.flatten(1))

        lin_pc = torch.reshape(lin_pc, (x.shape[0], 1, 16, 16))
        lin_pc_skip = torch.reshape(lin_pc_skip, (x.shape[0], 1, 16, 16))
        x_nrm = torch.reshape(x_nrm, (x.shape[0], 1, 16, 16))
        x_nrm_skip = torch.reshape(x_nrm_skip, (x.shape[0], 1, 16, 16))
        x = torch.cat((lin_pc, lin_pc_skip, x_nrm, x_nrm_skip),1)

        x_1 = F.leaky_relu(self.conv_up_1(x))
        x_2 = F.leaky_relu(self.conv_up_2(x_1))
        x_4 = F.leaky_relu(self.conv_up_3(x_2))

        x = F.leaky_relu(self.conv1(x_1))
        x = F.leaky_relu(self.conv1_up(x))
        x = F.leaky_relu(self.conv2(torch.cat((x, x_2),1)))
        x = F.leaky_relu(self.conv2_up(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv3_up(x))
        x = F.leaky_relu(self.conv4(torch.cat((x, x_4),1)))
        x = self.conv4_up(x)
        return x



class eikonal_linear(nn.Module):
    def __init__(self):
        super(eikonal_linear, self).__init__()

        self.linear = LinearNet(2000, 1024, [1500 for i in range(2)], nonlin=torch.sin)

    def forward(self, x):
        x = self.linear(x.flatten(1))
        x = torch.reshape(x, (x.shape[0], 1, 32, 32))
        y = torch.ones_like(x)
        z = torch.cat((x,y),1)
        return z