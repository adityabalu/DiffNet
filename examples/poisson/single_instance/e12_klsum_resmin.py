import os
import sys
import json
import torch
import numpy as np

import matplotlib
# matplotlib.use("pgf")
matplotlib.rcParams.update({
    # 'font.family': 'serif',
    'font.size':12,
})
from matplotlib import pyplot as plt

from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
seed_everything(42)

import DiffNet
from DiffNet.networks.wgan import GoodNetwork
from DiffNet.DiffNetFEM import DiffNet2DFEM
from DiffNet.datasets.single_instances.klsum import Dataset
from DiffNet.datasets.single_instances.rectangles import RectangleManufactured
from DiffNet.networks.autoencoders import AE

def stiffness_vs_values_conv(tensor, N, nsd=2, stride=1):
    if nsd == 2:
        conv_gp = nn.functional.conv2d
    elif nsd == 3:
        conv_gp = nn.functional.conv3d

    result_list = []
    for i in range(len(N)):
        result_list.append(conv_gp(tensor, N[i], stride=stride))
    return torch.cat(result_list, 1)

class Poisson(DiffNet2DFEM):
    """docstring for Poisson"""
    def __init__(self, network, dataset, **kwargs):
        super(Poisson, self).__init__(network, dataset, **kwargs)
        self.mapping_type = kwargs.get('mapping_type', 'no_network')

        self.Kmatrices = nn.ParameterList()
        Kmx = np.array([[4.,-1.,-1.,-2.],[-1.,4.,-2.,-1],[-1.,-2.,4.,-1.],[-2.,-1.,-1.,4.]])/6.
        Kmx = torch.FloatTensor(Kmx)
        for j in range(4):
            k = Kmx[j,:].reshape((2,2))
            print("k = ", k*6)
            self.Kmatrices.append(nn.Parameter(k.unsqueeze(0).unsqueeze(1), requires_grad=False))
        # print("self.Kmatrices[0] = ", self.Kmatrices[0])
        print("self.Kmatrices[0] = ", self.Kmatrices[0].shape)

    def loss(self, u, inputs_tensor, forcing_tensor):

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        R = torch.zeros_like(u)
        R_split = stiffness_vs_values_conv(u, self.Kmatrices)

        R[:,0, 0:-1, 0:-1] += R_split[:,0, :, :]
        R[:,0, 0:-1, 1:  ] += R_split[:,1, :, :]
        R[:,0, 1:  , 0:-1] += R_split[:,2, :, :]
        R[:,0, 1:  , 1:  ] += R_split[:,3, :, :]

        loss = torch.norm(R,'fro')**2
        # print("R = \n", R)
        # print("loss = ", loss)
        # exit()
        return loss        

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        if self.mapping_type == 'no_network':
            return self.network[0], inputs_tensor, forcing_tensor
        elif self.mapping_type == 'network':
            nu = inputs_tensor[:,0:1,:,:]
            return self.network(nu), inputs_tensor, forcing_tensor

    def configure_optimizers(self):
        """
        Configure optimizer for network parameters
        """
        lr = self.learning_rate
        # opts = [torch.optim.LBFGS(self.network.parameters(), lr=1.0, max_iter=5)]
        opts = [torch.optim.Adam(self.network.parameters(), lr=lr)]
        return opts, []

    def on_epoch_end(self):
        fig, axs = plt.subplots(1, 2, figsize=(2*2,1.2),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        self.network.eval()
        inputs, forcing = self.dataset[0]

        u, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        k = nu.squeeze().detach().cpu()
        u = u.squeeze().detach().cpu()

        im0 = axs[0].imshow(k,cmap='jet')
        fig.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(u,cmap='jet')
        fig.colorbar(im1, ax=axs[1])  
        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    domain_size = 32
    # mapping_type = 'network'
    mapping_type = 'no_network'
    if mapping_type == 'no_network':
        u_tensor = np.ones((1,1,domain_size, domain_size))
        network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    elif mapping_type == 'network':
        network = AE(in_channels=1, out_channels=1, dims=domain_size, n_downsample=3)
    dataset = Dataset('example-coefficients-3.txt', domain_size=domain_size)
    # dataset = RectangleManufactured(domain_size=domain_size)
    basecase = Poisson(network, dataset, batch_size=1)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="klsum-resmin")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=20, deterministic=True, profiler="simple")

    # ------------------------
    # 4 Training
    # ------------------------

    trainer.fit(basecase)

    # ------------------------
    # 5 SAVE NETWORK
    # ------------------------
    torch.save(basecase.network, os.path.join(logger.log_dir, 'network.pt'))


if __name__ == '__main__':
    main()