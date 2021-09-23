import os
import sys
import math
import json
import torch
import numpy as np

import scipy.io
from scipy import ndimage
import matplotlib
from skimage import io
# matplotlib.use("pgf")
matplotlib.rcParams.update({
    # 'font.family': 'serif',
    'font.size':12,
})
from matplotlib import pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
seed_everything(42)

import DiffNet
from DiffNet.DiffNetFEM import DiffNet2DFEM
from torch.utils import data



class Burg2DXT(data.Dataset):
    'PyTorch dataset for PCVox'
    def __init__(self, domain_size=64):
        """
        Initialization
        """

        x = np.linspace(-1, 1, domain_size)
        t = np.linspace(0, 1, domain_size)

        xx , tt = np.meshgrid(x, t)
        self.x = xx
        self.y = tt
        self.bc1 = np.ones_like(xx)*(-10)
        self.bc1_val = np.zeros_like(xx)
        self.bc1[:,0] = 1.0
        self.bc1_val[:,0] = np.cos(2*math.pi*2*x)
        # print(self.bc1)
        # print(self.bc1_val)
        # exit()
        self.bc2 = np.ones_like(xx)*(-10)
        self.bc2[0,:] = 1
        self.bc2[-1,:] = 1

        self.n_samples = 100


    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.x, self.bc1, self.bc2, self.bc1_val])
        forcing = np.ones_like(self.x)*(0.01/math.pi)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)


class Burgers(DiffNet2DFEM):
    """docstring for Eiqonal"""
    def __init__(self, network, dataset, **kwargs):
        super(Burgers, self).__init__(network, dataset, **kwargs)

    def loss(self, u, inputs_tensor, forcing_tensor):

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        bc1 = inputs_tensor[:,1:2,:,:]
        bc1_val = inputs_tensor[:,3:4,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        u = torch.where(bc1>=(-5.0), u*0.0 + bc1_val, u)
        u = torch.where(bc2>=(-5.0), u*0.0, u)

        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_xx_gp = self.gauss_pt_evaluation_der2_x(u)
        u_t_gp = self.gauss_pt_evaluation_der_y(u)

        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(u_gp)
        res_elmwise1 = transformation_jacobian * (u_t_gp + u_gp*u_x_gp)**2

        res_elmwise = torch.sum(res_elmwise1, 1) 
        loss = torch.mean(res_elmwise) 
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        # return self.network(inputs_tensor), inputs_tensor, forcing_tensor
        return self.network[0], inputs_tensor, forcing_tensor

    def configure_optimizers(self):
        """
        Configure optimizer for network parameters
        """
        lr = self.learning_rate
        opts = [torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        # opts = [torch.optim.Adam(self.network)]
        return opts, []

    def on_epoch_end(self):
        fig, axs = plt.subplots(1, 4, figsize=(2*4,1.2),
                            subplot_kw={'aspect': 'auto'}, squeeze=True)
        # for ax in axs:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        self.network.eval()
        inputs, forcing = self.dataset[0]

        u, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        bc1 = inputs_tensor[:,1:2,:,:]
        bc1_val = inputs_tensor[:,3:4,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        u = torch.where(bc1>=0.0, u*0.0 + bc1_val, u)
        u = torch.where(bc2>0.01, u*0.0, u)

        u = u.squeeze().detach().cpu()
        bc1 = bc1.squeeze().detach().cpu()
        bc2 = bc2.squeeze().detach().cpu()

        im0 = axs[0].imshow(u,cmap='jet')
        fig.colorbar(im0, ax=axs[0])
        axs[0].set_xlim([0.0,u.shape[0]])
        axs[0].set_ylim([0.0,u.shape[1]])
        x = np.linspace(-1.0,1.0,u.shape[0])
        im1 = axs[1].plot(x.tolist(), u[:,10].tolist())
        axs[1].set_xlim([-1.0,1.0])
        axs[1].set_ylim([-1.0,1.0])
        im2 = axs[2].plot(x.tolist(), u[:,30].tolist())
        axs[2].set_xlim([-1.0,1.0])
        axs[2].set_ylim([-1.0,1.0])
        im3 = axs[3].plot(x.tolist(), u[:,40].tolist())
        axs[3].set_xlim([-1.0,1.0])
        axs[3].set_ylim([-1.0,1.0])
        plt.tight_layout()
        # im1 = axs[1].imshow(u,cmap='jet', vmin=-1.0, vmax=1.0)
        # fig.colorbar(im1, ax=axs[1])  
        # im2 = axs[2].imshow(u>0,cmap='Greys')
        # fig.colorbar(im2, ax=axs[2])
        # im3 = axs[3].imshow(res_elmwise,cmap='jet', vmin=0.0, vmax=1.0)
        # fig.colorbar(im3, ax=axs[3])  

        # im2 = axs[2].imshow(bc1,cmap='jet')
        # fig.colorbar(im2, ax=axs[2])
        # im3 = axs[3].imshow(bc2,cmap='jet')
        # fig.colorbar(im3, ax=axs[3])  

        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    u_tensor = np.ones((1,1,257,257))
    # u_tensor = (-1)*np.random.rand(1,1,256,256)
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    dataset = Burg2DXT(domain_size=257)
    basecase = Burgers(network, dataset, batch_size=1, fem_basis_deg=2)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="2d_space_time")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=30, deterministic=True, profiler="simple")

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