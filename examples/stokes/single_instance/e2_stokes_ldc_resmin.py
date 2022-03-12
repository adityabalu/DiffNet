import os
import sys
import math
import json
import torch
import numpy as np

import scipy.io
from scipy import ndimage
import matplotlib
# from skimage import io
# matplotlib.use("pgf")
matplotlib.rcParams.update({
    # 'font.family': 'serif',
    'font.size':8,
})
from matplotlib import pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
seed_everything(42)

import DiffNet
from DiffNet.DiffNetFEM import DiffNet2DFEM
from torch.utils import data
from e1_stokes_base_resmin import Stokes2D

class Stokes_LDC_Dataset(data.Dataset):
    'PyTorch dataset for Stokes_MMS_Dataset'
    def __init__(self, domain_size=64, Re=1):
        """
        Initialization
        """

        x = np.linspace(0, 1, domain_size)
        y = np.linspace(0, 1, domain_size)

        xx , yy = np.meshgrid(x, y)
        self.x = xx
        self.y = yy
        # bc1 for fixed boundaries
        self.bc1 = np.zeros_like(xx)
        self.bc1[ 0, :] = 1.0
        self.bc1[-1, :] = 1.0
        self.bc1[ :, 0] = 1.0
        self.bc1[ :,-1] = 1.0

        self.bc2 = np.zeros_like(xx)
        self.bc2[ 0, :] = 1.0
        self.bc2[-1, :] = 1.0
        self.bc2[ :, 0] = 1.0
        self.bc2[ :,-1] = 1.0

        self.bc3 = np.zeros_like(xx)
        self.bc3[0:1,0:1] = 1.0

        self.Re = Re
        self.n_samples = 100

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.x, self.y, self.bc1, self.bc2, self.bc3])

        forcing = np.ones_like(self.x)*(1/self.Re)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

class Stokes_LDC(Stokes2D):
    """docstring for Stokes_LDC"""
    def __init__(self, network, dataset, **kwargs):
        super(Stokes_LDC, self).__init__(network, dataset, **kwargs)

        u_bc = np.zeros_like(self.dataset.x); u_bc[-1,:] = 1. - 4. * (self.dataset.x[-1,:]-0.5)**2
        v_bc = np.zeros_like(self.dataset.x)
        p_bc = np.zeros_like(self.dataset.x)

        self.u_bc = torch.FloatTensor(u_bc)
        self.v_bc = torch.FloatTensor(v_bc)
        self.p_bc = torch.FloatTensor(p_bc)

    def loss(self, pred, inputs_tensor, forcing_tensor):
        R1, R2, R3 = self.calc_residuals(pred, inputs_tensor, forcing_tensor)
        # loss = torch.norm(R1, 'fro') + torch.norm(R2, 'fro') + torch.norm(R3, 'fro')
        return torch.norm(R1, 'fro'), torch.norm(R2, 'fro'), torch.norm(R3, 'fro')

    def training_step(self, batch, batch_idx, optimizer_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        loss_vals = self.loss(u, inputs_tensor, forcing_tensor)
        # self.log('PDE_loss', sum(loss_vals).item())
        self.log('loss_u', loss_vals[0].item())
        self.log('loss_v', loss_vals[1].item())
        self.log('loss_p', loss_vals[2].item())
        # return loss_vals[optimizer_idx]
        return {"loss": loss_vals[optimizer_idx]}

    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs["loss"]
        # unorm = training_step_outputs["unorm"]
        # self.log('loss', loss.item())
        # self.log('unorm', unorm.item())
        return training_step_outputs

    def configure_optimizers(self):
        # print("self.network[0].parameters() = ", self.network[0].parameters())
        lr = self.learning_rate
        # opts = [torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        opts = [torch.optim.Adam(self.network, lr=lr), torch.optim.Adam(self.network, lr=lr), torch.optim.Adam(self.network, lr=lr)]
        # opts = [torch.optim.Adam(self.network, lr=lr), torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        return opts, []

    def plot_contours(self, u, v, p, u_x, v_y):
        fig, axs = plt.subplots(3, 3, figsize=(4*3,2.4*3),
                            subplot_kw={'aspect': 'auto'}, squeeze=True)

        for ax_row in axs:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([]) 
        
        div = u_x + v_y

        im0 = axs[0,0].imshow(u,cmap='jet', origin='lower')
        fig.colorbar(im0, ax=axs[0,0])
        im1 = axs[0,1].imshow(v,cmap='jet',origin='lower')
        fig.colorbar(im1, ax=axs[0,1])  
        im2 = axs[0,2].imshow(p,cmap='jet',origin='lower')
        fig.colorbar(im2, ax=axs[0,2])
        x = np.linspace(0, 1, u.shape[0])
        y = np.linspace(0, 1, u.shape[1])

        # im = axs[1,0].imshow(u-self.u_exact.numpy(),cmap='jet', origin='lower')
        # fig.colorbar(im, ax=axs[1,0])
        # im = axs[1,1].imshow(v-self.v_exact.numpy(),cmap='jet',origin='lower')
        # fig.colorbar(im, ax=axs[1,1])  
        # im = axs[1,2].imshow(p-self.p_exact.numpy(),cmap='jet',origin='lower')
        # fig.colorbar(im, ax=axs[1,2])

        im3 = axs[1,0].imshow(div,cmap='jet',origin='lower')
        fig.colorbar(im3, ax=axs[1,0])  
        im4 = axs[1,1].imshow((u**2 + v**2)**0.5,cmap='jet',origin='lower')
        fig.colorbar(im4, ax=axs[1,1])
        xx , yy = np.meshgrid(x, y)
        im5 = axs[1,2].streamplot(xx, yy, u, v, color='k', cmap='jet')        

        baseline_cut = np.array([
                [0.0032066932270914394, -0.0007171314741036827],
                [-0.08300988047808777, 0.15482071713147427],
                [-0.12839219123505985, 0.2657370517928288],
                [-0.15676031872509966, 0.3358565737051793],
                [-0.18517529880478095, 0.4149003984063745],
                [-0.20523043824701204, 0.501593625498008],
                [-0.20285211155378485, 0.5819123505976096],
                [-0.15568717131474114, 0.664780876494024],
                [-0.06084860557768934, 0.7336254980079682],
                [0.08302342629482051, 0.7960956175298806],
                [0.2402690836653384, 0.8445418326693228],
                [0.40455490039840614, 0.8853386454183267],
                [0.5471942629482069, 0.9159362549800798],
                [0.670280478087649, 0.9376095617529882],
                [0.7653800796812744, 0.9567330677290837],
                [0.8395198406374498, 0.9682071713147412],
                [0.9961859760956173, 0.9937051792828686]
            ])

        mid_idx = int(self.domain_size/2)
        im = axs[2,0].plot(self.dataset.y[:,mid_idx], u[:,mid_idx],label='DiffNet')
        im = axs[2,0].plot(baseline_cut[:,1], baseline_cut[:,0],label='Ghia')
        axs[2,0].legend()
        # im = axs[1,1].imshow(v-self.v_exact.numpy(),cmap='jet',origin='lower')
        # fig.colorbar(im, ax=axs[1,1])          

        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    domain_size = 64
    dir_string = "stokes_ldc"
    max_epochs = 1000

    x = np.linspace(0, 1, domain_size)
    y = np.linspace(0, 1, domain_size)
    xx , yy = np.meshgrid(x, y)

    dataset = Stokes_LDC_Dataset(domain_size=domain_size)
    v1 = np.zeros_like(dataset.x) # np.sin(math.pi*xx)*np.cos(math.pi*yy)
    v2 = np.zeros_like(dataset.x) # -np.cos(math.pi*xx)*np.sin(math.pi*yy)
    p  = np.zeros_like(dataset.x) # np.sin(math.pi*xx)*np.sin(math.pi*yy)
    u_tensor = np.expand_dims(np.array([v1,v2,p]),0)
    print(u_tensor.shape)
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    print("network = ", network)
    basecase = Stokes_LDC(network, dataset, domain_size=domain_size, batch_size=1, fem_basis_deg=1, learning_rate=0.001)

    # Initialize trainer
    logger = pl.loggers.TensorBoardLogger('.', name=dir_string)
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=max_epochs, deterministic=True, profiler="simple")

    # Training
    trainer.fit(basecase)
    # Save network
    torch.save(basecase.network, os.path.join(logger.log_dir, 'network.pt'))


if __name__ == '__main__':
    main()