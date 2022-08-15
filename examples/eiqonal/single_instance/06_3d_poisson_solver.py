import os
import sys
import json
import math
import torch
import numpy as np

import matplotlib
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
from DiffNet.networks.wgan import GoodNetwork
from DiffNet.DiffNetFEM import DiffNet3DFEM

def load_raw(fileName, **kwargs):
    def _configParser(cName):
        with open(cName, 'r') as configFile:
            configFile.readline()
            line1 = configFile.readline().split()
            bBoxMin = np.array([float(i) for i in line1])
            line2 = configFile.readline().split()
            bBoxMax = np.array([float(i) for i in line2])
            line3 = configFile.readline().split()
            numDiv = np.array([int(i) for i in line3])
            line4 = configFile.readline().split()
            gridSize = np.array([float(i) for i in line4])
            inOutVoxelNum = int(configFile.readline())
            boundaryVoxelNum = int(configFile.readline())
        return bBoxMax, bBoxMin, numDiv, gridSize

    inOutName = fileName + 'inouts.raw'
    configName = fileName + 'VoxelConfig.txt'
    inOut = np.fromfile(inOutName, dtype=np.dtype('uint8'))
    inOut = (inOut / 254.0 > 0.25).astype(float)
    bBoxMax, bBoxMin, numDiv, gridSize = _configParser(configName)
    inOut = np.reshape(inOut, numDiv, order='F')
    return inOut, numDiv, gridSize, bBoxMin

class VoxelIMBackRAW(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, filename, domain_size=64):
        """
        Initialization
        """

        vox, _, _ , _  = load_raw(filename)
        domain = np.ones((domain_size, domain_size, domain_size))
        domain[32:32+vox.shape[0],32:32+vox.shape[1],32:32+vox.shape[2]] = 1 - vox
        self.domain = domain

        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros_like(self.domain)
        self.bc1[(self.domain).astype('bool')] = 1
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros_like(self.domain)
        self.bc2[-1,:,:] = 1
        self.bc2[0,:,:] = 1
        self.bc2[:,0,:] = 1
        self.bc2[:,-1,:] = 1
        self.bc2[:,:,0] = 1
        self.bc2[:,:,-1] = 1
        self.n_samples = 100

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = np.zeros_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

class Poisson(DiffNet3DFEM):
    """docstring for Poisson"""
    def __init__(self, network, dataset, **kwargs):
        super(Poisson, self).__init__(network, dataset, **kwargs)
        x = np.linspace(0,1,self.domain_size)
        y = np.linspace(0,1,self.domain_size)
        z = np.linspace(0,1,self.domain_size)
        xx, yy, zz = np.meshgrid(x,y,z)

    def loss(self, u, inputs_tensor, forcing_tensor):

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:,:]
        bc1 = inputs_tensor[:,1:2,:,:,:]
        bc2 = inputs_tensor[:,2:3,:,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)


        nu_gp = self.gauss_pt_evaluation(nu)
        f_gp = self.gauss_pt_evaluation(f)
        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)
        u_z_gp = self.gauss_pt_evaluation_der_z(u)

        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(nu_gp)
        res_elmwise = transformation_jacobian * ((0.5 * nu_gp * (u_x_gp**2 + u_y_gp**2  + u_y_gp**2) - (u_gp * f_gp)))
        res_elmwise = torch.sum(res_elmwise, 1) 

        loss = torch.mean(res_elmwise)
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        return self.network[0], inputs_tensor, forcing_tensor

    # def configure_optimizers(self):
    #     """
    #     Configure optimizer for network parameters
    #     """
    #     lr = self.learning_rate
    #     opts = [torch.optim.Adam(self.network, lr=lr)]
    #     return opts, []

    # def training_step(self, batch, batch_idx, optimizer_idx):
    # # def training_step(self, batch, batch_idx):
    #     if self.current_epoch % 2:
    #         opt = 1
    #     else:
    #         opt = 0
    #     if optimizer_idx==opt:
    #         u, inputs_tensor, forcing_tensor = self.forward(batch)
    #         loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
    #         # self.log('PDE_loss', loss_val.item())
    #         # self.log('loss', loss_val.item())
    #     else:
    #         loss_val = torch.zeros((1), requires_grad=True)
    #     return {"loss": loss_val}

    def training_step(self, batch, batch_idx):
    # def training_step(self, batch, batch_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
        # self.log('PDE_loss', loss_val.item())
        # self.log('loss', loss_val.item())
        return {"loss": loss_val}

    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs["loss"]
        return training_step_outputs

    def configure_optimizers(self):
        lr = self.learning_rate
        # opts = [torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        opts = [torch.optim.Adam(self.network, lr=lr)]
        # opts = [torch.optim.Adam(self.network, lr=lr), torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
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

        f = forcing_tensor.squeeze().detach().cpu() # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)


        k = nu.squeeze().detach().cpu().numpy()
        u = u.squeeze().detach().cpu().numpy()
        im0 = axs[0].imshow(k[:,:,48],cmap='jet')
        fig.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(u[:,:,48],cmap='jet', vmin=0.0, vmax=1.0)
        fig.colorbar(im1, ax=axs[1])
        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')
        os.makedirs(os.path.join(self.logger[0].log_dir, 'viz_%s'%self.current_epoch))
        knorm = k/k.max()
        (((knorm*255.0).astype('uint8')).flatten(order='F')).tofile(os.path.join(self.logger[0].log_dir,'viz_%s'%self.current_epoch,'diffusivity.raw'))
        unorm = np.clip(u,0.0,1.0)
        (((unorm*255.0).astype('uint8')).flatten(order='F')).tofile(os.path.join(self.logger[0].log_dir,'viz_%s'%self.current_epoch,'u.raw'))


def main():
    filename = 'Engine'
    dataset = VoxelIMBackRAW(filename, domain_size=196)
    u_tensor = np.ones_like(dataset.domain)
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    basecase = Poisson(network, dataset, batch_size=1, domain_size=196, learning_rate=0.01, nsd=3)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="voxel-humvee")
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