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



class LDC(data.Dataset):
    'PyTorch dataset for LDC'
    def __init__(self, domain_size=64, Re=400):
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
        self.bc1[:,0:1] = 1.0
        self.bc1[:,-1:] = 1.0
        self.bc1[0:1,:] = 1.0

        self.bc2 = np.zeros_like(xx)
        self.bc2_val = np.zeros_like(xx)
        self.bc2[-1:,:] = 1.0

        self.bc3 = np.zeros_like(xx)
        self.bc3[0:1,0:1] = 1.0

        self.Re = Re
        self.n_samples = 100


    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.x, self.bc1, self.bc2, self.bc3])

        forcing = np.ones_like(self.x)*(1/self.Re)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)


class Stokes(DiffNet2DFEM):
    """docstring for Eiqonal"""
    def __init__(self, network, dataset, **kwargs):
        super(Stokes, self).__init__(network, dataset, **kwargs)

    def loss(self, pred, inputs_tensor, forcing_tensor):

        f = forcing_tensor # renaming variable
        
        u = pred[:,0:1,:,:]
        v = pred[:,1:2,:,:]
        p = pred[:,2:3,:,:]

        # extract diffusivity and boundary conditions here
        x = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]
        bc3 = inputs_tensor[:,3:4,:,:]

        # apply boundary conditions
        u = torch.where(bc1>=0.5, u*0.0, u)
        u = torch.where(bc2>=0.5, u*0.0 + 1.0, u)
        # u = torch.where(bc2>=0.5, u*0.0 + 4.0*x*(1-x), u)

        v = torch.where(torch.logical_or((bc1>=0.5),(bc2>=0.5)), v*0.0, v)

        p = torch.where(bc3>=0.5, p*0.0, p)

        u_gp = self.gauss_pt_evaluation(u)
        v_gp = self.gauss_pt_evaluation(v)
        p_gp = self.gauss_pt_evaluation(p)
        p_x_gp = self.gauss_pt_evaluation_der_x(p)
        f_gp = self.gauss_pt_evaluation(f)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)
        v_x_gp = self.gauss_pt_evaluation_der_x(v)
        v_y_gp = self.gauss_pt_evaluation_der_y(v)

        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(u_gp)
        advec_term = u_gp*u_gp*u_x_gp + u_gp*v_gp*u_y_gp + u_gp*v_gp*v_x_gp + v_gp*v_gp*v_y_gp
        stokes_term = (u_x_gp**2 + u_y_gp**2 + v_x_gp**2 + v_y_gp**2)*f_gp
        press_term = p_gp*(u_x_gp + v_y_gp)
        res_elmwise1 = transformation_jacobian * (advec_term + stokes_term - press_term)**2
        # res_elmwise1 = transformation_jacobian * ((u_x_gp**2 + v_y_gp**2)*f_gp - p_gp*(u_x_gp + v_y_gp))**2
        res_elmwise2 = transformation_jacobian * ((p_gp*(u_x_gp + v_y_gp))**2 + 0.005*p_x_gp**2)

        res_elmwise = 1000*torch.sum(res_elmwise1, 1) + torch.sum(res_elmwise2, 1) 
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
        # opts = [torch.optim.Adam(self.network, lr=lr)]
        schd = []
        # schd = [torch.optim.lr_scheduler.ExponentialLR(opts[0], gamma=0.7)]
        return opts, schd


    def on_epoch_end(self):
        fig, axs = plt.subplots(1, 6, figsize=(2*6,1.2),
                            subplot_kw={'aspect': 'auto'}, squeeze=True)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        self.network.eval()
        inputs, forcing = self.dataset[0]

        pred, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

        f = forcing_tensor # renaming variable
        
        u = pred[:,0:1,:,:]
        v = pred[:,1:2,:,:]
        p = pred[:,2:3,:,:]

        # extract diffusivity and boundary conditions here
        x = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]
        bc3 = inputs_tensor[:,3:4,:,:]

        # apply boundary conditions
        u = torch.where(bc1>=0.05, u*0.0, u)
        u = torch.where(bc2>=0.05, u*0.0 + 1.0, u)

        v = torch.where(torch.logical_or((bc1>=0.5),(bc2>=0.5)), v*0.0, v)
        p = torch.where(bc3>=0.5, p*0.0, p)

        u_x = self.gauss_pt_evaluation_der_x(u)[:,0,:,:].squeeze().detach().cpu()
        v_y = self.gauss_pt_evaluation_der_y(v)[:,0,:,:].squeeze().detach().cpu()

        u = u.squeeze().detach().cpu()
        v = v.squeeze().detach().cpu()
        p = p.squeeze().detach().cpu()
        bc1 = bc1.squeeze().detach().cpu()
        bc2 = bc2.squeeze().detach().cpu()

        div = u_x + v_y

        im0 = axs[0].contourf(u,cmap='jet',levels=5)
        fig.colorbar(im0, ax=axs[0])
        im1 = axs[1].contourf(v,cmap='jet',levels=5)
        fig.colorbar(im1, ax=axs[1])  
        im2 = axs[2].contourf(p,cmap='jet',levels=5)
        fig.colorbar(im2, ax=axs[2])
        x = np.linspace(0, 1, u.shape[0])
        y = np.linspace(0, 1, u.shape[1])

        im3 = axs[3].contourf(np.log10(abs(div)),cmap='jet',levels=5)
        fig.colorbar(im3, ax=axs[3])  
        im4 = axs[4].contourf((u**2 + v**2)**0.5,levels=5,cmap='jet')
        fig.colorbar(im4, ax=axs[4])

        xx , yy = np.meshgrid(x, y)
        im5 = axs[5].streamplot(xx, yy, u, v, color='k', cmap='jet')

        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    u_tensor = np.zeros((1,3,256,256))
    # u_tensor = np.random.rand(1,3,256,256)
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    dataset = LDC(domain_size=256)
    basecase = Stokes(network, dataset, domain_size=256, batch_size=1, fem_basis_deg=1)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="lid_driven_cavity")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=1000, deterministic=True, profiler="simple")

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