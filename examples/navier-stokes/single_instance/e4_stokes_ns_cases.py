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
from b1_stokes_ns_resmin_base import Stokes_NS_Base_2D
from DiffNet.networks.autoencoders import AE
# from e1_stokes_base_resmin import Stokes2D

from pytorch_lightning.callbacks.base import Callback

torch.set_printoptions(precision=10)

class OptimSwitchLBFGS(Callback):
    def __init__(self, epochs=50):
        self.switch_epoch = epochs
        self.print_declaration = False

    def on_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.switch_epoch:
            if not self.print_declaration:
                print("======================Switching to LBFGS after {} epochs ======================".format(self.switch_epoch))
                self.print_declaration = True
            opts = [torch.optim.LBFGS(pl_module.net_u.parameters(), lr=pl_module.learning_rate, max_iter=5),
                        torch.optim.LBFGS(pl_module.net_v.parameters(), lr=pl_module.learning_rate, max_iter=5),
                        # torch.optim.LBFGS(pl_module.net_p.parameters(), lr=pl_module.learning_rate, max_iter=5),
                        torch.optim.Adam(pl_module.net_p.parameters(), lr=pl_module.learning_rate)
                        ]
            trainer.optimizers = opts

class LDC_Dataset(data.Dataset):
    'PyTorch dataset for Stokes_MMS_Dataset'
    def __init__(self, domain_lengths=(1.,1.), domain_sizes=(32,32), Re=1):
        """
        Initialization
        """

        x = np.linspace(0, domain_lengths[0], domain_sizes[0])
        y = np.linspace(0, domain_lengths[1], domain_sizes[1])

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
        self.nu = np.ones_like(self.x) / self.Re

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.x, self.y, self.bc1, self.bc2, self.bc3, self.nu])

        forcing = np.ones_like(self.x)*(1/self.Re)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

class Stokes_NS_LDC(Stokes_NS_Base_2D):
    """docstring for Stokes_NS_LDC"""
    def __init__(self, network, dataset, **kwargs):
        super(Stokes_NS_LDC, self).__init__(network, dataset, **kwargs)
        self.plot_frequency = kwargs.get('plot_frequency', 1)

        self.net_u = network[0]
        self.net_v = network[1]
        self.net_p = network[2]

        self.Re = self.dataset.Re
        self.viscosity = 1. / self.Re
        self.pspg_param = self.h**2 * self.Re / 12.

        ue, ve, pe = self.exact_solution(self.dataset.x, self.dataset.y)
        self.u_exact = torch.FloatTensor(ue)
        self.v_exact = torch.FloatTensor(ve)
        self.p_exact = torch.FloatTensor(pe)

        fx_gp, fy_gp = self.forcing(self.xgp, self.ygp)
        self.fx_gp = torch.FloatTensor(fx_gp)
        self.fy_gp = torch.FloatTensor(fy_gp)

        u_bc = np.zeros_like(self.dataset.x); u_bc[-1,:] = 1. - 16. * (self.dataset.x[-1,:]-0.5)**4
        v_bc = np.zeros_like(self.dataset.x)
        p_bc = np.zeros_like(self.dataset.x)

        self.u_bc = torch.FloatTensor(u_bc)
        self.v_bc = torch.FloatTensor(v_bc)
        self.p_bc = torch.FloatTensor(p_bc)

        if self.eq_type == 'stokes':
            numerical = np.loadtxt('ns-ldc-numerical-results/midline_cuts_Re1_regularized_128x128.txt', delimiter=",", skiprows=1)
        elif self.eq_type == 'ns':
            numerical = np.loadtxt('ns-ldc-numerical-results/midline_cuts_Re100_regularized_128x128.txt', delimiter=",", skiprows=1)
        self.midline_X = numerical[:,0]
        self.midline_Y = numerical[:,0]
        self.midline_U = numerical[:,1]
        self.midline_V = numerical[:,2]
        self.topline_P = numerical[:,3]

    def plot_contours(self, u, v, p, u_x_gp, v_y_gp):
        fig, axs = plt.subplots(3, 3, figsize=(4*3,2.4*3),
                            subplot_kw={'aspect': 'auto'}, squeeze=True)

        for i in range(axs.shape[0]-1):
            for j in range(axs.shape[1]):
                axs[i,j].set_xticks([])
                axs[i,j].set_yticks([])
        
        div_gp = u_x_gp + v_y_gp
        div_elmwise = torch.sum(div_gp, 0)
        div_total = torch.sum(div_elmwise)

        interp_method = 'bilinear'
        im0 = axs[0,0].imshow(u,cmap='jet', origin='lower', interpolation=interp_method)
        fig.colorbar(im0, ax=axs[0,0]); axs[0,0].set_title(r'$u_x$')
        im1 = axs[0,1].imshow(v,cmap='jet',origin='lower', interpolation=interp_method)
        fig.colorbar(im1, ax=axs[0,1]); axs[0,1].set_title(r'$u_y$')
        im2 = axs[0,2].imshow(p,cmap='jet',origin='lower', interpolation=interp_method)
        fig.colorbar(im2, ax=axs[0,2]); axs[0,2].set_title(r'$p$')

        im3 = axs[1,0].imshow(div_elmwise,cmap='jet',origin='lower', interpolation=interp_method)
        fig.colorbar(im3, ax=axs[1,0]); axs[1,0].set_title(r'$\int(\nabla\cdot u) d\Omega = $' + '{:.3e}'.format(div_total.item()))
        im4 = axs[1,1].imshow((u**2 + v**2)**0.5,cmap='jet',origin='lower', interpolation=interp_method)
        fig.colorbar(im4, ax=axs[1,1]); axs[1,1].set_title(r'$\sqrt{u_x^2+u_y^2}$')
        x = np.linspace(0, 1, self.domain_sizeX)
        y = np.linspace(0, 1, self.domain_sizeY)
        xx , yy = np.meshgrid(x, y)
        im5 = axs[1,2].streamplot(xx, yy, u, v, color='k', cmap='jet'); axs[1,2].set_title("Streamlines")

        mid_idxX = int(self.domain_sizeX/2)
        mid_idxY = int(self.domain_sizeY/2)
        im = axs[2,0].plot(self.dataset.y[:,mid_idxX], u[:,mid_idxX],label='DiffNet')
        im = axs[2,0].plot(self.midline_Y,self.midline_U,label='Numerical')
        axs[2,0].set_xlabel('y'); axs[2,0].legend(); axs[2,0].set_title(r'$u_x @ x=0.5$')
        im = axs[2,1].plot(self.dataset.x[mid_idxY,:], v[mid_idxY,:],label='DiffNet')
        im = axs[2,1].plot(self.midline_X,self.midline_V,label='Numerical')
        axs[2,1].set_xlabel('x'); axs[2,1].legend(); axs[2,1].set_title(r'$u_y @ y=0.5$')
        im = axs[2,2].plot(self.dataset.x[-1,:], p[-1,:],label='DiffNet')
        im = axs[2,2].plot(self.midline_X,self.topline_P,label='Numerical')
        axs[2,2].set_xlabel('x'); axs[2,2].legend(); axs[2,2].set_title(r'$p @ y=1.0$')

        fig.suptitle("Re = {:.1f}, Nx = {}, Ny = {}, LR = {:.1e}, epochs = {}".format(self.Re, self.domain_sizeX, self.domain_sizeY, self.learning_rate, self.current_epoch), fontsize=12)

        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')
        

def main():
    lx = 1.
    ly = 1.
    Nx = 32
    Ny = 32
    domain_size = Nx
    Re = 100.
    dir_string = "ns_ldc_NN"
    max_epochs = 1001
    plot_frequency = 20
    LR = 5e-3 #1.e-2 #
    opt_switch_epochs = max_epochs
    load_from_prev = False
    load_version_id = 37

    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    xx , yy = np.meshgrid(x, y)

    # eq_type = 'stokes'
    eq_type = 'ns'
    mapping_type = 'no_network'
    # mapping_type = 'network'

    dataset = LDC_Dataset(domain_lengths=(lx,ly), domain_sizes=(Nx,Ny), Re=Re)
    if load_from_prev:
        print("LOADING FROM PREVIOUS VERSION: ", load_version_id)
        case_dir = './ns_ldc_NN/version_'+str(load_version_id)
        net_u = torch.load(os.path.join(case_dir, 'net_u.pt'))
        net_v = torch.load(os.path.join(case_dir, 'net_v.pt'))
        net_p = torch.load(os.path.join(case_dir, 'net_p.pt'))
    else:
        print("INITIALIZING PARAMETERS TO ZERO")
        if mapping_type == 'no_network':
            v1 = np.zeros_like(dataset.x)
            v2 = np.zeros_like(dataset.x)
            p  = np.zeros_like(dataset.x)
            u_tensor = np.expand_dims(np.array([v1,v2,p]),0)
            net_u = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor[:,0:1,:,:]), requires_grad=True)])
            net_v = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor[:,1:2,:,:]), requires_grad=True)])
            net_p = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor[:,2:3,:,:]), requires_grad=True)])
        elif mapping_type == 'network':
            net_u = AE(in_channels=1, out_channels=1, dims=domain_size, n_downsample=3)
            net_v = AE(in_channels=1, out_channels=1, dims=domain_size, n_downsample=3)
            net_p = AE(in_channels=1, out_channels=1, dims=domain_size, n_downsample=3)

    network = (net_u, net_v, net_p)
    basecase = Stokes_NS_LDC(network, dataset, eq_type=eq_type, mapping_type=mapping_type, domain_lengths=(lx,ly), domain_sizes=(Nx,Ny), batch_size=1, fem_basis_deg=1, learning_rate=LR, plot_frequency=plot_frequency)

    # Initialize trainer
    logger = pl.loggers.TensorBoardLogger('.', name=dir_string)
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    lbfgs_switch = OptimSwitchLBFGS(epochs=opt_switch_epochs)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping,lbfgs_switch],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=max_epochs, deterministic=True, profiler="simple")

    # Training
    trainer.fit(basecase)
    # Save network
    torch.save(basecase.net_u, os.path.join(logger.log_dir, 'net_u.pt'))
    torch.save(basecase.net_v, os.path.join(logger.log_dir, 'net_v.pt'))
    torch.save(basecase.net_p, os.path.join(logger.log_dir, 'net_p.pt'))


if __name__ == '__main__':
    main()