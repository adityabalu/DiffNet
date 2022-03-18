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

class Stokes_MMS_Dataset(data.Dataset):
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

class Stokes_LDC(DiffNet2DFEM):
    """docstring for Stokes_LDC"""
    def __init__(self, network, dataset, **kwargs):
        super(Stokes_LDC, self).__init__(network[0], dataset, **kwargs)
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

        self.u_bc = torch.FloatTensor(self.u_exact)
        self.v_bc = torch.FloatTensor(self.v_exact)
        self.p_bc = torch.FloatTensor(self.p_exact)

        numerical = np.loadtxt('stokes-ldc-numerical-results/midline_cuts_Re1_regularized_128x128.txt', delimiter=",", skiprows=1)
        self.midline_X = numerical[:,0]
        self.midline_Y = numerical[:,0]
        self.midline_U = numerical[:,1]
        self.midline_V = numerical[:,2]
        self.topline_P = numerical[:,3]

    def exact_solution(self, x, y):
        print("exact_solution -- MMS class called")
        pi = math.pi
        sin = np.sin
        cos = np.cos        
        u_exact =  sin(pi*x)*cos(pi*y)
        v_exact = -cos(pi*x)*sin(pi*y)
        p_exact =  sin(pi*x)*sin(pi*y)
        return u_exact, v_exact, p_exact

    def forcing(self, x, y):
        print("forcing -- MMS class called")
        pi = math.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp        
        fx =  2*pi**2*sin(pi*x)*cos(pi*y) + pi*sin(pi*y)*cos(pi*x)
        fy = -2*pi**2*sin(pi*y)*cos(pi*x) + pi*sin(pi*x)*cos(pi*y)
        return fx, fy

    def Q1_vector_assembly(self, Aglobal, Aloc_all):
        Aglobal[:,0, 0:-1, 0:-1] += Aloc_all[:,0, :, :]
        Aglobal[:,0, 0:-1, 1:  ] += Aloc_all[:,1, :, :]
        Aglobal[:,0, 1:  , 0:-1] += Aloc_all[:,2, :, :]
        Aglobal[:,0, 1:  , 1:  ] += Aloc_all[:,3, :, :]
        return Aglobal

    def calc_residuals(self, pred, inputs_tensor, forcing_tensor):
        visco = self.viscosity

        N_values = self.Nvalues.type_as(pred[0])
        dN_x_values = self.dN_x_values.type_as(pred[0])
        dN_y_values = self.dN_y_values.type_as(pred[0])
        gpw = self.gpw.type_as(pred[0])

        f1 = self.fx_gp.type_as(pred[0])
        f2 = self.fy_gp.type_as(pred[0])

        u_bc = self.u_bc.unsqueeze(0).unsqueeze(0).type_as(pred[0])
        v_bc = self.v_bc.unsqueeze(0).unsqueeze(0).type_as(pred[0])
        p_bc = self.p_bc.unsqueeze(0).unsqueeze(0).type_as(pred[0])


        f = forcing_tensor # renaming variable

        u_pred = pred[0] #[:,0:1,:,:]
        v_pred = pred[1] #[:,1:2,:,:]
        p_pred = pred[2] #[:,2:3,:,:]

        # extract diffusivity and boundary conditions here
        x = inputs_tensor[:,0:1,:,:]
        y = inputs_tensor[:,1:2,:,:]
        bc1 = inputs_tensor[:,2:3,:,:]
        bc2 = inputs_tensor[:,3:4,:,:]
        bc3 = inputs_tensor[:,4:5,:,:]

        # DERIVE NECESSARY VALUES
        trnsfrm_jac = (0.5*self.h)**2
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        # apply boundary conditions
        u_pred = torch.where(bc1>=0.5, u_bc, u_pred)
        v_pred = torch.where(bc2>=0.5, v_bc, v_pred)
        p_pred = torch.where(bc3>=0.5, p_bc, p_pred)
        
        u = self.gauss_pt_evaluation(u_pred)
        v = self.gauss_pt_evaluation(v_pred)
        p = self.gauss_pt_evaluation(p_pred)
        p_x = self.gauss_pt_evaluation_der_x(p_pred)
        p_y = self.gauss_pt_evaluation_der_y(p_pred)
        u_x = self.gauss_pt_evaluation_der_x(u_pred)
        u_y = self.gauss_pt_evaluation_der_y(u_pred)
        v_x = self.gauss_pt_evaluation_der_x(v_pred)
        v_y = self.gauss_pt_evaluation_der_y(v_pred)

        # CALCULATION STARTS
        # lhs
        W_U1x = N_values*u_x
        W_U2y = N_values*v_y
        Wx_U1x = dN_x_values*u_x
        Wy_U1y = dN_y_values*u_y
        Wx_U2x = dN_x_values*v_x
        Wy_U2y = dN_y_values*v_y
        Wx_P = dN_x_values*p
        Wy_P = dN_y_values*p
        Wx_Px = dN_x_values*p_x
        Wy_Py = dN_y_values*p_y
        # rhs
        W_F1 = N_values*f1
        W_F2 = N_values*f2

        # integrated values on lhs & rhs
        temp1 = visco*(Wx_U1x+Wy_U1y) - Wx_P - W_F1
        temp2 = visco*(Wx_U2x+Wy_U2y) - Wy_P - W_F2
        temp3 = W_U1x+W_U2y + self.pspg_param*(Wx_Px+Wy_Py)

        # unassembled residual
        R_split_1 = torch.sum(temp1*JxW, 2) # sum across all GP
        R_split_2 = torch.sum(temp2*JxW, 2) # sum across all GP
        R_split_3 = torch.sum(temp3*JxW, 2) # sum across all GP

        # assembly
        R1 = torch.zeros_like(u_pred); R1 = self.Q1_vector_assembly(R1, R_split_1)
        R2 = torch.zeros_like(v_pred); R2 = self.Q1_vector_assembly(R2, R_split_2)
        R3 = torch.zeros_like(p_pred); R3 = self.Q1_vector_assembly(R3, R_split_3)

        # add boundary conditions to R <---- this step is very important
        R1 = torch.where(bc1>=0.5, u_bc, R1)
        R2 = torch.where(bc2>=0.5, v_bc, R2)
        R3 = torch.where(bc3>=0.5, p_bc, R3)

        return R1, R2, R3

    def loss(self, pred, inputs_tensor, forcing_tensor):
        R1, R2, R3 = self.calc_residuals(pred, inputs_tensor, forcing_tensor)
        # loss = torch.norm(R1, 'fro') + torch.norm(R2, 'fro') + torch.norm(R3, 'fro')
        return torch.norm(R1, 'fro'), torch.norm(R2, 'fro'), torch.norm(R3, 'fro')

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        return self.net_u[0], self.net_v[0], self.net_p[0], inputs_tensor, forcing_tensor

    def training_step(self, batch, batch_idx, optimizer_idx):
        u, v, p, inputs_tensor, forcing_tensor = self.forward(batch)
        loss_vals = self.loss((u, v, p), inputs_tensor, forcing_tensor)
        self.log('loss_u', loss_vals[0].item())
        self.log('loss_v', loss_vals[1].item())
        self.log('loss_p', loss_vals[2].item())
        return {"loss": loss_vals[optimizer_idx]}

    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs["loss"]
        return training_step_outputs

    def configure_optimizers(self):
        lr = self.learning_rate
        # opts = [torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        opts = [torch.optim.Adam(self.net_u, lr=lr), torch.optim.Adam(self.net_v, lr=lr), torch.optim.Adam(self.net_p, lr=lr)]
        return opts, []

    def on_epoch_end(self):
        # self.network.eval()
        self.net_u.eval()
        self.net_v.eval()
        self.net_p.eval()
        inputs, forcing = self.dataset[0]
        u, v, p, u_x_gp, v_y_gp = self.do_query(inputs, forcing)

        u = u.squeeze().detach().cpu()
        v = v.squeeze().detach().cpu()
        p = p.squeeze().detach().cpu()
        u_x_gp = u_x_gp.squeeze().detach().cpu()
        v_y_gp = v_y_gp.squeeze().detach().cpu()

        if self.current_epoch % self.plot_frequency == 0:
            self.plot_contours(u, v, p, u_x_gp, v_y_gp)

    def do_query(self, inputs, forcing):
        u, v, p, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.net_u.parameters())), forcing.unsqueeze(0).type_as(next(self.net_u.parameters()))))

        f = forcing_tensor # renaming variable

        # extract diffusivity and boundary conditions here
        x = inputs_tensor[:,0:1,:,:]
        y = inputs_tensor[:,1:2,:,:]
        bc1 = inputs_tensor[:,2:3,:,:]
        bc2 = inputs_tensor[:,3:4,:,:]
        bc3 = inputs_tensor[:,4:5,:,:]

        # apply boundary conditions
        u_bc = self.u_bc.unsqueeze(0).unsqueeze(0).type_as(u)
        v_bc = self.v_bc.unsqueeze(0).unsqueeze(0).type_as(u)
        p_bc = self.p_bc.unsqueeze(0).unsqueeze(0).type_as(u)

        u = torch.where(bc1>=0.5, u_bc, u)
        v = torch.where(bc2>=0.5, v_bc, v)
        p = torch.where(bc3>=0.5, p_bc, p)

        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        v_y_gp = self.gauss_pt_evaluation_der_y(v)

        return u, v, p, u_x_gp, v_y_gp

    def plot_contours(self, u, v, p, u_x_gp, v_y_gp):
        fig, axs = plt.subplots(3, 3, figsize=(4*3,2.4*3),
                            subplot_kw={'aspect': 'auto'}, squeeze=True)

        for ax_row in axs:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([]) 
        
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
        x = np.linspace(0, 1, u.shape[0])
        y = np.linspace(0, 1, u.shape[1])

        im = axs[1,0].imshow(u-self.u_exact,cmap='jet', origin='lower', interpolation=interp_method)
        fig.colorbar(im, ax=axs[1,0]); axs[1,0].set_title(r'$(u_x - u_x^{exact})$')
        im = axs[1,1].imshow(v-self.v_exact,cmap='jet',origin='lower', interpolation=interp_method)
        fig.colorbar(im, ax=axs[1,1]); axs[1,1].set_title(r'$(u_y - u_y^{exact})$')
        im = axs[1,2].imshow(p-self.p_exact,cmap='jet',origin='lower', interpolation=interp_method)
        fig.colorbar(im, ax=axs[1,2]); axs[1,2].set_title(r'$(p - p^{exact})$')

        im3 = axs[2,0].imshow(div_elmwise,cmap='jet',origin='lower', interpolation=interp_method)
        fig.colorbar(im3, ax=axs[2,0]); axs[2,0].set_title(r'$\int(\nabla\cdot u) d\Omega = $' + '{:.3e}'.format(div_total.item()))
        im4 = axs[2,1].imshow((u**2 + v**2)**0.5,cmap='jet',origin='lower', interpolation=interp_method)
        fig.colorbar(im4, ax=axs[2,1]); axs[2,1].set_title(r'$\sqrt{u_x^2+u_y^2}$')

        xx , yy = np.meshgrid(x, y)
        im5 = axs[2,2].streamplot(xx, yy, u, v, color='k', cmap='jet'); axs[2,2].set_title("Streamlines")

        fig.suptitle("Re = {:.1f}, N = {}, LR = {:.1e}".format(self.Re, self.domain_size, self.learning_rate), fontsize=12)

        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    domain_size = 16
    Re = 1.
    dir_string = "stokes_mms_loss2"
    LR = 3e-4
    max_epochs = 201
    opt_switch_epochs = max_epochs
    plot_frequency = 20

    x = np.linspace(0, 1, domain_size)
    y = np.linspace(0, 1, domain_size)
    xx , yy = np.meshgrid(x, y)

    dataset = Stokes_MMS_Dataset(domain_size=domain_size, Re=Re)
    v1 = np.zeros_like(dataset.x)
    v2 = np.zeros_like(dataset.x)
    p  = np.zeros_like(dataset.x)
    u_tensor = np.expand_dims(np.array([v1,v2,p]),0)
    
    # network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    net_u = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor[:,0:1,:,:]), requires_grad=True)])
    net_v = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor[:,1:2,:,:]), requires_grad=True)])
    net_p = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor[:,2:3,:,:]), requires_grad=True)])
    network = (net_u, net_v, net_p)

    basecase = Stokes_LDC(network, dataset, domain_size=domain_size, batch_size=1, fem_basis_deg=1, learning_rate=LR, plot_frequency=plot_frequency)

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