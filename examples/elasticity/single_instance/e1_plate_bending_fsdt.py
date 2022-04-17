import os
import sys
import math
import json
import torch
import numpy as np

import scipy.io
from scipy import ndimage
import matplotlib
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

class Elastic_FSDT_Dataset(data.Dataset):
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

        self.bc3 = self.bc1

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

class Elastic_FSDT(DiffNet2DFEM):
    """docstring for Elastic_FSDT"""
    def __init__(self, network, dataset, **kwargs):
        super(Elastic_FSDT, self).__init__(network[0], dataset, **kwargs)
        self.plot_frequency = kwargs.get('plot_frequency', 1)

        self.net_w = network[0]
        self.net_phi_x = network[1]
        self.net_phi_y = network[2]

        self.Re = self.dataset.Re
        self.pspg_param = self.h**2 * self.Re / 12.

        fx_gp, fy_gp = self.forcing(self.xgp, self.ygp)
        self.fx_gp = torch.FloatTensor(fx_gp)
        self.fy_gp = torch.FloatTensor(fy_gp)

        ###########################################
        # Define and initialize boundary conditions
        # Insert code here
        self.w_bc = torch.FloatTensor(np.zeros_like(self.dataset.x))
        self.phi_x_bc = torch.FloatTensor(np.zeros_like(self.dataset.x))
        self.phi_y_bc = torch.FloatTensor(np.zeros_like(self.dataset.x))
        ########################################### 

    def forcing(self, x, y):
        print("forcing -- LDC class called")
        fx = np.ones_like(x)
        fy = np.ones_like(x)
        return fx, fy


    def Q1_vector_assembly(self, Aglobal, Aloc_all):
        Aglobal[:,0, 0:-1, 0:-1] += Aloc_all[:,0, :, :]
        Aglobal[:,0, 0:-1, 1:  ] += Aloc_all[:,1, :, :]
        Aglobal[:,0, 1:  , 0:-1] += Aloc_all[:,2, :, :]
        Aglobal[:,0, 1:  , 1:  ] += Aloc_all[:,3, :, :]
        return Aglobal

    def calc_residuals(self, pred, inputs_tensor, forcing_tensor):

        hx = self.h
        hy = self.h

        N_values = self.Nvalues.type_as(pred[0])
        dN_x_values = self.dN_x_values.type_as(pred[0])
        dN_y_values = self.dN_y_values.type_as(pred[0])
        gpw = self.gpw.type_as(pred[0])

        f1 = self.fx_gp.type_as(pred[0])
        f2 = self.fy_gp.type_as(pred[0])

        w_bc = self.w_bc.unsqueeze(0).unsqueeze(0).type_as(pred[0])
        phi_x_bc = self.phi_x_bc.unsqueeze(0).unsqueeze(0).type_as(pred[0])
        phi_y_bc = self.phi_y_bc.unsqueeze(0).unsqueeze(0).type_as(pred[0])


        f = forcing_tensor # renaming variable

        w_pred = pred[0] #[:,0:1,:,:]
        phi_x_pred = pred[1] #[:,1:2,:,:]
        phi_y_pred = pred[2] #[:,2:3,:,:]

        # extract diffusivity and boundary conditions here
        x = inputs_tensor[:,0:1,:,:]
        y = inputs_tensor[:,1:2,:,:]
        bc1 = inputs_tensor[:,2:3,:,:]
        bc2 = inputs_tensor[:,3:4,:,:]
        bc3 = inputs_tensor[:,4:5,:,:]

        # DERIVE NECESSARY VALUES
        trnsfrm_jac = (0.5*hx)*(0.5*hy)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        # apply boundary conditions
        w_pred = torch.where(bc2>=0.5, w_bc, w_pred)
        phi_x_pred = torch.where(bc2>=0.5, phi_x_bc, phi_x_pred)
        phi_y_pred = torch.where(bc2>=0.5, phi_y_bc, phi_y_pred)

        # Init and compute constants - eqn 
        E = 1.0
        v = 0.25
        h = 0.1
        K_s = 1.0
        D_11 = (E * h**3) / (12 * (1 - v**2))
        D_22 = D_11
        D_12 = (E * v * h**3) / (12 * (1 - v**2))
        D_66 = (E * h**3) / (12 * (1 + v))
        A_44 = (E * h) / (2 * (1 + v))
        A_55 = (E * h) / (2 * (1 + v))

        # Compute quadrature pts
        w_gp = self.gauss_pt_evaluation(w_pred)
        w_x_gp = self.gauss_pt_evaluation_der_x(w_pred)
        w_y_gp = self.gauss_pt_evaluation_der_y(w_pred)
        phi_x_gp = self.gauss_pt_evaluation(phi_x_pred)
        phi_x_x_gp = self.gauss_pt_evaluation_der_x(phi_x_pred)
        phi_x_y_gp = self.gauss_pt_evaluation_der_y(phi_x_pred)
        phi_y_gp = self.gauss_pt_evaluation(phi_y_pred)
        phi_y_x_gp = self.gauss_pt_evaluation_der_x(phi_y_pred)
        phi_y_y_gp = self.gauss_pt_evaluation_der_y(phi_y_pred)

        # Init forcing function q
        q = torch.ones_like(w_gp)

        # Compute eqn 4
        Q_x = K_s * A_55 * (phi_x_gp + w_x_gp)
        Q_y = K_s * A_44 * (phi_y_gp + w_y_gp)
        M_xx = (D_11 * phi_x_x_gp) + (D_12 * phi_y_y_gp)
        M_yy = (D_12 * phi_x_x_gp) + (D_22 * phi_y_y_gp)
        M_xy = D_66 * (phi_x_y_gp + phi_y_x_gp)

        # Compute eqn 3
        # lhs
        lhs_3a = (dN_x_values * Q_x) + (dN_y_values * Q_y)
        lhs_3b = (dN_x_values * M_xx) + (dN_y_values * M_xy) + (N_values * Q_x)
        lhs_3c = (dN_x_values * M_xy) + (dN_y_values * M_yy) + (N_values * Q_y)
        # rhs
        rhs_3a = N_values * q

        temp1 = lhs_3a - rhs_3a
        temp2 = lhs_3b
        temp3 = lhs_3c

        # unassembled residual
        R_split_1 = torch.sum(temp1*JxW, 2) # sum across all GP
        R_split_2 = torch.sum(temp2*JxW, 2) # sum across all GP
        R_split_3 = torch.sum(temp3*JxW, 2) # sum across all GP

        # assembly
        R1 = torch.zeros_like(w_pred); R1 = self.Q1_vector_assembly(R1, R_split_1)
        R2 = torch.zeros_like(w_pred); R2 = self.Q1_vector_assembly(R2, R_split_2)
        R3 = torch.zeros_like(w_pred); R3 = self.Q1_vector_assembly(R3, R_split_3)

        # add boundary conditions to R <---- this step is very important
        R1 = torch.where(bc2>=0.5, w_bc, R1)
        R2 = torch.where(bc2>=0.5, phi_x_bc, R2)
        R3 = torch.where(bc2>=0.5, phi_y_bc, R3)

        return R1, R2, R3

    def loss(self, pred, inputs_tensor, forcing_tensor):
        R1, R2, R3 = self.calc_residuals(pred, inputs_tensor, forcing_tensor)
        return torch.norm(R1, 'fro'), torch.norm(R2, 'fro'), torch.norm(R3, 'fro')

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        return self.net_w[0], self.net_phi_x[0], self.net_phi_y[0], inputs_tensor, forcing_tensor

    def training_step(self, batch, batch_idx, optimizer_idx):
        w, phi_x, phi_y, inputs_tensor, forcing_tensor = self.forward(batch)
        loss_vals = self.loss((w, phi_x, phi_y), inputs_tensor, forcing_tensor)
        self.log('loss_w', loss_vals[0].item())
        self.log('loss_phi_x', loss_vals[1].item())
        self.log('loss_phi_y', loss_vals[2].item())
        return {"loss": loss_vals[optimizer_idx]}

    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs["loss"]
        return training_step_outputs

    def configure_optimizers(self):
        lr = self.learning_rate
        opts = [torch.optim.Adam(self.net_w, lr=lr), torch.optim.Adam(self.net_phi_x, lr=lr), torch.optim.Adam(self.net_phi_y, lr=lr)]
        return opts, []

    def on_epoch_end(self):
        # self.network.eval()
        self.net_w.eval()
        self.net_phi_x.eval()
        self.net_phi_y.eval()
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
        u, v, p, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.net_w.parameters())), forcing.unsqueeze(0).type_as(next(self.net_w.parameters()))))

        f = forcing_tensor # renaming variable

        # extract diffusivity and boundary conditions here
        x = inputs_tensor[:,0:1,:,:]
        y = inputs_tensor[:,1:2,:,:]
        bc1 = inputs_tensor[:,2:3,:,:]
        bc2 = inputs_tensor[:,3:4,:,:]
        bc3 = inputs_tensor[:,4:5,:,:]

        # apply boundary conditions
        u_bc = self.w_bc.unsqueeze(0).unsqueeze(0).type_as(u)
        v_bc = self.phi_x_bc.unsqueeze(0).unsqueeze(0).type_as(u)
        p_bc = self.phi_y_bc.unsqueeze(0).unsqueeze(0).type_as(u)

        u = torch.where(bc2>=0.5, u_bc, u)
        v = torch.where(bc2>=0.5, v_bc, v)
        p = torch.where(bc2>=0.5, p_bc, p)

        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        v_y_gp = self.gauss_pt_evaluation_der_y(v)

        return u, v, p, u_x_gp, v_y_gp

    def plot_contours(self, u, v, p, u_x_gp, v_y_gp):
        fig, axs = plt.subplots(2, 3, figsize=(4*3,2.4*3),
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
        fig.colorbar(im0, ax=axs[0,0]); axs[0,0].set_title(r'$w$')
        im1 = axs[0,1].imshow(v,cmap='jet',origin='lower', interpolation=interp_method)
        fig.colorbar(im1, ax=axs[0,1]); axs[0,1].set_title(r'$\phi_x$')
        im2 = axs[0,2].imshow(p,cmap='jet',origin='lower', interpolation=interp_method)
        fig.colorbar(im2, ax=axs[0,2]); axs[0,2].set_title(r'$\phi_y$')

        im3 = axs[1,0].plot(u[int(self.domain_size/2),:])
        im4 = axs[1,1].plot(u[:,int(self.domain_size/2)])

        if self.current_epoch % 50 == 0:
            plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    domain_size = 32
    Re = 100.
    dir_string = "Elastic_FSDT_" + str(domain_size)
    max_epochs = 250
    plot_frequency = 10
    LR = 4e-3
    opt_switch_epochs = max_epochs

    x = np.linspace(0, 1, domain_size)
    y = np.linspace(0, 1, domain_size)
    xx , yy = np.meshgrid(x, y)

    dataset = Elastic_FSDT_Dataset(domain_size=domain_size, Re=Re)
    w = np.zeros_like(dataset.x)
    phi_x = np.zeros_like(dataset.x)
    phi_y  = np.zeros_like(dataset.x)
    u_tensor = np.expand_dims(np.array([w,phi_x,phi_y]),0)
    
    # network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    net_w = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor[:,0:1,:,:]), requires_grad=True)])
    net_phi_x = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor[:,1:2,:,:]), requires_grad=True)])
    net_phi_y = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor[:,2:3,:,:]), requires_grad=True)])
    network = (net_w, net_phi_x, net_phi_y)

    basecase = Elastic_FSDT(network, dataset, domain_size=domain_size, batch_size=1, fem_basis_deg=1, learning_rate=LR, plot_frequency=plot_frequency)

    # Initialize trainer
    logger = pl.loggers.TensorBoardLogger('.', name=dir_string)
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    lbfgs_switch = OptimSwitchLBFGS(epochs=opt_switch_epochs)

    trainer = Trainer(gpus=[0],callbacks=[lbfgs_switch],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=max_epochs, deterministic=True, profiler="simple")

    # Training
    trainer.fit(basecase)
    # Save network
    torch.save(basecase.net_w, os.path.join(logger.log_dir, 'net_w.pt'))
    torch.save(basecase.net_phi_x, os.path.join(logger.log_dir, 'net_phi_x.pt'))
    torch.save(basecase.net_phi_y, os.path.join(logger.log_dir, 'net_phi_y.pt'))


if __name__ == '__main__':
    main()