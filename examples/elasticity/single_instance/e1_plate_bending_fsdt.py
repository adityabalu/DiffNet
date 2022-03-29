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

class Elastic_FSDT_Dataset(data.Dataset):
    'PyTorch dataset for Elastic_FSDT_Dataset'
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
        # bc2 for fixed boundaries
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

class Elastic_FSDT(DiffNet2DFEM):
    """docstring for Elastic_FSDT"""
    def __init__(self, network, dataset, **kwargs):
        super(Elastic_FSDT, self).__init__(network, dataset, **kwargs)

        self.plot_frequency = kwargs.get('plot_frequency', 1)

        self.Re = self.dataset.Re
        self.pspg_param = self.h**2 * self.Re / 12.

        ue, ve, pe = self.exact_solution(self.dataset.x, self.dataset.y)
        self.u_exact = torch.FloatTensor(ue)
        self.v_exact = torch.FloatTensor(ve)
        self.p_exact = torch.FloatTensor(pe)

        fx_gp, fy_gp = self.forcing(self.xgp, self.ygp)
        self.fx_gp = torch.FloatTensor(fx_gp)
        self.fy_gp = torch.FloatTensor(fy_gp)

        self.u_bc = self.u_exact
        self.v_bc = self.v_exact
        self.p_bc = self.p_exact

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
        # print("pred type = ", pred.type(), ", pred.shape = ", pred.shape)
        # exit()
        N_values = self.Nvalues.type_as(pred)
        dN_x_values = self.dN_x_values.type_as(pred)
        dN_y_values = self.dN_y_values.type_as(pred)
        gpw = self.gpw.type_as(pred)

        fx_gp = self.fx_gp.type_as(pred)
        fy_gp = self.fy_gp.type_as(pred)

        u_bc = self.u_bc.unsqueeze(0).unsqueeze(0).type_as(pred)
        v_bc = self.v_bc.unsqueeze(0).unsqueeze(0).type_as(pred)
        p_bc = self.p_bc.unsqueeze(0).unsqueeze(0).type_as(pred)


        f = forcing_tensor # renaming variable

        u = pred[:,0:1,:,:]
        v = pred[:,1:2,:,:]
        p = pred[:,2:3,:,:]

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
        u = torch.where(bc1>=0.5, u_bc, u)
        v = torch.where(bc2>=0.5, v_bc, v)
        p = torch.where(bc3>=0.5, p_bc, p)
        # u = torch.where(bc2>=0.5, u*0.0 + 1.0, u)
        # u = torch.where(bc2>=0.5, u*0.0 + 4.0*x*(1-x), u)

        # v = torch.where(torch.logical_or((bc1>=0.5),(bc2>=0.5)), v*0.0, v)

        # p = torch.where(bc3>=0.5, p*0.0, p)

        u_gp = self.gauss_pt_evaluation(u)
        v_gp = self.gauss_pt_evaluation(v)
        p_gp = self.gauss_pt_evaluation(p)
        p_x_gp = self.gauss_pt_evaluation_der_x(p)
        p_y_gp = self.gauss_pt_evaluation_der_y(p)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)
        v_x_gp = self.gauss_pt_evaluation_der_x(v)
        v_y_gp = self.gauss_pt_evaluation_der_y(v)

        # CALCULATION STARTS
        # lhs
        W_U1x = N_values*u_x_gp*JxW
        W_U2y = N_values*v_y_gp*JxW
        Wx_U1x = dN_x_values*u_x_gp*JxW
        Wy_U1y = dN_y_values*u_y_gp*JxW
        Wx_U2x = dN_x_values*v_x_gp*JxW
        Wy_U2y = dN_y_values*v_y_gp*JxW
        Wx_P = dN_x_values*p_gp*JxW
        Wy_P = dN_y_values*p_gp*JxW
        Wx_Px = dN_x_values*p_x_gp*JxW
        Wy_Py = dN_y_values*p_y_gp*JxW
        # rhs
        W_F1 = N_values*fx_gp*JxW
        W_F2 = N_values*fy_gp*JxW

        # integrated values on lhs & rhs
        temp1 = self.dataset.Re*(Wx_U1x+Wy_U1y) - Wx_P - W_F1
        temp2 = self.dataset.Re*(Wx_U2x+Wy_U2y) - Wy_P - W_F2
        temp3 = W_U1x+W_U2y + self.pspg_param*(Wx_Px+Wy_Py)

        # unassembled residual
        R_split_1 = torch.sum(temp1, 2) # sum across all GP
        R_split_2 = torch.sum(temp2, 2) # sum across all GP
        R_split_3 = torch.sum(temp3, 2) # sum across all GP

        # assembly
        R1 = torch.zeros_like(u); R1 = self.Q1_vector_assembly(R1, R_split_1)
        R2 = torch.zeros_like(u); R2 = self.Q1_vector_assembly(R2, R_split_2)
        R3 = torch.zeros_like(u); R3 = self.Q1_vector_assembly(R3, R_split_3)

        # add boundary conditions to R <---- this step is very important
        R1 = torch.where(bc1>=0.5, u_bc, R1)
        R2 = torch.where(bc2>=0.5, v_bc, R2)
        R3 = torch.where(bc3>=0.5, p_bc, R3)

        return R1, R2, R3

    def loss(self, pred, inputs_tensor, forcing_tensor):
        R1, R2, R3 = self.calc_residuals(pred, inputs_tensor, forcing_tensor)
        loss = torch.norm(R1, 'fro') + torch.norm(R2, 'fro') + torch.norm(R3, 'fro')
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
        # opts = [torch.optim.LBFGS(self.network, lr=0.0001, max_iter=1)]
        opts = [torch.optim.Adam(self.network, lr=lr)]
        schd = []
        # schd = [torch.optim.lr_scheduler.ExponentialLR(opts[0], gamma=0.7)]
        return opts, schd

    def on_epoch_end(self):
        self.network.eval()
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
        pred, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

        f = forcing_tensor # renaming variable

        u = pred[:,0:1,:,:]
        v = pred[:,1:2,:,:]
        p = pred[:,2:3,:,:]

        # extract diffusivity and boundary conditions here
        x = inputs_tensor[:,0:1,:,:]
        y = inputs_tensor[:,1:2,:,:]
        bc1 = inputs_tensor[:,2:3,:,:]
        bc2 = inputs_tensor[:,3:4,:,:]
        bc3 = inputs_tensor[:,4:5,:,:]

        # apply boundary conditions
        u_bc = self.u_bc.unsqueeze(0).unsqueeze(0).type_as(pred)
        v_bc = self.v_bc.unsqueeze(0).unsqueeze(0).type_as(pred)
        p_bc = self.p_bc.unsqueeze(0).unsqueeze(0).type_as(pred)
        # u = torch.where(bc1>=0.05, u*0.0, u)
        # u = torch.where(bc2>=0.05, u*0.0 + 1.0, u)

        # v = torch.where(torch.logical_or((bc1>=0.5),(bc2>=0.5)), v*0.0, v)
        # p = torch.where(bc3>=0.5, p*0.0, p)
        u = torch.where(bc1>=0.5, u_bc, u)
        v = torch.where(bc2>=0.5, v_bc, v)
        p = torch.where(bc3>=0.5, p_bc, p)

        u_x = self.gauss_pt_evaluation_der_x(u)
        v_y = self.gauss_pt_evaluation_der_y(v)

        return u, v, p, u_x, v_y

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
    dir_string = "stokes_mms"
    max_epochs = 201
    LR = 3e-4
    plot_frequency=50

    x = np.linspace(0, 1, domain_size)
    y = np.linspace(0, 1, domain_size)
    xx , yy = np.meshgrid(x, y)

    dataset = Elastic_FSDT_Dataset(domain_size=domain_size)
    v1 = np.zeros_like(dataset.x) # np.sin(math.pi*xx)*np.cos(math.pi*yy)
    v2 = np.zeros_like(dataset.x) # -np.cos(math.pi*xx)*np.sin(math.pi*yy)
    p  = np.zeros_like(dataset.x) # np.sin(math.pi*xx)*np.sin(math.pi*yy)
    u_tensor = np.expand_dims(np.array([v1,v2,p]),0)
    print(u_tensor.shape)
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    basecase = Elastic_FSDT(network, dataset, domain_size=domain_size, batch_size=1, fem_basis_deg=1, learning_rate=LR, plot_frequency=plot_frequency)

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