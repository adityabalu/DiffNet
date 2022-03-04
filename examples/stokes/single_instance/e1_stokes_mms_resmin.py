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
        self.bc1[:,0:1] = 1.0
        self.bc1[:,-1:] = 1.0
        self.bc1[0:1,:] = 1.0
        self.bc1[-1:,:] = 1.0

        self.bc2 = np.zeros_like(xx)
        self.bc2_val = np.zeros_like(xx)
        # self.bc2[-1:,:] = 1.0

        self.bc3 = np.zeros_like(xx)
        # self.bc3[0:1,0:1] = 1.0

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


class Stokes(DiffNet2DFEM):
    """docstring for Eiqonal"""
    def __init__(self, network, dataset, **kwargs):
        super(Stokes, self).__init__(network, dataset, **kwargs)
        
        self.Re = self.dataset.Re

        ue, ve, pe = self.exact_solution(self.dataset.x, self.dataset.y)
        self.u_exact = torch.FloatTensor(ue)
        self.v_exact = torch.FloatTensor(ve)
        self.p_exact = torch.FloatTensor(pe)

        fx_gp, fy_gp = self.forcing(self.xgp, self.ygp)
        self.fx_gp = torch.FloatTensor(fx_gp)
        self.fy_gp = torch.FloatTensor(fy_gp)
        
        self.pspg_param = self.h**2 * self.Re / 12.

    def exact_solution(self, x, y):
        pi = math.pi
        sin = np.sin
        cos = np.cos        
        u_exact =  sin(pi*x)*cos(pi*y)
        v_exact = -cos(pi*x)*sin(pi*y)
        p_exact =  sin(pi*x)*sin(pi*y)
        return u_exact, v_exact, p_exact

    def forcing(self, x, y):
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

    def loss(self, pred, inputs_tensor, forcing_tensor):
        # print("pred type = ", pred.type(), ", pred.shape = ", pred.shape)
        # exit()
        N_values = self.Nvalues.type_as(pred)
        dN_x_values = self.dN_x_values.type_as(pred)
        dN_y_values = self.dN_y_values.type_as(pred)
        gpw = self.gpw.type_as(pred)
        fx_gp = self.fx_gp.type_as(pred)
        fy_gp = self.fy_gp.type_as(pred)
        u_bc = self.u_exact.unsqueeze(0).unsqueeze(0).type_as(pred)
        v_bc = self.v_exact.unsqueeze(0).unsqueeze(0).type_as(pred)
        p_bc = self.p_exact.unsqueeze(0).unsqueeze(0).type_as(pred)


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
        v = torch.where(bc1>=0.5, v_bc, v)
        p = torch.where(bc1>=0.5, p_bc, p)
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
        temp3 = W_U1x+W_U2y #+ self.pspg_param*(Wx_Px+Wy_Py)

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
        R2 = torch.where(bc1>=0.5, v_bc, R2)
        R3 = torch.where(bc1>=0.5, p_bc, R3)

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
        opts = [torch.optim.LBFGS(self.network, lr=0.0001, max_iter=1)]
        # opts = [torch.optim.Adam(self.network, lr=lr)]
        schd = []
        # schd = [torch.optim.lr_scheduler.ExponentialLR(opts[0], gamma=0.7)]
        return opts, schd

    def on_epoch_end(self):
        self.network.eval()
        inputs, forcing = self.dataset[0]
        u, v, p, u_x, v_y = self.do_query(inputs, forcing)
        self.plot_contours(u, v, p, u_x, v_y)

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
        u_bc = self.u_exact.unsqueeze(0).unsqueeze(0).type_as(pred)
        v_bc = self.v_exact.unsqueeze(0).unsqueeze(0).type_as(pred)
        p_bc = self.p_exact.unsqueeze(0).unsqueeze(0).type_as(pred)
        # u = torch.where(bc1>=0.05, u*0.0, u)
        # u = torch.where(bc2>=0.05, u*0.0 + 1.0, u)

        # v = torch.where(torch.logical_or((bc1>=0.5),(bc2>=0.5)), v*0.0, v)
        # p = torch.where(bc3>=0.5, p*0.0, p)
        u = torch.where(bc1>=0.5, u_bc, u)
        v = torch.where(bc1>=0.5, v_bc, v)
        p = torch.where(bc1>=0.5, p_bc, p)
        
        u_x = self.gauss_pt_evaluation_der_x(u)[:,0,:,:].squeeze().detach().cpu()
        v_y = self.gauss_pt_evaluation_der_y(v)[:,0,:,:].squeeze().detach().cpu()

        u = u.squeeze().detach().cpu()
        v = v.squeeze().detach().cpu()
        p = p.squeeze().detach().cpu()        

        return u, v, p, u_x, v_y

    def plot_contours(self, u, v, p, u_x, v_y):
        fig, axs = plt.subplots(2, 6, figsize=(2*6,1.2*2),
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

        im3 = axs[0,3].imshow(div,cmap='jet',origin='lower')
        fig.colorbar(im3, ax=axs[0,3])  
        im4 = axs[0,4].imshow((u**2 + v**2)**0.5,cmap='jet',origin='lower')
        fig.colorbar(im4, ax=axs[0,4])

        xx , yy = np.meshgrid(x, y)
        im5 = axs[0,5].streamplot(xx, yy, u, v, color='k', cmap='jet')

        im = axs[1,0].imshow(u-self.u_exact.numpy(),cmap='jet', origin='lower')
        fig.colorbar(im, ax=axs[1,0])
        im = axs[1,1].imshow(v-self.v_exact.numpy(),cmap='jet',origin='lower')
        fig.colorbar(im, ax=axs[1,1])  
        im = axs[1,2].imshow(p-self.p_exact.numpy(),cmap='jet',origin='lower')
        fig.colorbar(im, ax=axs[1,2])

        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')


        baseline_cut = np.array([[0.0032066932270914394, -0.0007171314741036827],
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
                [0.9961859760956173, 0.9937051792828686]])

        plt.figure()
        plt.plot(u[0,:], 'k--', label='u')
        plt.plot(v[:,0], 'k:', label='v')
        plt.legend()
        plt.savefig(os.path.join(self.logger[0].log_dir, 'linecut_' + str(self.current_epoch) + '.png'))
        plt.close('all')

def main():
    domain_size = 16
    dir_string = "stokes_mms"
    max_epochs = 1000

    x = np.linspace(0, 1, domain_size)
    y = np.linspace(0, 1, domain_size)
    xx , yy = np.meshgrid(x, y)

    dataset = LDC(domain_size=domain_size)
    v1 = np.zeros_like(dataset.x) # np.sin(math.pi*xx)*np.cos(math.pi*yy)
    v2 = np.zeros_like(dataset.x) # -np.cos(math.pi*xx)*np.sin(math.pi*yy)
    p  = np.zeros_like(dataset.x) # np.sin(math.pi*xx)*np.sin(math.pi*yy)
    u_tensor = np.expand_dims(np.array([v1,v2,p]),0)
    print(u_tensor.shape)
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    basecase = Stokes(network, dataset, domain_size=domain_size, batch_size=1, fem_basis_deg=1)

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