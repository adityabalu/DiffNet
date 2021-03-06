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

from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
seed_everything(42)

import DiffNet
from DiffNet.networks.wgan import GoodNetwork
from DiffNet.DiffNetFEM import DiffNet2DFEM
# from DiffNet.datasets.single_instances.rectangles import RectangleManufactured
from DiffNet.datasets.single_instances.rectangles import AdvDiff2dRectangle
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

class AdvDiff2d(DiffNet2DFEM):
    """docstring for AdvDiff2d"""
    def __init__(self, network, dataset, **kwargs):
        super(AdvDiff2d, self).__init__(network, dataset, **kwargs)
        self.save_frequency = kwargs.get('save_frequency', 1)
        self.mapping_type = kwargs.get('mapping_type', 'no_network')

        self.Pe = 1.0
        self.adv = np.array([np.cos(math.pi/6.), np.sin(math.pi/6.)])
        self.advMag = np.linalg.norm(self.adv,2)
        self.diffusivity = 1e-4;
        self.gamma = 2.*self.Pe / self.h
        
        self.u_exact = self.exact_solution(self.xx.numpy(),self.yy.numpy())
        self.tau = 1. / (2. * self.advMag / self.h + 4 * self.diffusivity / self.h**2)

        self.f_gp = self.forcing(self.xgp, self.ygp)

        self.Kmatrices = nn.ParameterList()
        Aet = np.array([[-1.0,-0.5,1.0,0.5],[-0.5,-1.0,0.5,1.0],[-1.0,-0.5,1.0,0.5],[-0.5,-1.0,0.5,1.0]])/6.*self.h; 
        AconvX = np.array([[-1.0,1.0,-0.5,0.5],[-1.0,1.0,-0.5,0.5],[-0.5,0.5,-1.0,1.0],[-0.5,0.5,-1.0,1.0]])/6.*self.h; 
        AconvY = np.array([[-1.0,-0.5,1.0,0.5],[-0.5,-1.0,0.5,1.0],[-1.0,-0.5,1.0,0.5],[-0.5,-1.0,0.5,1.0]])/6.*self.h; 
        # Aed = np.array([[2.0,-2.0, 1.0,-1.0],[-2.0, 2.0,-1.0, 1.0], [1.0,-1.0, 2.0,-2.0],[-1.0, 1.0,-2.0, 2.0]])/6.;
        Aed = np.array([[4.,-1.,-1.,-2.],[-1.,4.,-2.,-1],[-1.,-2.,4.,-1.],[-2.,-1.,-1.,4.]])/6.
        supgXX = np.array([[ 1.00,-1.00, 0.50,-0.50],[-1.00, 1.00,-0.50, 0.50],[ 0.50,-0.50, 1.00,-1.00],[-0.50, 0.50,-1.00, 1.00]])/3.
        supgXY = np.array([[ 0.75, 0.75,-0.75,-0.75],[-0.75,-0.75, 0.75, 0.75],[ 0.75, 0.75,-0.75,-0.75],[-0.75,-0.75, 0.75, 0.75]])/3.
        supgYX = np.array([[ 0.75,-0.75, 0.75,-0.75],[ 0.75,-0.75, 0.75,-0.75],[-0.75, 0.75,-0.75, 0.75],[-0.75, 0.75,-0.75, 0.75]])/3.
        supgYY = np.array([[ 1.00, 0.50,-1.00,-0.50],[ 0.50, 1.00,-0.50,-1.00],[-1.00,-0.50, 1.00, 0.50],[-0.50,-1.00, 0.50, 1.00]])/3.
        Kmx = torch.FloatTensor(
                self.adv[0]*AconvX
                + self.adv[1]*AconvY
                + self.diffusivity*Aed
                + self.tau * self.adv[0]*self.adv[0]*supgXX
                + self.tau * self.adv[0]*self.adv[1]*supgXY
                + self.tau * self.adv[1]*self.adv[0]*supgYX
                + self.tau * self.adv[1]*self.adv[1]*supgYY
            )
        for j in range(4):
            k = Kmx[j,:].reshape((2,2))
            print("k = ", k*6)
            self.Kmatrices.append(nn.Parameter(k.unsqueeze(0).unsqueeze(1), requires_grad=False))
        # print("self.Kmatrices[0] = ", self.Kmatrices[0])
        print("self.Kmatrices[0].shape = ", self.Kmatrices[0].shape)

    def exact_solution(self, x,y):
        return np.zeros_like(x)
        # return np.sin(math.pi*x)*np.exp(-self.dataset.decay_rt*y)
        # return np.sin(2.*math.pi*x)*np.sin(2.*math.pi*y)
        # return torch.sin(math.pi*x)*torch.sin(math.pi*y)

    def forcing(self, x, y):
        sin = torch.sin
        cos = torch.cos
        exp = torch.exp
        # return 2. * math.pi**2 * sin(math.pi * x) * cos(math.pi * y)
        # return 2. * math.pi**2 * sin(math.pi * x) * sin(math.pi * y)
        # return sin(math.pi * x) * exp(-y) * (self.diffusivity * math.pi**2 - 1.)
        return torch.zeros_like(x)

    def loss(self, u, inputs_tensor, forcing_tensor):
        N_values = self.Nvalues.type_as(u)
        dN_x_values = self.dN_x_values.type_as(u)
        dN_y_values = self.dN_y_values.type_as(u)
        gpw = self.gpw.type_as(u)
        f_gp = self.f_gp.type_as(u)

        # extract diffusivity and boundary conditions here
        f = forcing_tensor # renaming variable
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # DERIVE NECESSARY VALUES
        trnsfrm_jac = (0.5*self.h)**2
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        # apply boundary conditions
        # NOTE: we do add the BC to the residual later, but adding the residual to u is also very important
        #       because ideally we want to calculate the values of A*u when A is BC adjusted. But since we
        #       are not altering the convolution kernel "Kmatrices" (i.e., effectively the values of A), thus
        #       we will end up with bad values in R at the interior points
        u = torch.where(bc1>0.5,u*0.0+1.,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        # CALCULATION STARTS
        nu_gp = self.gauss_pt_evaluation(nu)
        # lhs
        vux = N_values*u_x_gp*JxW
        vuy = N_values*u_y_gp*JxW
        vxux = dN_x_values*u_x_gp*JxW
        vxuy = dN_x_values*u_y_gp*JxW
        vyux = dN_y_values*u_x_gp*JxW
        vyuy = dN_y_values*u_y_gp*JxW
        # rhs
        vf = (N_values + self.tau*(self.adv[0]*dN_x_values+self.adv[1]*dN_y_values))*f_gp*JxW

        # integrated values on lhs & rhs
        temp = (self.adv[0]*vux + self.adv[1]*vuy
                + self.diffusivity*nu_gp*(vxux+vyuy)
                + self.tau*self.adv[0]*self.adv[0]*vxux
                + self.tau*self.adv[0]*self.adv[1]*vxuy
                + self.tau*self.adv[1]*self.adv[0]*vyux
                + self.tau*self.adv[1]*self.adv[1]*vyuy
            )

        v_lhs = torch.sum(temp, 2) # sum across all GP
        v_rhs = torch.sum(vf, 2) # sum across all gauss points

        # unassembled residual
        R_split = v_lhs - v_rhs
        # matvec = stiffness_vs_values_conv(u, self.Kmatrices)
        # diff = v_lhs - matvec
        # print("diffnorm = ", torch.norm(diff, 'fro'))

        # R_split = matvec

        # assembly
        R = torch.zeros_like(u)
        R[:,0, 0:-1, 0:-1] += R_split[:,0, :, :]
        R[:,0, 0:-1, 1:  ] += R_split[:,1, :, :]
        R[:,0, 1:  , 0:-1] += R_split[:,2, :, :]
        R[:,0, 1:  , 1:  ] += R_split[:,3, :, :]
        # add boundary conditions to R <---- this step is very important
        R = torch.where(bc1>0.5,R*0.0+1.,R)
        R = torch.where(bc2>0.5,R*0.0,R)

        # loss = torch.norm(R,'fro')**2
        loss = torch.sum(R**2)
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        if self.mapping_type == 'no_network':
            return self.network[0], inputs_tensor, forcing_tensor
        elif self.mapping_type == 'network':
            nu = inputs_tensor[:,0:1,:,:]
            return self.network(nu), inputs_tensor, forcing_tensor

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
        # loss_val = self.loss_resmin(u, inputs_tensor, forcing_tensor).mean()
        # loss_val = self.loss_resmin_mass(u, inputs_tensor, forcing_tensor).mean()
        # loss_val = self.loss_matvec(u, inputs_tensor, forcing_tensor).mean()
        # self.log('PDE_loss', loss_val.item())
        # self.log('loss', loss_val.item())
        return {"loss": loss_val}

    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs["loss"]
        return training_step_outputs

    def configure_optimizers(self):
        lr = self.learning_rate
        # opts = [torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        opts = [torch.optim.Adam(self.network.parameters(), lr=lr)]
        # opts = [torch.optim.Adam(self.network, lr=lr), torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        return opts, []

    def on_epoch_end(self):
        self.network.eval()
        inputs, forcing = self.dataset[0]
        nu, f, u = self.do_query(inputs, forcing)

        if self.current_epoch % self.save_frequency == 0:
            with open(os.path.join(self.logger[0].log_dir, "save_info.txt"), "a") as myfile:
                myfile.write("Last save @ epoch {}\n".format(self.current_epoch))
            # Save network
            torch.save(self.network, os.path.join(self.logger[0].log_dir, 'network.pt'))
            self.plot_contours(nu, f, u)

    def do_query(self, inputs, forcing):
        u, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

        f = forcing_tensor.squeeze().detach().cpu() # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        # u0 = self.dataset.u0.unsqueeze(0).unsqueeze(0).type_as(u)
        u = torch.where(bc1>0.5,u*0.0+1.,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        nu = nu.squeeze().detach().cpu()
        u = u.squeeze().detach().cpu()

        return nu, f, u

    def plot_contours(self,nu,f,u):
        fig, axs = plt.subplots(1, 4, figsize=(2*4,1.2),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        self.u_curr = u

        u_exact = self.u_exact.squeeze()
        diff = u - u_exact
        # print(np.linalg.norm(diff.flatten())/self.domain_size)
        im0 = axs[0].imshow(f,cmap='jet', vmin=0.0, vmax=20.0)
        # fig.colorbar(im0, ax=axs[0], ticks=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0])
        fig.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(u,cmap='jet')
        fig.colorbar(im1, ax=axs[1])
        im2 = axs[2].imshow(u_exact,cmap='jet', vmin=0.0, vmax=1.0)
        fig.colorbar(im2, ax=axs[2])
        im3 = axs[3].imshow(diff,cmap='jet') #, vmin=0.0, vmax=0.5)
        fig.colorbar(im3, ax=axs[3])
        ff = self.forcing(self.xgp, self.ygp)
        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    # u_tensor = np.random.randn(1,1,256,256)

    caseId = 0
    domain_size = 64
    dir_string = "adv-diff-2d"
    LR = 1e-2
    max_epochs = 11
    save_frequency = 5
    load_from_prev = False
    load_version_id = 60
    # mapping_type = 'network'
    mapping_type = 'no_network'

    if load_from_prev:
        print("LOADING FROM PREVIOUS VERSION: ", load_version_id)
        case_dir = os.path.join('.', dir_string, 'version_'+str(load_version_id))
        network = torch.load(os.path.join(case_dir, 'network.pt'))
    else:
        print("INITIALIZING PARAMETERS TO ZERO")
        u_tensor = np.ones((1,1,domain_size,domain_size))
    
    if mapping_type == 'no_network':
        network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    elif mapping_type == 'network':
        network = AE(in_channels=1, out_channels=1, dims=domain_size, n_downsample=3)
    dataset = AdvDiff2dRectangle(domain_size=domain_size)
    basecase = AdvDiff2d(network, dataset, batch_size=1, domain_size=domain_size, learning_rate=LR, mapping_type=mapping_type, save_frequency=save_frequency)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name=dir_string)
    # logger = pl.loggers.TensorBoardLogger('.', name="simple-resmin")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=max_epochs, deterministic=True, profiler="simple")

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