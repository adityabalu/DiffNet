import os
import sys
import json
import math
import torch
import numpy as np
import time
import libconf

from attrdict import AttrDict
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
from DiffNet.datasets.single_instances.rectangles import RectangleManufactured
from torch.utils.data import DataLoader
from utils import plot_losses

class Poisson(DiffNet2DFEM):
    """docstring for Poisson"""
    def __init__(self, network, dataset, **kwargs):
        super(Poisson, self).__init__(network, **kwargs)
        self.u_exact = self.exact_solution(dataset.xx,dataset.yy)
        self.f_gp = self.forcing_func(self.xgp,self.ygp)

        self.u_bc = torch.FloatTensor(self.u_exact)

        self.cfg = kwargs.get('cfg', AttrDict({'optimizer':"lbfgs", 'loss_type':"energy", 'plot_frequency':50}))
        self.optimizer = self.cfg.optimizer
        self.loss_type = self.cfg.loss_type
        self.plot_frequency = self.cfg.plot_frequency
        if self.loss_type == "energy":
            self.loss_func = self.loss_EnergyMin
        elif self.loss_type == "resmin":
            self.loss_func = self.loss_ResMin


    def exact_solution(self, x,y):
        m = 3.; n = 2.
        pi = math.pi
        sin = np.sin
        cos = np.cos
        return sin(m*pi*x)*cos(n*pi*y)

    def forcing_func(self,x,y):
        m = 3.; n = 2.
        pi = math.pi
        sin = np.sin
        cos = np.cos
        f = (m**2 + n**2) * pi**2 * sin(m*pi*x)*cos(n*pi*y)
        return f

    def Q1_2D_vector_assembly(self, Aglobal, Aloc_all):
        Aglobal[:,0, 0:-1, 0:-1] += Aloc_all[:,0, :, :]
        Aglobal[:,0, 0:-1, 1:  ] += Aloc_all[:,1, :, :]
        Aglobal[:,0, 1:  , 0:-1] += Aloc_all[:,2, :, :]
        Aglobal[:,0, 1:  , 1:  ] += Aloc_all[:,3, :, :]
        return Aglobal

    def loss_ResMin(self, u, inputs_tensor, forcing_tensor):
        N_values = self.Nvalues.type_as(u)
        dN_x_values = self.dN_x_values.type_as(u)
        dN_y_values = self.dN_y_values.type_as(u)
        gpw = self.gpw.type_as(u)
        f_gp = self.f_gp.type_as(u).unsqueeze(1)
        u_bc = self.u_bc.unsqueeze(0).unsqueeze(0).type_as(u)

        # extract diffusivity and boundary conditions here
        f = forcing_tensor # renaming variable
        nu = inputs_tensor[:,0:1,...]
        bc1 = inputs_tensor[:,1:2,...]
        bc2 = inputs_tensor[:,2:3,...]

        # DERIVE NECESSARY VALUES
        trnsfrm_jac = (0.5*self.h)**3
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0).type_as(u)

        # apply boundary conditions
        # NOTE: Adding the residual to u is very important
        #       After this step, "u" is complete with BCs
        #       We later need to do another BC operation after the residual R is calculated
        # u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u_bc,u)

        nu_gp = self.gauss_pt_evaluation(nu).unsqueeze(1)
        u_x_gp = self.gauss_pt_evaluation_der_x(u).unsqueeze(1)
        u_y_gp = self.gauss_pt_evaluation_der_y(u).unsqueeze(1)

        # CALCULATION STARTS
        vxux = dN_x_values*u_x_gp
        vyuy = dN_y_values*u_y_gp
        lhs = nu_gp * (vxux+vyuy)
        rhs = N_values*f_gp
        # residual
        residual = lhs-rhs

        # unassembled residual
        R_split = torch.sum(residual*JxW, 2) # sum across all gauss points
        # assembly
        R = torch.zeros_like(u); R = self.Q1_2D_vector_assembly(R, R_split)

        # Boundary operation on residual <---- below step is very important
        # Set the residuals on the Dirichlet boundaries to zero 
        R = torch.where(bc2>0.5,R*0.0,R)

        loss = torch.sum(R**2)
        return loss

    def loss_EnergyMin(self, u, inputs_tensor, forcing_tensor):

        f_gp = self.f_gp.type_as(u)
        u_bc = self.u_bc.unsqueeze(0).unsqueeze(0).type_as(u)
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,...]
        bc1 = inputs_tensor[:,1:2,...]
        bc2 = inputs_tensor[:,2:3,...]

        # apply boundary conditions
        # u = torch.where(bc1>0.5,1.0+u*0.0,u)
        # u = torch.where(bc2>0.5,u*0.0,u)
        u = torch.where(bc2>0.5,u_bc,u)


        nu_gp = self.gauss_pt_evaluation(nu)
        # f_gp = self.gauss_pt_evaluation(f)
        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(nu_gp)
        res_elmwise = transformation_jacobian * ((0.5 * nu_gp * (u_x_gp**2 + u_y_gp**2) - (u_gp * f_gp)))
        
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
        # loss_val = self.loss_EnergyMin(u, inputs_tensor, forcing_tensor).mean()
        loss_val = self.loss_func(u, inputs_tensor, forcing_tensor).mean()
        # self.log('PDE_loss', loss_val.item())
        # self.log('loss', loss_val.item())
        self.log("loss", loss_val)
        return {"loss": loss_val}

    # def training_step_end(self, training_step_outputs):
    #     loss = training_step_outputs["loss"]
    #     self.log('loss', loss.item())
    #     return training_step_outputs

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.optimizer == "adam":
            print("Choosing Adam")
            opts = [torch.optim.Adam(self.network, lr=lr)]
        elif self.optimizer == "lbfgs":
            print("Choosing LBFGS")
            opts = [torch.optim.LBFGS(self.network, lr=1.0, max_iter=10)]
        elif self.optimizer == "sgd":
            print("Choosing SGD")
            opts = [torch.optim.SGD(self.network, lr=lr)]
        # opts = [torch.optim.Adam(self.network, lr=lr), torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        return opts, []

class MyPrintingCallback(pl.callbacks.Callback):
    # def on_train_start(self, trainer, pl_module):
    #     print("Training is starting")
    # def on_train_end(self, trainer, pl_module):
    #     print("Training is ending")
    # def on_validation_epoch_end(self, trainer, pl_module):
    #     print("On validation epoch end")
    def __init__(self, **kwargs):
        super(MyPrintingCallback, self).__init__(**kwargs)
        self.num_query = 2

    def on_train_epoch_end(self, trainer, pl_module):
        # print("On training epoch end")
        # print("pl_module name = ", pl_module.__class__.__name__)
        # print(trainer.train_dataloader.dataset)
        if trainer.current_epoch % pl_module.cfg.plot_frequency == 0:
            pl_module.network.eval()
            nu, f, u = self.do_query(trainer, pl_module)
            nu = nu.detach().cpu().squeeze()
            u = u.detach().cpu().squeeze()
            f = f.detach().cpu().squeeze()
            f = torch.FloatTensor(pl_module.forcing_func(trainer.train_dataloader.dataset.xx, trainer.train_dataloader.dataset.yy))
            self.plot_contours(trainer, pl_module, torch.tile(nu,(self.num_query,1,1)), torch.tile(f,(self.num_query,1,1)), torch.tile(u,(self.num_query,1,1)))
            if trainer.current_epoch > 2:
                plot_losses(trainer.logger.log_dir)

    def do_query(self, trainer, pl_module):
        inputs, forcing = trainer.train_dataloader.dataset[0:self.num_query]

        u, inputs_tensor, forcing_tensor = pl_module.forward((inputs.unsqueeze(0).type_as(next(pl_module.network.parameters())), forcing.unsqueeze(0).type_as(next(pl_module.network.parameters()))))

        f = forcing_tensor
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,...]
        bc1 = inputs_tensor[:,1:2,...]
        bc2 = inputs_tensor[:,2:3,...]

        # apply boundary conditions
        u_bc = pl_module.u_bc.unsqueeze(0).unsqueeze(0).type_as(u)
        u = torch.where(bc2>0.5,u_bc,u)

        return nu, f, u

    def plot_contours(self,trainer,pl_module,nu,f,u):
        plt_num_row = self.num_query
        plt_num_col = 4
        fig, axs = plt.subplots(plt_num_row, plt_num_col, figsize=(2*plt_num_col,1.2*plt_num_row),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax_row in axs:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])

        for idx in range(self.num_query):

            # extract diffusivity and boundary conditions here
            ki = nu[idx,...]
            ui = u[idx,...]
            fi = f[idx,...]

            im0 = axs[idx][0].imshow(ki,cmap='jet')
            fig.colorbar(im0, ax=axs[idx,0])
            im1 = axs[idx][1].imshow(fi,cmap='jet')
            fig.colorbar(im1, ax=axs[idx,1])
            im1 = axs[idx][2].imshow(ui,cmap='jet')
            fig.colorbar(im1, ax=axs[idx,2])
            im1 = axs[idx][3].imshow(ui-pl_module.u_exact,cmap='jet')
            fig.colorbar(im1, ax=axs[idx,3])
        plt.savefig(os.path.join(trainer.logger.log_dir, 'contour_' + str(trainer.current_epoch) + '.png'))
        trainer.logger.experiment.add_figure('Contour Plots', fig, trainer.current_epoch)
        plt.close('all')

def main():
    with open('conf_e8_2d.inp') as f:
        cfg = libconf.load(f)
    domain_size = cfg.domain_size
    max_epochs = cfg.max_epochs
    LR=cfg.LR
    # loss_type = cfg.loss_type # energy, resmin
    # optimizer = cfg.optimizer # adam, lbfgs, sgd
    print(f"Config = ({cfg.loss_type}, {cfg.optimizer})")
    dir_string = "poisson-mms-2d"
    u_tensor = np.ones((1,1,domain_size,domain_size))
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    dataset = RectangleManufactured(domain_size=domain_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    basecase = Poisson(network, dataset, batch_size=1, domain_size=domain_size, learning_rate=LR, nsd=2, fem_basis_deg=1, cfg=cfg)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name=dir_string)
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)
    printsave = MyPrintingCallback()

    trainer = pl.Trainer(accelerator='gpu',devices=1,
                         callbacks=[checkpoint,printsave],
                         logger=[logger,csv_logger],
                         max_epochs=max_epochs,
                         fast_dev_run=False
                         )

    # ------------------------
    # 4 Training
    # ------------------------
    ts = time.time()
    trainer.fit(basecase, dataloader)
    te = time.time()
    print("[TIMING] Trainer took {:.3f} sec".format(te-ts))

    # ------------------------
    # 5 SAVE NETWORK
    # ------------------------
    torch.save(basecase.network, os.path.join(logger.log_dir, 'network.pt'))
    with open(os.path.join(logger.log_dir, 'conf.txt'), 'w') as f:
        print(libconf.dumps(cfg), file=f)

    # L2 error calculation
    print("Calculating L2 error:")
    basecase.network.eval()
    inputs, forcing = trainer.train_dataloader.dataset[0]
    nu, f, u = printsave.do_query(trainer, basecase)
    basecase.calc_l2_err(u.detach())

if __name__ == '__main__':
    main()
