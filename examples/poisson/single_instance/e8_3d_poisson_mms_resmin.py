import os
import sys
import json
import math
import torch
import numpy as np
import libconf

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
from DiffNet.datasets.single_instances.cuboids import CuboidManufactured
from torch.utils.data import DataLoader
# from pytorch_lightning.pytorch.callbacks import Callback
import time
from utils import plot_losses


class Poisson(DiffNet3DFEM):
    """docstring for Poisson"""
    def __init__(self, network, dataset, **kwargs):
        super(Poisson, self).__init__(network, **kwargs)
        # x = np.linspace(0,1,self.domain_size)
        # y = np.linspace(0,1,self.domain_size)
        # z = np.linspace(0,1,self.domain_size)        
        # xx, yy, zz = np.meshgrid(x,y,z)

        # x_1d = np.linspace(0,1,6)
        # y_1d = np.linspace(0,1,5)
        # z_1d = np.linspace(0,1,4)        
        # xx, yy, zz = self.meshgrid_3d(x_1d, y_1d, z_1d)
        # self.u_exact = self.exact_solution(self.dataset.xx,self.dataset.yy,self.dataset.zz)
        self.u_exact = self.exact_solution(dataset.xx,dataset.yy,dataset.zz)
        self.f_gp = self.forcing(self.xgp,self.ygp,self.zgp)
        self.diffusivity = 1.

        # u_bc = np.zeros_like(dataset.xx)
        self.u_bc = torch.FloatTensor(self.u_exact)

        self.optimizer = kwargs.get('optimizer', "lbfgs")
        self.loss_type = kwargs.get('loss_type', "energy")
        if self.loss_type == "energy":
            self.loss_func = self.loss_EnergyMin
        elif self.loss_type == "resmin":
            self.loss_func = self.loss_ResMin


    def exact_solution(self, x,y,z):
        pi = math.pi
        sin = np.sin
        cos = np.cos
        return sin(pi*x)*sin(3.*pi*y)*sin(3.*pi*z)

    def forcing(self,x,y,z):
        pi = math.pi
        sin = np.sin
        cos = np.cos
        f = 19. * pi**2 * sin(pi*x)*sin(3.*pi*y)*sin(3*pi*z)
        return f

    def Q1_3D_vector_assembly(self, Aglobal, Aloc_all):
        Aglobal[:,0, 0:-1, 0:-1, 0:-1] += Aloc_all[:,0, :, :, :]
        Aglobal[:,0, 0:-1, 0:-1, 1:  ] += Aloc_all[:,1, :, :, :]
        Aglobal[:,0, 0:-1, 1:  , 0:-1] += Aloc_all[:,2, :, :, :]
        Aglobal[:,0, 0:-1, 1:  , 1:  ] += Aloc_all[:,3, :, :, :]
        Aglobal[:,0, 1:  , 0:-1, 0:-1] += Aloc_all[:,4, :, :, :]
        Aglobal[:,0, 1:  , 0:-1, 1:  ] += Aloc_all[:,5, :, :, :]
        Aglobal[:,0, 1:  , 1:  , 0:-1] += Aloc_all[:,6, :, :, :]
        Aglobal[:,0, 1:  , 1:  , 1:  ] += Aloc_all[:,7, :, :, :]
        return Aglobal

    def loss_ResMin(self, u, inputs_tensor, forcing_tensor):
        N_values = self.Nvalues.type_as(u)
        dN_x_values = self.dN_x_values.type_as(u)
        dN_y_values = self.dN_y_values.type_as(u)
        dN_z_values = self.dN_z_values.type_as(u)
        gpw = self.gpw.type_as(u)
        f_gp = self.f_gp.type_as(u)
        u_bc = self.u_bc.unsqueeze(0).unsqueeze(0).type_as(u)

        # extract diffusivity and boundary conditions here
        f = forcing_tensor # renaming variable
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # DERIVE NECESSARY VALUES
        trnsfrm_jac = (0.5*self.h)**3
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        # apply boundary conditions
        # NOTE: Adding the residual to u is very important
        #       After this step, "u" is complete with BCs
        #       We later need to do another BC operation after the residual R is calculated
        # u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u_bc,u)

        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)
        u_z_gp = self.gauss_pt_evaluation_der_z(u)

        # CALCULATION STARTS
        # lhs
        vxux = dN_x_values*u_x_gp*JxW
        vyuy = dN_y_values*u_y_gp*JxW
        vzuz = dN_z_values*u_z_gp*JxW
        # rhs
        vf = N_values*f_gp*JxW

        # integrated values on lhs & rhs
        v_lhs = torch.sum(self.diffusivity*(vxux+vyuy+vzuz), 2) # sum across all GP
        v_rhs = torch.sum(vf, 2) # sum across all gauss points

        # unassembled residual
        R_split = v_lhs - v_rhs
        # assembly
        R = torch.zeros_like(u); R = self.Q1_3D_vector_assembly(R, R_split)

        # Boundary operation on residual <---- below step is very important
        # Set the residuals on the Dirichlet boundaries to zero 
        R = torch.where(bc2>0.5,R*0.0,R)

        loss = torch.sum(R**2)
        return loss

    def loss_EnergyMin(self, u, inputs_tensor, forcing_tensor):

        f_gp = self.f_gp.type_as(u)
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:,:]
        bc1 = inputs_tensor[:,1:2,:,:,:]
        bc2 = inputs_tensor[:,2:3,:,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)


        nu_gp = self.gauss_pt_evaluation(nu)
        # f_gp = self.gauss_pt_evaluation(f)
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

    def on_train_epoch_end(self, trainer, pl_module):
        # print("On training epoch end")
        # print("pl_module name = ", pl_module.__class__.__name__)
        # print(trainer.train_dataloader.dataset)
        pl_module.network.eval()
        nu, f, u = self.do_query(trainer, pl_module)
        nu = nu.detach().cpu().squeeze()
        u = u.detach().cpu().squeeze()
        f = f.detach().cpu().squeeze()
        self.plot_contours(trainer, pl_module, nu, f, u)

    def do_query(self, trainer, pl_module):
        inputs, forcing = trainer.train_dataloader.dataset[0]

        u, inputs_tensor, forcing_tensor = pl_module.forward((inputs.unsqueeze(0).type_as(next(pl_module.network.parameters())), forcing.unsqueeze(0).type_as(next(pl_module.network.parameters()))))

        f = forcing_tensor
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        return nu, f, u

    def plot_contours(self,trainer,pl_module,nu,f,u):
        fig, axs = plt.subplots(3, 4, figsize=(2*4,1.2*3),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax_row in axs:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
        
        u_exact = pl_module.u_exact
        diff = u - u_exact

        sliceidx = int(pl_module.domain_size / 2)
        # print(np.linalg.norm(diff.flatten())/self.domain_size)
        # z-slices
        row_id = 0
        sliceZ = int(pl_module.domain_size / 2)
        im = axs[row_id,0].imshow(f[sliceZ,:,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,0]) #ticks=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        im = axs[row_id,1].imshow(u[sliceZ,:,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,1])
        im = axs[row_id,2].imshow(u_exact[sliceZ,:,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,2])
        im = axs[row_id,3].imshow(diff[sliceZ,:,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,3])
        # y-slices
        row_id = 1
        sliceY = int(pl_module.domain_size / 2)
        im = axs[row_id,0].imshow(f[:,sliceY,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,0]) #ticks=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        im = axs[row_id,1].imshow(u[:,sliceY,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,1])
        im = axs[row_id,2].imshow(u_exact[:,sliceY,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,2])
        im = axs[row_id,3].imshow(diff[:,sliceY,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,3])
        # x-slices
        row_id = 2
        sliceX = int(pl_module.domain_size / 2)
        im = axs[row_id,0].imshow(f[:,:,sliceX],cmap='jet'); fig.colorbar(im, ax=axs[row_id,0]) #ticks=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        im = axs[row_id,1].imshow(u[:,:,sliceX],cmap='jet'); fig.colorbar(im, ax=axs[row_id,1])
        im = axs[row_id,2].imshow(u_exact[:,:,sliceX],cmap='jet'); fig.colorbar(im, ax=axs[row_id,2])
        im = axs[row_id,3].imshow(diff[:,:,sliceX],cmap='jet'); fig.colorbar(im, ax=axs[row_id,3])

        plt.savefig(os.path.join(trainer.logger.log_dir, 'contour_' + str(trainer.current_epoch) + '.png'))
        trainer.logger.experiment.add_figure('Contour Plots', fig, trainer.current_epoch)
        plt.close('all')

        # fig, axs = plt.subplots(len(self.vis_sample_ids), 6, figsize=(2*4,1.2*len(self.vis_sample_ids)),
        #                         subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        #     fig.suptitle('Contour Plots')
        #     for ax_row in axs:
        #         for ax in ax_row:
        #             ax.set_xticks([])
        #             ax.set_yticks([])
        #     self.network.eval()
        #     for idx in range(len(self.vis_sample_ids)):
        #         coeff = self.dataset[self.vis_sample_ids[idx]]

        #         # coeff is on cpu (because it is directly indexed from dataset)
        #         # so it has to be moved to the computation device (gpu/cpu) using type_as
        #         coeff = (torch.tensor(np.expand_dims(coeff, axis=0))).type_as(next(self.network.parameters()))
        #         u_gen, gen_input, nu_4d_tensor, forcing_4d_tensor = self.forward(coeff)

        #         u_gen = self.apply_padding_on(u_gen)
        #         k = nu_4d_tensor.squeeze().detach().cpu()
        #         u = u_gen.squeeze().detach().cpu()

        #         sliceidx0 = int(1 * self.nx / 4)
        #         sliceidx1 = int(2 * self.nx / 4)
        #         sliceidx2 = int(3 * self.nx / 4)

        #         im = axs[idx][0].imshow(k[sliceidx0,:,:],cmap='jet')
        #         fig.colorbar(im, ax=axs[idx, 0])
        #         im = axs[idx][1].imshow(k[sliceidx1,:,:],cmap='jet')
        #         fig.colorbar(im, ax=axs[idx, 1])
        #         im = axs[idx][2].imshow(k[sliceidx2,:,:],cmap='jet')
        #         fig.colorbar(im, ax=axs[idx, 2])
        #         im = axs[idx][3].imshow(u[sliceidx0,:,:],cmap='jet')
        #         fig.colorbar(im, ax=axs[idx, 3])
        #         im = axs[idx][4].imshow(u[sliceidx1,:,:],cmap='jet')
        #         fig.colorbar(im, ax=axs[idx, 4])
        #         im = axs[idx][5].imshow(u[sliceidx2,:,:],cmap='jet')
        #         fig.colorbar(im, ax=axs[idx, 5])

def main():
    with open('conf_e8_poisson3d.inp') as f:
        cfg = libconf.load(f)
    domain_size = cfg.domain_size
    max_epochs = cfg.max_epochs
    LR=cfg.LR
    loss_type = cfg.loss_type # energy, resmin
    optimizer = cfg.optimizer # adam, lbfgs, sgd
    print(f"Config = ({loss_type}, {optimizer})")
    dir_string = "poisson-mms-3d"
    u_tensor = np.ones((1,1,domain_size,domain_size,domain_size))
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    dataset = CuboidManufactured(domain_size=domain_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    basecase = Poisson(network, dataset, batch_size=1, domain_size=domain_size, learning_rate=LR, nsd=3, fem_basis_deg=1, loss_type=loss_type, optimizer=optimizer)

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
    plot_losses(logger.log_dir)

    # L2 error calculation
    print("Calculating L2 error:")
    basecase.network.eval()
    inputs, forcing = trainer.train_dataloader.dataset[0]
    nu, f, u = printsave.do_query(trainer, basecase)
    basecase.calc_l2_err(u.detach())

if __name__ == '__main__':
    main()
