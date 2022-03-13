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
from DiffNet.datasets.single_instances.cuboids import CuboidManufactured


class Poisson(DiffNet3DFEM):
    """docstring for Poisson"""
    def __init__(self, network, dataset, **kwargs):
        super(Poisson, self).__init__(network, dataset, **kwargs)
        # x = np.linspace(0,1,self.domain_size)
        # y = np.linspace(0,1,self.domain_size)
        # z = np.linspace(0,1,self.domain_size)        
        # xx, yy, zz = np.meshgrid(x,y,z)

        # x_1d = np.linspace(0,1,6)
        # y_1d = np.linspace(0,1,5)
        # z_1d = np.linspace(0,1,4)        
        # xx, yy, zz = self.meshgrid_3d(x_1d, y_1d, z_1d)
        self.u_exact = self.exact_solution(self.dataset.xx,self.dataset.yy,self.dataset.zz)
        self.f_gp = self.forcing(self.xgp,self.ygp,self.zgp)
        self.diffusivity = 1.

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

    def loss(self, u, inputs_tensor, forcing_tensor):
        N_values = self.Nvalues.type_as(u)
        dN_x_values = self.dN_x_values.type_as(u)
        dN_y_values = self.dN_y_values.type_as(u)
        dN_z_values = self.dN_z_values.type_as(u)
        gpw = self.gpw.type_as(u)
        f_gp = self.f_gp.type_as(u)

        # extract diffusivity and boundary conditions here
        f = forcing_tensor # renaming variable
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # DERIVE NECESSARY VALUES
        trnsfrm_jac = (0.5*self.h)**3
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        # apply boundary conditions
        # NOTE: we do add the BC to the residual later, but adding the residual to u is also very important
        #       because ideally we want to calculate the values of A*u when A is BC adjusted. But since we
        #       are not altering the convolution kernel "Kmatrices" (i.e., effectively the values of A), thus
        #       we will end up with bad values in R at the interior points
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)

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
        # add boundary conditions to R <---- this step is very important
        R = torch.where(bc1>0.5,1.+R*0.0,R)
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
        loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
        # self.log('PDE_loss', loss_val.item())
        # self.log('loss', loss_val.item())
        return {"loss": loss_val}

    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs["loss"]
        self.log('loss', loss.item())
        return training_step_outputs

    def configure_optimizers(self):
        lr = self.learning_rate
        opts = [torch.optim.LBFGS(self.network, lr=1.0, max_iter=10)]
        # opts = [torch.optim.Adam(self.network, lr=lr)]
        # opts = [torch.optim.Adam(self.network, lr=lr), torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        return opts, []

    def on_epoch_end(self):
        self.network.eval()
        inputs, forcing = self.dataset[0]
        nu, f, u = self.do_query(inputs, forcing)
        nu = nu.squeeze().detach().cpu()
        u = u.squeeze().detach().cpu()
        self.plot_contours(nu, f, u)

    def do_query(self, inputs, forcing):
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

        return nu, f, u

    def plot_contours(self,nu,f,u):
        fig, axs = plt.subplots(3, 4, figsize=(2*4,1.2*3),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax_row in axs:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
        
        u_exact = self.u_exact
        diff = u - u_exact

        sliceidx = int(self.domain_size / 2)
        # print(np.linalg.norm(diff.flatten())/self.domain_size)
        # z-slices
        row_id = 0
        sliceZ = int(self.domain_size / 2)
        im = axs[row_id,0].imshow(f[sliceZ,:,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,0]) #ticks=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        im = axs[row_id,1].imshow(u[sliceZ,:,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,1])
        im = axs[row_id,2].imshow(u_exact[sliceZ,:,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,2])
        im = axs[row_id,3].imshow(diff[sliceZ,:,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,3])
        # y-slices
        row_id = 1
        sliceY = int(self.domain_size / 2)
        im = axs[row_id,0].imshow(f[:,sliceY,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,0]) #ticks=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        im = axs[row_id,1].imshow(u[:,sliceY,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,1])
        im = axs[row_id,2].imshow(u_exact[:,sliceY,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,2])
        im = axs[row_id,3].imshow(diff[:,sliceY,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,3])
        # x-slices
        row_id = 2
        sliceX = int(self.domain_size / 2)
        im = axs[row_id,0].imshow(f[:,:,sliceX],cmap='jet'); fig.colorbar(im, ax=axs[row_id,0]) #ticks=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        im = axs[row_id,1].imshow(u[:,:,sliceX],cmap='jet'); fig.colorbar(im, ax=axs[row_id,1])
        im = axs[row_id,2].imshow(u_exact[:,:,sliceX],cmap='jet'); fig.colorbar(im, ax=axs[row_id,2])
        im = axs[row_id,3].imshow(diff[:,:,sliceX],cmap='jet'); fig.colorbar(im, ax=axs[row_id,3])

        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
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
    domain_size = 24
    dir_string = "poisson-mms-resmin-3d"
    max_epochs = 10
    u_tensor = np.ones((1,1,domain_size,domain_size,domain_size))
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    dataset = CuboidManufactured(domain_size=domain_size)
    basecase = Poisson(network, dataset, batch_size=1, domain_size=domain_size, learning_rate=0.01, nsd=3, fem_basis_deg=1)

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

    # L2 error calculation
    basecase.dataset[0]
    inputs, forcing = basecase.dataset[0]
    nu, f, u = basecase.do_query(inputs, forcing) 
    print("Calculating L2 error:")
    basecase.calc_l2_err(u.detach())

if __name__ == '__main__':
    main()
