import os
import sys
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
from DiffNet.networks.immdiff_networks import ConvNet
from DiffNet.DiffNetFEM import DiffNet2DFEM, DiffNet3DFEM
import PIL
from torch.utils import data

import torch
from NURBSDiff.curve_eval import CurveEval




# def im_io(filepath):
#     image = io.imread(filepath).astype(bool).astype(float)
    

#     return im2pc(image)

# def im2pc(image, nx, ny):
#     pc = []
#     normals = []
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             if image[i,j] == 1.0:
#                 boundary = 0
#                 if i < image.shape[0] - 1:
#                     if image[i+1,j] == 0:
#                         boundary = 1
#                 if j < image.shape[1] - 1:
#                     if image[i,j+1] == 0:
#                         boundary = 1
#                 if i > 0:
#                     if image[i-1,j] == 0:
#                         boundary = 1
#                 if j > 0:
#                     if image[i,j-1] == 0:
#                         boundary = 1
#                 if i < image.shape[0] - 1  and j < image.shape[1] - 1:
#                     if image[i+1,j+1] == 0:
#                         boundary = 1
#                 if i < image.shape[0] - 1  and j > 0:
#                     if image[i+1,j-1] == 0:
#                         boundary = 1
#                 if i > 0 and j < image.shape[1] - 1:
#                     if image[i-1,j+1] == 0:
#                         boundary = 1
#                 if i > 0 and j > 0:
#                     if image[i-1,j-1] == 0:
#                         boundary = 1
#                 if boundary == 1:
#                     pc.append([i+0.5,j+0.5])
#                     normals.append([nx[i,j]/(nx[i,j]**2 + ny[i,j]**2), ny[i,j]/(nx[i,j]**2 + ny[i,j]**2)])
#     return np.array(pc), np.array(normals)


class PC3D(data.Dataset):
    'PyTorch dataset for 3D point cloud data'
    def __init__(self, domain_size=32):
        """
        Initialization
        """
        pt_cloud = []
        for _ in range(1000):
            vec = np.random.randn(3)
            vec /= 4*np.linalg.norm(vec)
            pt_cloud.append(vec)
        pt_cloud = np.array(pt_cloud)
        self.normals = pt_cloud*4.0
        self.pc = pt_cloud + 0.5

        self.domain = np.ones((domain_size,domain_size,domain_size))
        self.domain_size = domain_size
        self.n_samples = 100

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.pc, self.normals]) # 2, Npoint, 2
        forcing = np.ones_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

    



class Eiqonal(DiffNet3DFEM):
    """docstring for Eiqonal"""
    def __init__(self, network, dataset, **kwargs):
        super(Eiqonal, self).__init__(network, dataset, **kwargs)
        self.mapping_type = kwargs.get('mapping_type', 'no_network')
        self.nsd = kwargs.get('nsd', 3)
        self.network = network

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
        f = forcing_tensor # renaming variable

        # init bin widths
        hx = self.h
        hy = self.h
        hz = self.h

        # init vars for weak formulation
        N_values = self.Nvalues.type_as(u)
        # print('* '* 10)
        # print(N_values.shape)
        # print('* '* 10)
        # exit()
        gpw = self.gpw.type_as(u)
        
        # u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)
        u_z_gp = self.gauss_pt_evaluation_der_z(u)

        # bin width (reimann sum)
        trnsfrm_jac = (0.5*hx)*(0.5*hy)*(0.5*hz)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        # print('* '* 10)
        # print(JxW.shape)
        # print(N_values.shape)
        # print('* '* 10)
        # exit()
        # Eikonal Equation
        eikonal_lhs = (u_x_gp)**2 + (u_y_gp)**2 + (u_z_gp)**2
        eikonal_rhs = self.h 
        R1 = JxW * N_values * (eikonal_lhs - eikonal_rhs)

        # extract diffusivity and boundary conditions here
        pc = inputs_tensor[:,0:1,:,:].type_as(f)
        normals = inputs_tensor[:,1:2,:,:].type_as(f)

        # apply boundary conditions
        nidx = (pc[:,:,:,0]/self.hx).type(torch.LongTensor).to(pc.device)
        nidy = (pc[:,:,:,1]/self.hy).type(torch.LongTensor).to(pc.device)
        nidz = (pc[:,:,:,1]/self.hz).type(torch.LongTensor).to(pc.device)

        # u_pts_grid =  torch.stack([
        #         torch.stack([
        #             torch.stack([u[b,0,nidx[b,0,:],nidy[b,0,:]] for b in range(u.size(0))]),
        #             torch.stack([u[b,0,nidx[b,0,:]+1,nidy[b,0,:]] for b in range(u.size(0))])]),
        #         torch.stack([
        #             torch.stack([u[b,0,nidx[b,0,:],nidy[b,0,:]+1] for b in range(u.size(0))]),
        #             torch.stack([u[b,0,nidx[b,0,:]+1,nidy[b,0,:]+1] for b in range(u.size(0))])])
        #         ]).unsqueeze(2)


        # print('* '*10)
        # print('u_pts_grid = ', u_pts_grid.shape)
        # print('* '*10)
        # exit()

        u_pts_grid =  torch.stack([
        torch.stack([
        torch.stack([
            torch.stack([
                u[b,0,nidx[b,0,:]*self.fem_basis_deg+d3,nidy[b,0,:]*self.fem_basis_deg+d2,nidz[b,0,:]*self.fem_basis_deg+d1] for b in range(u.size(0))])
                for d3 in range(self.fem_basis_deg + 1) ]) for d2 in range(self.fem_basis_deg + 1)]) for d1 in range(self.fem_basis_deg + 1)]).unsqueeze(3)

        # print('* '* 10)
        # print(u_pts_grid.shape)
        # print('* '* 10)
        # exit()

        x_pts = pc[:,:,:,0] - nidx.type_as(pc)*self.hx 
        y_pts = pc[:,:,:,1] - nidy.type_as(pc)*self.hy
        z_pts = pc[:,:,:,2] - nidz.type_as(pc)*self.hz

        xi_pts = (x_pts*2)/self.hx - 1
        eta_pts = (y_pts*2)/self.hy - 1
        chi_pts = (z_pts*2)/self.hz - 1

        N_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)*self.bf_1d_th(chi_pts).unsqueeze(2)
        dN_x_values_pts = self.bf_1d_der_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)*self.bf_1d_th(chi_pts).unsqueeze(2)
        dN_y_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_der_th(eta_pts).unsqueeze(1)*self.bf_1d_th(chi_pts).unsqueeze(2)
        dN_z_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)*self.bf_1d_der_th(chi_pts).unsqueeze(2)

        u_pts = torch.sum(torch.sum(torch.sum(N_values_pts*u_pts_grid,0),0),0) # Field values at pc locations
        u_x_pts = torch.sum(torch.sum(torch.sum(dN_x_values_pts*u_pts_grid,0),0),0)
        u_y_pts = torch.sum(torch.sum(torch.sum(dN_y_values_pts*u_pts_grid,0),0),0)
        u_z_pts = torch.sum(torch.sum(torch.sum(dN_z_values_pts*u_pts_grid,0),0),0)

        R1 = torch.sum(R1, 1)

        res_eikonal = torch.zeros_like(u); res_eikonal = self.Q1_3D_vector_assembly(res_eikonal,R1)


        # print('* '*10)
        # print('u ', u.shape)
        # print('R1 ', R1.shape)
        # print('res eikonal ', res_eikonal.shape)
        # print('PC ', pc.shape)
        # print('normals ', normals.shape)
        # print('N_values_pts ', N_values_pts.shape)
        # print('dN_x_values_pts ', dN_x_values_pts.shape)
        # print('dN_y_values_pts ', dN_y_values_pts.shape)
        # print('dN_z_values_pts ', dN_z_values_pts.shape)
        # print('u_pts ', u_pts.shape)
        # print('u_pts_grid ', u_pts_grid.shape)
        # print('u_x_pts ', u_x_pts.shape)
        # print('u_y_pts ', u_y_pts.shape)
        # print('u_z_pts ', u_z_pts.shape)
        # print('* '*10)

        # exit()
        # Second loss - boundary loss
        sdf_recon_loss = torch.sum((u_pts - 0.0)**2)

        # Third loss - boundary reconstruction
        normals_loss = torch.nn.functional.mse_loss(u_x_pts, normals[:,:,:,0]) + torch.nn.functional.mse_loss(u_y_pts, normals[:,:,:,1]) + torch.nn.functional.mse_loss(u_z_pts, normals[:,:,:,2])

        # res_eikonal = torch.norm(res_eikonal, 'fro')
        # res_eikonal = torch.sum(res_eikonal)

        # reg_loss = torch.exp(-100*(torch.abs(u)))
        # print('* '*10)
        # # print(R1.shape)
        # print(res_eikonal)
        # print(sdf_recon_loss)
        # print(normals_loss)
        # print(reg_loss)
        # print('* '*10)
        # exit()

        loss = res_eikonal + sdf_recon_loss + normals_loss
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        if self.mapping_type == 'no_network':
            return self.network[0], inputs_tensor, forcing_tensor
        elif self.mapping_type == 'network':
            nu = inputs_tensor[:,0:1,:,:].squeeze(1)
            return self.network(nu).unsqueeze(1), inputs_tensor, forcing_tensor

    def configure_optimizers(self):
        """
        Configure optimizer for network parameters
        """
        lr = self.learning_rate
        opts = [torch.optim.LBFGS(self.network.parameters(), lr=1.0, max_iter=5)]
        # opts = [torch.optim.Adam(self.network.parameters(), lr=lr)]
        return opts

    # def training_step(self, batch, batch_idx):
    #     u, inputs_tensor, forcing_tensor = self.forward(batch)
    #     loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
    #     return {"loss": loss_val}

    def on_epoch_end(self):
        # fig, axs = plt.subplots(1, 2, 1, figsize=(6, 6, 6),
        #                     subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True, projection='3d')

        
        # for ax in axs:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_zticks([])
        self.network.eval()
        inputs, forcing = self.dataset[0]

        u, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), 
                                                         forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

        f = forcing_tensor # renaming variable
        
        # init bin widths
        # hx = self.h
        # hy = self.h

        # # init vars for weak formulation
        # N_values = self.Nvalues.type_as(u)
        # gpw = self.gpw.type_as(u)
        
        # # u_gp = self.gauss_pt_evaluation(u)
        # u_x_gp = self.gauss_pt_evaluation_der_x(u)
        # u_y_gp = self.gauss_pt_evaluation_der_y(u)

        # # bin width (reimann sum)
        # trnsfrm_jac = (0.5*hx)*(0.5*hy)
        # JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        
        # # Eikonal Equation
        # eikonal_lhs = (u_x_gp)**2 + (u_y_gp)**2
        # eikonal_rhs = self.h 
        # R1 = JxW * N_values * (eikonal_lhs - eikonal_rhs)

        # # extract diffusivity and boundary conditions here
        # pc = inputs_tensor[:,0:1,:,:].type_as(f)
        # normals = inputs_tensor[:,1:2,:,:].type_as(f)

        # # apply boundary conditions
        # nidx = (pc[:,:,:,0]/self.hx).type(torch.LongTensor).to(pc.device)
        # nidy = (pc[:,:,:,1]/self.hy).type(torch.LongTensor).to(pc.device)

        # u_pts_grid =  torch.stack([
        #         torch.stack([
        #             torch.stack([u[b,0,nidx[b,0,:],nidy[b,0,:]] for b in range(u.size(0))]),
        #             torch.stack([u[b,0,nidx[b,0,:]+1,nidy[b,0,:]] for b in range(u.size(0))])]),
        #         torch.stack([
        #             torch.stack([u[b,0,nidx[b,0,:],nidy[b,0,:]+1] for b in range(u.size(0))]),
        #             torch.stack([u[b,0,nidx[b,0,:]+1,nidy[b,0,:]+1] for b in range(u.size(0))])])
        #         ]).unsqueeze(2)

        # x_pts = pc[:,:,:,0] - nidx.type_as(pc)*self.hx 
        # y_pts = pc[:,:,:,1] - nidy.type_as(pc)*self.hy

        # xi_pts = (x_pts*2)/self.hx - 1
        # eta_pts = (y_pts*2)/self.hy - 1

        # N_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        # dN_x_values_pts = self.bf_1d_der_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        # dN_y_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_der_th(eta_pts).unsqueeze(1)

        # u_pts = torch.sum(torch.sum(N_values_pts*u_pts_grid,0),0)
        # u_x_pts = torch.sum(torch.sum(dN_x_values_pts*u_pts_grid,0),0)
        # u_y_pts = torch.sum(torch.sum(dN_y_values_pts*u_pts_grid,0),0)

        # # Second loss - boundary loss
        # sdf_recon_loss = torch.sum((u_pts - 0.0)**2)

        # # Third loss - boundary reconstruction
        # # normals_loss = torch.nn.functional.mse_loss(u_x_pts, normals[:,:,:,0]) + torch.nn.functional.mse_loss(u_y_pts, normals[:,:,:,1])

        
        # res_eikonal = torch.zeros_like(u); res_eikonal = self.Q1_vector_assembly(res_eikonal,torch.sum(R1,1))

        # print('* '*10)
        # print('u ', u.shape)
        # print('R1 ', R1.shape)
        # print('PC ', pc.shape)
        # print('N_values_pts ', N_values_pts.shape)
        # print('u_pts ', u_pts.shape)
        # print('u_pts_grid ', u_pts_grid.shape)
        # print('u_x_pts ', u_x_pts.shape)
        # print('normals ', normals.shape)
        # print('res_eikonal ', res_eikonal.shape)
        # print('* '*10)
        # exit()

        # u_x_pts = u_x_pts.squeeze().detach().cpu()
        # u_y_pts = u_y_pts.squeeze().detach().cpu()
        # pc = pc.squeeze().detach().cpu()
        # normals = normals.squeeze().detach().cpu()
        u = u.squeeze().detach().cpu()
        # print(u.shape)
        # exit()
        # sdf_boundary_residual = sdf_boundary_residual.squeeze().detach().cpu()
        # res_eikonal = res_eikonal.squeeze().detach().cpu()

        # im0 = axs[0].imshow(u,cmap='jet')
        # fig.colorbar(im0, ax=axs[0])
        # axs[0].set_title('u')

        # im1 = axs[1].imshow(res_eikonal, cmap='jet')
        # fig.colorbar(im1, ax=axs[1])
        # axs[1].set_title('eik res')

        # axs[2].quiver(pc[:,0], pc[:,-1], normals[:,0], normals[:,-1], angles='xy', scale_units='xy')
        # axs[2].set_title('True nrmls')

        # axs[3].quiver(pc[:,0], pc[:,-1], u_x_pts[:], u_y_pts[:], angles='xy', scale_units='xy')
        # axs[3].set_title('Pred nrmls')

        fig, axs = plt.subplots(1, 3, figsize=(6,6),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        # for ax_row in axs:
        #     for ax in ax_row:
        #         ax.set_xticks([])
        #         ax.set_yticks([])

        for ax in axs:
           ax.set_xticks([])
           ax.set_yticks([])


        row_id = 0
        sliceZ = int(self.domain_size / 2)
        im = axs[row_id].imshow(u[sliceZ,:,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id]) #ticks=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        # im = axs[row_id,1].imshow(u[sliceZ,:,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,1])
        # im = axs[row_id,2].imshow(u_exact[sliceZ,:,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,2])
        # im = axs[row_id,3].imshow(diff[sliceZ,:,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,3])
        # y-slices
        row_id = 1
        sliceY = int(self.domain_size / 2)
        im = axs[row_id].imshow(u[:,sliceY,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id]) #ticks=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        # im = axs[row_id,1].imshow(u[:,sliceY,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,1])
        # im = axs[row_id,2].imshow(u_exact[:,sliceY,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,2])
        # im = axs[row_id,3].imshow(diff[:,sliceY,:],cmap='jet'); fig.colorbar(im, ax=axs[row_id,3])
        # # x-slices
        row_id = 2
        sliceX = int(self.domain_size / 2)
        im = axs[row_id].imshow(u[:,:,sliceX],cmap='jet'); fig.colorbar(im, ax=axs[row_id]) #ticks=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        # im = axs[row_id,1].imshow(u[:,:,sliceX],cmap='jet'); fig.colorbar(im, ax=axs[row_id,1])
        # im = axs[row_id,2].imshow(u_exact[:,:,sliceX],cmap='jet'); fig.colorbar(im, ax=axs[row_id,2])
        # im = axs[row_id,3].imshow(diff[:,:,sliceX],cmap='jet'); fig.colorbar(im, ax=axs[row_id,3])

        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')


        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    Nx = Ny = Nz = 32
    LR=3e-4
    mapping_type = 'no_network'
    u_tensor = np.random.randn(1,1,Ny, Nx, Nz)
    if mapping_type == 'no_network':
        network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    elif mapping_type == 'network':
        # network = AE(in_channels=1, out_channels=1, dims=Nx, n_downsample=1)
        network = ConvNet(902, 256, [500 for i in range(6)], nonlin=torch.sin)
    dataset = PC3D(domain_size=32)
    basecase = Eiqonal(network, dataset, batch_size=1, fem_basis_deg=1, domain_size=Nx, domain_length=1.0, learning_rate=LR, mapping_type=mapping_type, nsd=3)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="curve_reconstruction_sphere")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=500, deterministic=True, profiler="simple")

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