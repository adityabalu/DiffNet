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
from DiffNet.DiffNetFEM import DiffNet2DFEM
from DiffNet.DiffNetFDM import DiffNetFDM
import PIL
from torch.utils import data

import torch
from NURBSDiff.curve_eval import CurveEval



class NURBSPC(data.Dataset):
    'PyTorch dataset for PCVox'
    def __init__(self, domain_size=256):
        """
        Initialization
        """
        control_points = np.zeros((16, 2))
        control_points[7:9:,0] = np.ones((1,2))

        int_pts_x = np.linspace(start=.1, stop=.9, num=5)
        int_pts_x_neg = np.flipud(int_pts_x)

        int_pts_y = np.random.uniform(low=0.2, high=0.7, size=5)
        int_pts_y_neg = np.flipud(int_pts_y) * -1.

        int_pts_pos = np.dstack((int_pts_x, int_pts_y)).squeeze()
        int_pts_neg = np.dstack((int_pts_x_neg, int_pts_y_neg)).squeeze()

        control_points[2:7,:] = int_pts_pos
        control_points[9:14,:] = int_pts_neg

        curve_eval = CurveEval(m=16, dimension=2, p=2, out_dim=1000, method='tc', dvc='cpu')
        curve_eval.eval()
        weights = torch.ones(1, 16, 1)
        output = curve_eval(torch.cat((torch.from_numpy(control_points).unsqueeze(0), weights), axis=-1).float())
        

        self.pc = output.numpy() + 0.5
        self.normals = output.numpy() * 4.0


        self.domain = np.ones((domain_size,domain_size))
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

    



class Eiqonal(DiffNet2DFEM,DiffNetFDM):
    """docstring for Eiqonal"""
    def __init__(self, network, dataset, **kwargs):
        super(Eiqonal, self).__init__(network, dataset, **kwargs)
        self.mapping_type = kwargs.get('mapping_type', 'no_network')
        self.loss_type = kwargs.get('loss_type', 'FDM')

        self.network = network

    def Q1_vector_assembly(self, Aglobal, Aloc_all):
        Aglobal[:,0, 0:-1, 0:-1] += Aloc_all[:,0, :, :]
        Aglobal[:,0, 0:-1, 1:  ] += Aloc_all[:,1, :, :]
        Aglobal[:,0, 1:  , 0:-1] += Aloc_all[:,2, :, :]
        Aglobal[:,0, 1:  , 1:  ] += Aloc_all[:,3, :, :]
        return Aglobal


    def loss(self, u, inputs_tensor, forcing_tensor):
        f = forcing_tensor # renaming variable

        # init bin widths
        hx = self.h
        hy = self.h

        # init vars for weak formulation
        N_values = self.Nvalues.type_as(u)
        gpw = self.gpw.type_as(u)
        
        # u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        # bin width (reimann sum)
        trnsfrm_jac = (0.5*hx)*(0.5*hy)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        
        # Eikonal Equation
        eikonal_lhs = (u_x_gp)**2 + (u_y_gp)**2
        eikonal_rhs = self.h 
        R1 = JxW * N_values * (eikonal_lhs - eikonal_rhs)

        # extract diffusivity and boundary conditions here
        pc = inputs_tensor[:,0:1,:,:].type_as(f)
        normals = inputs_tensor[:,1:2,:,:].type_as(f)

        # apply boundary conditions
        nidx = (pc[:,:,:,0]/self.hx).type(torch.LongTensor).to(pc.device)
        nidy = (pc[:,:,:,1]/self.hy).type(torch.LongTensor).to(pc.device)

        u_pts_grid =  torch.stack([
                torch.stack([
                    torch.stack([u[b,0,nidx[b,0,:],nidy[b,0,:]] for b in range(u.size(0))]),
                    torch.stack([u[b,0,nidx[b,0,:]+1,nidy[b,0,:]] for b in range(u.size(0))])]),
                torch.stack([
                    torch.stack([u[b,0,nidx[b,0,:],nidy[b,0,:]+1] for b in range(u.size(0))]),
                    torch.stack([u[b,0,nidx[b,0,:]+1,nidy[b,0,:]+1] for b in range(u.size(0))])])
                ]).unsqueeze(2)

        x_pts = pc[:,:,:,0] - nidx.type_as(pc)*self.hx 
        y_pts = pc[:,:,:,1] - nidy.type_as(pc)*self.hy

        xi_pts = (x_pts*2)/self.hx - 1
        eta_pts = (y_pts*2)/self.hy - 1

        N_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        dN_x_values_pts = self.bf_1d_der_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        dN_y_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_der_th(eta_pts).unsqueeze(1)

        u_pts = torch.sum(torch.sum(N_values_pts*u_pts_grid,0),0) # Field values at pc locations
        u_x_pts = torch.sum(torch.sum(dN_x_values_pts*u_pts_grid,0),0)
        u_y_pts = torch.sum(torch.sum(dN_y_values_pts*u_pts_grid,0),0)


        # print('* '*10)
        # print('u ', u.shape)
        # print('R1 ', R1.shape)
        # print('PC ', pc.shape)
        # print('N_values_pts ', N_values_pts.shape)
        # print('u_pts ', u_pts.shape)
        # print('u_pts_grid ', u_pts_grid.shape)
        # print('u_x_pts ', u_x_pts.shape)
        # print('normals ', normals.shape)
        # print('* '*10)

        # exit()
        # Second loss - boundary loss
        sdf_recon_loss = torch.sum((u_pts - 0.0)**2)

        # Third loss - boundary reconstruction
        normals_loss = torch.nn.functional.mse_loss(u_x_pts, normals[:,:,:,0]) + torch.nn.functional.mse_loss(u_y_pts, normals[:,:,:,1])

        R1 = torch.sum(R1, 1)

        res_eikonal = torch.zeros_like(u); res_eikonal = self.Q1_vector_assembly(res_eikonal,R1)
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

        loss = res_eikonal + sdf_recon_loss + (100.0 * normals_loss)# + (reg_loss)
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

    def training_step(self, batch, batch_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
        return {"loss": loss_val}

    def on_epoch_end(self):
        fig, axs = plt.subplots(1, 3, figsize=(2*7,4),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        self.network.eval()
        inputs, forcing = self.dataset[0]

        u, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), 
                                                         forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

        f = forcing_tensor # renaming variable
        
        # init bin widths
        hx = self.h
        hy = self.h

        # init vars for weak formulation
        N_values = self.Nvalues.type_as(u)
        gpw = self.gpw.type_as(u)
        
        # u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        # bin width (reimann sum)
        trnsfrm_jac = (0.5*hx)*(0.5*hy)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        
        # Eikonal Equation
        eikonal_lhs = (u_x_gp)**2 + (u_y_gp)**2
        eikonal_rhs = self.h 
        R1 = JxW * N_values * (eikonal_lhs - eikonal_rhs)

        # extract diffusivity and boundary conditions here
        pc = inputs_tensor[:,0:1,:,:].type_as(f)
        normals = inputs_tensor[:,1:2,:,:].type_as(f)

        # apply boundary conditions
        nidx = (pc[:,:,:,0]/self.hx).type(torch.LongTensor).to(pc.device)
        nidy = (pc[:,:,:,1]/self.hy).type(torch.LongTensor).to(pc.device)

        u_pts_grid =  torch.stack([
                torch.stack([
                    torch.stack([u[b,0,nidx[b,0,:],nidy[b,0,:]] for b in range(u.size(0))]),
                    torch.stack([u[b,0,nidx[b,0,:]+1,nidy[b,0,:]] for b in range(u.size(0))])]),
                torch.stack([
                    torch.stack([u[b,0,nidx[b,0,:],nidy[b,0,:]+1] for b in range(u.size(0))]),
                    torch.stack([u[b,0,nidx[b,0,:]+1,nidy[b,0,:]+1] for b in range(u.size(0))])])
                ]).unsqueeze(2)

        x_pts = pc[:,:,:,0] - nidx.type_as(pc)*self.hx 
        y_pts = pc[:,:,:,1] - nidy.type_as(pc)*self.hy

        xi_pts = (x_pts*2)/self.hx - 1
        eta_pts = (y_pts*2)/self.hy - 1

        N_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        dN_x_values_pts = self.bf_1d_der_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        dN_y_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_der_th(eta_pts).unsqueeze(1)

        u_pts = torch.sum(torch.sum(N_values_pts*u_pts_grid,0),0)
        u_x_pts = torch.sum(torch.sum(dN_x_values_pts*u_pts_grid,0),0)
        u_y_pts = torch.sum(torch.sum(dN_y_values_pts*u_pts_grid,0),0)

        # Second loss - boundary loss
        sdf_recon_loss = torch.sum((u_pts - 0.0)**2)

        # Third loss - boundary reconstruction
        # normals_loss = torch.nn.functional.mse_loss(u_x_pts, normals[:,:,:,0]) + torch.nn.functional.mse_loss(u_y_pts, normals[:,:,:,1])

        
        res_eikonal = torch.zeros_like(u); res_eikonal = self.Q1_vector_assembly(res_eikonal,torch.sum(R1,1))

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

        u_x_pts = u_x_pts.squeeze().detach().cpu()
        u_y_pts = u_y_pts.squeeze().detach().cpu()
        pc = pc.squeeze().detach().cpu()
        normals = normals.squeeze().detach().cpu()
        u = u.squeeze().detach().cpu()
        # sdf_boundary_residual = sdf_boundary_residual.squeeze().detach().cpu()
        res_eikonal = res_eikonal.squeeze().detach().cpu()


        im0 = axs[0].imshow(u,cmap='jet')
        fig.colorbar(im0, ax=axs[0])
        axs[0].set_title('u')

        im1 = axs[1].imshow(res_eikonal, vmin=0, vmax=1, cmap='gray')
        fig.colorbar(im1, ax=axs[1])
        axs[1].set_title('eik res')

        # print(pc.shape)
        # print(pc[:,0].shape)
        # print(pc[:,1].shape)
        # exit()
        axs[2].scatter(pc[:,0], pc[:,1])
        axs[2].set_title('geometry')

        # axs[2].quiver(pc[:,0], pc[:,-1], normals[:,0], normals[:,-1], angles='xy', scale_units='xy')
        # axs[2].set_title('True nrmls')

        # axs[3].quiver(pc[:,0], pc[:,-1], u_x_pts[:], u_y_pts[:], angles='xy', scale_units='xy')
        # axs[3].set_title('Pred nrmls')


        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    Nx = Ny = 256
    LR=3e-4
    mapping_type = 'no_network'
    u_tensor = np.random.randn(1,1,Ny, Nx)
    if mapping_type == 'no_network':
        network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    elif mapping_type == 'network':
        # network = AE(in_channels=1, out_channels=1, dims=Nx, n_downsample=1)
        network = ConvNet(902, 256, [500 for i in range(6)], nonlin=torch.sin)
    dataset = NURBSPC()
    basecase = Eiqonal(network, dataset, batch_size=1, fem_basis_deg=1, domain_size=Nx, domain_length=1.0, learning_rate=LR, mapping_type=mapping_type, loss_type='FEM')

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="curve_reconstruction_nurbs")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=1000, deterministic=True, profiler="simple")

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