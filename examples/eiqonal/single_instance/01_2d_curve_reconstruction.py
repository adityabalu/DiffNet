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
from DiffNet.networks.autoencoders import AE
from DiffNet.DiffNetFEM import DiffNet2DFEM
from DiffNet.DiffNetFDM import DiffNetFDM
import PIL
from torch.utils import data




def im_io(filepath):
    image = io.imread(filepath).astype(bool).astype(float)
    

    return im2pc(image)

def im2pc(image, nx, ny):
    pc = []
    normals = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] == 1.0:
                boundary = 0
                if i < image.shape[0] - 1:
                    if image[i+1,j] == 0:
                        boundary = 1
                if j < image.shape[1] - 1:
                    if image[i,j+1] == 0:
                        boundary = 1
                if i > 0:
                    if image[i-1,j] == 0:
                        boundary = 1
                if j > 0:
                    if image[i,j-1] == 0:
                        boundary = 1
                if i < image.shape[0] - 1  and j < image.shape[1] - 1:
                    if image[i+1,j+1] == 0:
                        boundary = 1
                if i < image.shape[0] - 1  and j > 0:
                    if image[i+1,j-1] == 0:
                        boundary = 1
                if i > 0 and j < image.shape[1] - 1:
                    if image[i-1,j+1] == 0:
                        boundary = 1
                if i > 0 and j > 0:
                    if image[i-1,j-1] == 0:
                        boundary = 1
                if boundary == 1:
                    pc.append([i+0.5,j+0.5])
                    normals.append([nx[i,j]/(nx[i,j]**2 + ny[i,j]**2), ny[i,j]/(nx[i,j]**2 + ny[i,j]**2)])
    return np.array(pc), np.array(normals)


class PCVox(data.Dataset):
    'PyTorch dataset for PCVox'
    def __init__(self, filename, domain_size=128):
        """
        Initialization
        """
        file, ext = os.path.splitext(filename)
        if ext in ['.png', '.jpg', '.bmp', '.tiff']:
            img = PIL.Image.open(filename).convert('L')
            # img = PIL.Image.open(filename).convert('L').resize((700, 300))
            img = (np.asarray(img)>0).astype('float')
        else:
            raise ValueError('invalid extension; extension not supported')

        # Define kernel for x differences
        kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        # Define kernel for y differences
        ky = np.array([[1,2,1] ,[0,0,0], [-1,-2,-1]])
        # Perform x convolution
        nx = ndimage.convolve(img,kx)
        # Perform y convolution
        ny = ndimage.convolve(img,ky)
        nx = np.divide(nx,(nx**2 + ny**2), out=np.zeros_like(nx), where=((nx**2 + ny**2)!=0))
        ny = np.divide(ny,(nx**2 + ny**2), out=np.zeros_like(ny), where=((nx**2 + ny**2)!=0))

        # bc1 will be source, sdf will be set to 0.5 at these locations
        self.pc, self.normals = im2pc(img,nx,ny)
        self.pc = self.pc/(img.shape[0])

        # pt_cloud = []
        # for _ in range(1000):
        #     vec = np.random.randn(2)
        #     vec /= 4*np.linalg.norm(vec)
        #     pt_cloud.append(vec)
        # pt_cloud = np.array(pt_cloud)
        # self.normals = pt_cloud*4.0
        # self.pc = pt_cloud + 0.5

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

    def lossFDM(self, u, inputs_tensor, forcing_tensor):
        f = forcing_tensor # renaming variable

        # init bin widths
        hx = self.h
        hy = self.h

        u_x = self.derivative_x(self.pad(u))
        u_y = self.derivative_y(self.pad(u))

        R1 = (u_x**2 + u_y**2) - 1.0


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

        # print(xi_pts, eta_pts)

        N_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        dN_x_values_pts = self.bf_1d_der_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        dN_y_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_der_th(eta_pts).unsqueeze(1)

        u_pts = torch.sum(torch.sum(N_values_pts*u_pts_grid,0),0)
        u_x_pts = torch.sum(torch.sum(dN_x_values_pts*u_pts_grid,0),0)
        u_y_pts = torch.sum(torch.sum(dN_y_values_pts*u_pts_grid,0),0)

        # Second loss - boundary loss
        sdf_recon_loss = torch.sum(u_pts**2)

        # Third loss - boundary reconstruction
        normals_loss = torch.nn.functional.mse_loss(u_x_pts, normals[:,:,:,0], reduction='sum') + torch.nn.functional.mse_loss(u_y_pts, normals[:,:,:,1], reduction='sum')

        # res_elmwise2 = JxW * normals
        # res_elmwise2 = JxW * (normals_lhs - normals_rhs)
        # Assemble 
        # print('***'*10)
        # print(torch.sum(res_elmwise1, 1).shape)
        # print('***'*10)
        # R1 = torch.zeros_like(u); R1 = self.Q1_vector_assembly(R1, torch.sum(res_elmwise1, 1))


        # reg_loss = torch.exp(-100*(u_reg**2))
        loss = torch.mean(R1**2) + sdf_recon_loss + normals_loss
        # loss = torch.norm(R1, 'fro') + 1000*sdf_recon_loss + 10*normals_loss
        # loss = torch.mean(res_elmwise1**2) + sdf_recon_loss
        return loss


    def loss(self, u, inputs_tensor, forcing_tensor):
        f = forcing_tensor # renaming variable

        # init bin widths
        hx = self.h
        hy = self.h

        # init vars for weak formulation
        N_values = self.Nvalues.type_as(u)
        gpw = self.gpw.type_as(u)
        
        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        # Eikonal Residual on the domain
        trnsfrm_jac = (0.5*hx)*(0.5*hy)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        # eikonal_lhs = N_values*(u_x_gp**2 + u_y_gp**2)
        # eikonal_rhs = 1.0
        # res_elmwise1 = torch.sum(JxW * (eikonal_lhs), 2)# \nabla \phi - 1 = 0  JxW addresses discretization of domain
        # R1 = torch.zeros_like(u); R1 = self.Q1_vector_assembly(R1, res_elmwise1)

        R1 = torch.sum(u_x_gp**2 + u_y_gp**2, 2) - 1.0


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

        # print(xi_pts, eta_pts)

        N_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        dN_x_values_pts = self.bf_1d_der_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        dN_y_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_der_th(eta_pts).unsqueeze(1)

        u_pts = torch.sum(torch.sum(N_values_pts*u_pts_grid,0),0)
        u_x_pts = torch.sum(torch.sum(dN_x_values_pts*u_pts_grid,0),0)
        u_y_pts = torch.sum(torch.sum(dN_y_values_pts*u_pts_grid,0),0)

        # Second loss - boundary loss
        sdf_recon_loss = torch.sum((u_pts - 0.0)**2)

        # Third loss - boundary reconstruction
        normals_loss = torch.nn.functional.mse_loss(u_x_pts, normals[:,:,:,0]) + torch.nn.functional.mse_loss(u_y_pts, normals[:,:,:,1])

        # res_elmwise2 = JxW * normals
        # res_elmwise2 = JxW * (normals_lhs - normals_rhs)
        # Assemble 
        # print('***'*10)
        # print(torch.sum(res_elmwise1, 1).shape)
        # print('***'*10)
        # R1 = torch.zeros_like(u); R1 = self.Q1_vector_assembly(R1, torch.sum(res_elmwise1, 1))

        reg_loss = torch.exp(-100*(torch.abs(u)))
        if self.current_epoch < 3:
            loss = 10*torch.norm(R1, 'fro') + 1000*sdf_recon_loss  + 5*reg_loss
        else:
            loss = 100*torch.norm(R1, 'fro') + 1000*sdf_recon_loss 
        print()
        print('R1:', torch.norm(R1, 'fro').item())
        print('SDF loss:', sdf_recon_loss.item())
        print('normals loss:', normals_loss.item())
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        if self.mapping_type == 'no_network':
            return self.network[0], inputs_tensor, forcing_tensor
        elif self.mapping_type == 'network':
            nu = inputs_tensor[:,0:1,:,:]
            return self.network(nu), inputs_tensor, forcing_tensor

    def configure_optimizers(self):
        """
        Configure optimizer for network parameters
        """
        lr = self.learning_rate
        opts = [torch.optim.LBFGS(self.network.parameters(), lr=1.0, max_iter=5)]
        # opts = [torch.optim.Adam(self.network.parameters(), lr=lr)]
        return opts

    def training_step(self, batch, batch_idx):
    # def training_step(self, batch, batch_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        if self.loss_type == 'FEM':
            loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
        elif self.loss_type == 'FDM':
            loss_val = self.lossFDM(u, inputs_tensor, forcing_tensor).mean()
        # loss_val = self.loss_resmin(u, inputs_tensor, forcing_tensor).mean()
        # loss_val = self.loss_resmin_mass(u, inputs_tensor, forcing_tensor).mean()
        # loss_val = self.loss_matvec(u, inputs_tensor, forcing_tensor).mean()
        # self.log('PDE_loss', loss_val.item())
        # self.log('loss', loss_val.item())
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
        N_values = self.Nvalues.type_as(f)
        gpw = self.gpw.type_as(f)
        

        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        # Eikonal Residual on the domain
        trnsfrm_jac = (0.5*hx)*(0.5*hy)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        eikonal_lhs = (u_x_gp**2 + u_y_gp**2)
        eikonal_rhs = 1.0
        res_elmwise1 = JxW * (eikonal_lhs)# \nabla \phi - 1 = 0  JxW addresses discretization of domain
        R1 = torch.zeros_like(u); R1 = self.Q1_vector_assembly(R1, res_elmwise1)
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

        # print(xi_pts, eta_pts)

        N_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        dN_x_values_pts = self.bf_1d_der_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        dN_y_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_der_th(eta_pts).unsqueeze(1)

        u_pts = torch.sum(torch.sum(N_values_pts*u_pts_grid,0),0)
        u_x_pts = torch.sum(torch.sum(dN_x_values_pts*u_pts_grid,0),0)
        u_y_pts = torch.sum(torch.sum(dN_y_values_pts*u_pts_grid,0),0)

        # Second loss - boundary loss
        sdf_recon_loss = torch.sum(u_pts**2)

        # Third loss - boundary reconstruction
        normals_loss = torch.sum((1.0 - (u_x_pts*normals[:,:,:,0] + u_y_pts*normals[:,:,:,1]))**2)

        # res_elmwise2 = JxW * normals
        # res_elmwise2 = JxW * (normals_lhs - normals_rhs)
        # Assemble 
        # print('***'*10)
        # print(torch.sum(res_elmwise1, 1).shape)
        # print('***'*10)
        # R1 = torch.zeros_like(u); R1 = self.Q1_vector_assembly(R1, torch.sum(res_elmwise1, 1))


        u = u.squeeze().detach().cpu()
        # sdf_boundary_residual = sdf_boundary_residual.squeeze().detach().cpu()
        R1 = R1.squeeze().detach().cpu()

        im0 = axs[0].imshow(u,cmap='jet')
        fig.colorbar(im0, ax=axs[0])
        axs[0].set_title('u')

        im1 = axs[1].imshow(R1, vmin=0, vmax=1, cmap='gray')
        fig.colorbar(im1, ax=axs[1])
        axs[1].set_title('Eikonal residual')

        im2 = axs[2].imshow(abs(u), cmap='jet')
        fig.colorbar(im2, ax=axs[2])
        axs[2].set_title('Unsigned Distance Field')

        # im3 = axs[3].imshow(bc1,cmap='jet')
        # fig.colorbar(im3, ax=axs[3])  
        # axs[3].set_title('source')

        # im4 = axs[4].imshow(bc2,cmap='jet')
        # fig.colorbar(im4, ax=axs[4])
        # axs[4].set_title('nrmls x')

        # im5 = axs[5].imshow(bc3,cmap='jet')
        # fig.colorbar(im5, ax=axs[5])  
        # axs[5].set_title('nrmls y')

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
        network = AE(in_channels=1, out_channels=1, dims=Nx, n_downsample=3)
    dataset = PCVox('../ImageDataset/bonefishes-1.png', domain_size=256)
    basecase = Eiqonal(network, dataset, batch_size=1, fem_basis_deg=1, domain_size=Nx, domain_length=1.0, learning_rate=LR, mapping_type=mapping_type, loss_type='FEM')

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="curve_reconstruction_final")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=2500, deterministic=True, profiler="simple")

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