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
from DiffNet.networks.conv11 import ImplicitConv
from DiffNet.DiffNetFEM import DiffNet2DFEM
import PIL
from torch.utils import data



def im_io(filepath):
    image = io.imread(filepath).astype(bool).astype(float)
    

    return im2pc(image)


def im2pcpix(image):
    pix_out = np.zeros_like(image)
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
                    pix_out[i,j] = 1.0
    return pix_out


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
        self.domain = np.ones_like(img)


        # Define kernel for x differences
        kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        # Define kernel for y differences
        ky = np.array([[1,2,1] ,[0,0,0], [-1,-2,-1]])
        # Perform x convolution
        nx=ndimage.convolve(img,kx)
        # Perform y convolution
        ny=ndimage.convolve(img,ky)

        # sdf = ndimage.distance_transform_edt(img)
        # sdf += (-1)*ndimage.distance_transform_edt(1-img)
        # plt.figure()
        # plt.imshow((-1)*sdf/(sdf.shape[0]),cmap='jet', vmin=-1.0, vmax=1.0)
        # plt.colorbar()
        # plt.show()
        # exit()
        
        # bc1 will be source, sdf will be set to 0.5 at these locations
        self.bc1 = im2pcpix(img)

        # bc2 will be the normals in x direction
        self.bc2 = np.divide(nx,(nx**2 + ny**2), out=np.zeros_like(nx), where=((nx**2 + ny**2)!=0))
        # bc2 will be the normals in x direction
        self.bc3 = np.divide(ny,(nx**2 + ny**2), out=np.zeros_like(ny), where=((nx**2 + ny**2)!=0))
        # print(self.bc1.shape, self.bc2.shape, self.bc3.shape)
        self.n_samples = 100    

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2, self.bc3])
        forcing = np.ones_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)


class Eiqonal(DiffNet2DFEM):
    """docstring for Eiqonal"""
    def __init__(self, network, dataset, **kwargs):
        super(Eiqonal, self).__init__(network, dataset, **kwargs)

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
        N_values = self.Nvalues.type_as(f)
        gpw = self.gpw.type_as(f)
        
        # extract diffusivity and boundary conditions here
        bc1 = inputs_tensor[:,1:2,:,:]  # sdf = 0.5 at boundary else 0.
        bc2 = inputs_tensor[:,2:3,:,:]  # normals in the x-direction
        bc3 = inputs_tensor[:,3:4,:,:]  # normals in the y-direction
        u = torch.unsqueeze(u, 0)

        # apply boundary conditions
        # build reconstruction boundary loss
        zeros = torch.zeros_like(u)
        sdf_recon = torch.where(bc1>0.5, u, zeros)

        # u = torch.where(bc1>0.5, u*0.0, u) # this may give misleading gradient if we are trying to learn B.C.s

        u_gp = self.gauss_pt_evaluation(u)
        bc2_gp = self.gauss_pt_evaluation(bc2)
        bc3_gp = self.gauss_pt_evaluation(bc3)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        # bin width (reimann sum)
        trnsfrm_jac = (0.5*hx)*(0.5*hy)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        # First loss - eikonal eqn ie   grad(u) = 1
        eikonal_lhs = (u_x_gp)**2 + (u_y_gp)**2
        # eikonal_rhs = 1.0
        eikonal_rhs = self.h**2
        res_elmwise1 = JxW * N_values * (eikonal_lhs - eikonal_rhs) # \nabla \phi - 1 = 0  JxW addresses discretization of domain

        # First loss regularizer
        # res_elmwise1_regularizer = torch.exp(-100*torch.abs(u_gp))

        # Second loss - normals
        # normals = 1.0 - (u_x_gp*bc2_gp + u_y_gp*bc3_gp) 
        # normals_lhs =  (N_values**2 * u_x_gp * bc2_gp) + (N_values**2 * u_y_gp * bc3_gp) # treat this as a directional eikonal??
        # normals_rhs = N_values * 1. 
        # normals = (u_x_gp - bc2_gp) + (u_y_gp - bc3_gp) # this makes more sense ie mse(pred_u_normals, normals)
        # res_elmwise2 = JxW * normals
        # res_elmwise2 = JxW * (normals_lhs - normals_rhs)

        # Third loss - boundary reconstruction
        sdf_recon_loss = torch.sum(sdf_recon**2)

        # Assemble 
        R1 = torch.zeros_like(u); R1 = self.Q1_vector_assembly(R1, torch.sum(res_elmwise1, 1))
        # R2 = torch.zeros_like(u); R2 = self.Q1_vector_assembly(R2, res_elmwise2)

        # print('*'*10)
        # print('u_x_gp = ', u_x_gp.shape)
        # print('JxW = ', JxW.shape)
        # print('N_values = ', N_values.shape)
        # print('JxW * N_values = ', (JxW * N_values).shape)
        # print('res_elmwise1 = ', res_elmwise1.shape)
        # print('res_elmwise2 = ', res_elmwise2.shape)
        # print('sum of res_elmwise1 = ', torch.sum(res_elmwise1, 1).shape)
        # print('sum of res_elmwise2 = ', torch.sum(res_elmwise2, 1).shape)
        # print('sdf_loss = ', sdf_recon_loss.shape)
        # print('*'*10)
        # exit()

        # ********** From elastic single instance
        # temp1 =  torch.Size([1, 4, 4, 31, 31])
        # R_split_1 =  torch.Size([1, 4, 31, 31])
        # R1 =  torch.Size([1, 1, 32, 32])
        # **********
        



        # res_elmwise = torch.sum(res_elmwise1, 1) + torch.sum(res_elmwise2, 1)  + sdf_recon_loss # + torch.sum(res_elmwise1_regularizer, 1)
        # res_elmwise = torch.norm(R1, 'fro') + torch.norm(R2, 'fro') + sdf_recon_loss
        loss = torch.norm(R1, 'fro') + sdf_recon_loss
        # loss = torch.mean(res_elmwise) 
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        return self.network[0], inputs_tensor, forcing_tensor

    def configure_optimizers(self):
        """
        Configure optimizer for network parameters
        """
        lr = self.learning_rate
        opt = torch.optim.Adam(self.network, lr=lr)
        return opt

    def on_epoch_end(self):
        fig, axs = plt.subplots(1, 5, figsize=(2*7,4),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        self.network.eval()
        inputs, forcing = self.dataset[0]

        u, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), 
                                                         forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:] 
        bc3 = inputs_tensor[:,3:4,:,:]

        # u_bc = torch.where(bc1>0.5,u*0.001,u)

        # build reconstruction boundary loss
        zeros = torch.zeros_like(u)
        sdf_boundary_residual = torch.where(bc1>0.5, u, zeros)

        bc1 = bc1.squeeze().detach().cpu()
        bc2 = bc2.squeeze().detach().cpu()
        bc3 = bc3.squeeze().detach().cpu()
        u = u.squeeze().detach().cpu()
        sdf_boundary_residual = sdf_boundary_residual.squeeze().detach().cpu()

        im0 = axs[0].imshow(u,cmap='jet')
        fig.colorbar(im0, ax=axs[0])
        axs[0].set_title('u')

        im1 = axs[1].imshow(sdf_boundary_residual, vmin=0, vmax=1, cmap='gray')
        fig.colorbar(im1, ax=axs[1])
        axs[1].set_title('sdf res')

        im2 = axs[2].imshow(bc1,cmap='jet')
        fig.colorbar(im2, ax=axs[2])  
        axs[2].set_title('source')

        im3 = axs[3].imshow(bc2,cmap='jet')
        fig.colorbar(im3, ax=axs[3])
        axs[3].set_title('nrmls x')

        im4 = axs[4].imshow(bc3,cmap='jet')
        fig.colorbar(im4, ax=axs[4])  
        axs[4].set_title('nrmls y')

        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    img_size = 256
    u_tensor = np.ones((1,1,256,256))
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor[:,0,:,:]), requires_grad=True)])
    dataset = PCVox('images/bell-8.png', domain_size=256)
    basecase = Eiqonal(network, dataset, batch_size=1, fem_basis_deg=1, domain_size=img_size, domain_length=img_size)

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
        max_epochs=1500, deterministic=True, profiler="simple")

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