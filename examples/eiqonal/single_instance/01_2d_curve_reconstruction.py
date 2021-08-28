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
    def __init__(self, filename, domain_size=64):
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

        x = np.linspace(0,1,domain_size)
        y = np.linspace(0,1,domain_size)

        xx , yy = np.meshgrid(x,y)
        self.x = xx
        self.y = yy
        
        # bc1 will be source, sdf will be set to 0 at these locations
        self.bc1 = im2pcpix(img)

        # self.bc1 = np.zeros_like(img)
        # self.bc1[:,0] = 1
        # self.bc1[:,-1] = 1
        # self.bc1[0,:] = 1
        # self.bc1[-1,:] = 1

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
        inputs = np.array([self.xx, self.yy, self.domain])
        forcing = np.ones_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)


class Eiqonal(DiffNet2DFEM):
    """docstring for Eiqonal"""
    def __init__(self, network, dataset, **kwargs):
        super(Eiqonal, self).__init__(network, dataset, **kwargs)

    def loss(self, u, inputs_tensor, forcing_tensor):

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]
        bc3 = inputs_tensor[:,3:4,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5, u*0.0, u)
        # dist = torch.where(bc1>0.5, (u)**2, u*0).mean()
        

        u_gp = self.gauss_pt_evaluation(u)
        bc1_gp = self.gauss_pt_evaluation(bc1)
        # bc2_gp = self.gauss_pt_evaluation(bc2)
        # bc3_gp = self.gauss_pt_evaluation(bc3)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        # dist2 = torch.where(bc1_gp>=0.1, (1 - (u_x_gp*bc3_gp + u_y_gp*bc2_gp))**2, bc2_gp*0).sum()
        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(u_gp)
        res_elmwise1 = transformation_jacobian * ((1 -( u_x_gp**2 + u_y_gp**2 ) )**2)
        # res_elmwise2 = transformation_jacobian * (1.0 - u_x_gp*bc2_gp - u_y_gp*bc3_gp) 
        res_elmwise = torch.sum(res_elmwise1, 1) #  + torch.sum(res_elmwise2, 1) 
        loss = torch.mean(res_elmwise) 
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        return self.network(inputs_tensor), inputs_tensor, forcing_tensor
        # return self.network[0], inputs_tensor, forcing_tensor

    # def configure_optimizers(self):
    #     """
    #     Configure optimizer for network parameters
    #     """
    #     lr = self.learning_rate
    #     opts = [torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
    #     # opts = [torch.optim.Adam(self.network)]
    #     return opts, []

    def on_epoch_end(self):
        fig, axs = plt.subplots(1, 4, figsize=(2*4,1.2),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        self.network.eval()
        inputs, forcing = self.dataset[0]

        u, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]
        bc3 = inputs_tensor[:,3:4,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,u*0.001,u)

        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(u_gp)
        res_elmwise = transformation_jacobian * (((u_x_gp**2 + u_y_gp**2) - 1.0).abs())
        res_elmwise = torch.sum(res_elmwise, 1) 

        k = bc1.squeeze().detach().cpu()
        bc2 = bc2.squeeze().detach().cpu()
        bc3 = bc3.squeeze().detach().cpu()
        u = u.squeeze().detach().cpu()
        res_elmwise = res_elmwise.squeeze().detach().cpu()

        im0 = axs[0].imshow(k,cmap='jet')
        fig.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(u,cmap='jet')
        fig.colorbar(im1, ax=axs[1])  
        # im2 = axs[2].imshow(u>0,cmap='Greys')
        # fig.colorbar(im2, ax=axs[2])
        # im3 = axs[3].imshow(res_elmwise,cmap='jet', vmin=0.0, vmax=1.0)
        # fig.colorbar(im3, ax=axs[3])  

        im2 = axs[2].imshow(bc2,cmap='jet')
        fig.colorbar(im2, ax=axs[2])
        im3 = axs[3].imshow(bc3,cmap='jet')
        fig.colorbar(im3, ax=axs[3])  

        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    u_tensor = np.ones((1,1,256,256))
    # u_tensor = (-1)*np.random.rand(1,1,256,256)
    # network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    network = ImplicitConv(in_channels=4, out_channels=1)
    dataset = PCVox('../../poisson/ImageDataset/apple-20.png', domain_size=256)
    # dataset = PCVox('../../poisson/ImageDataset/barbell-2.png', domain_size=256)
    basecase = Eiqonal(network, dataset, batch_size=1, fem_basis_deg=1)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="curve_reconstruction")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=2000, deterministic=True, profiler="simple")

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