import os
import sys
import json
import torch
import numpy as np
import math
import scipy.io
import matplotlib
from scipy import ndimage
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
from DiffNet.DiffNetFEM import DiffNet2DFEM
import PIL
from torch.utils import data


class PCVox(data.Dataset):
    'PyTorch dataset for PCVox'
    def __init__(self, filename, domain_size=128):
        """
        Initialization
        """
        points = np.load('point_cloud.npz')['arr_0']
        normals = np.load('normals.npz')['arr_0']
        # bc1 will be source, sdf will be set to 0.5 at these locations
        self.normals = normals[:,:,:2]
        self.pc = points*0.45 + 0.5
        self.area = np.zeros((self.pc.shape[0], self.pc.shape[1], 1))
        self.area[:, 1:-1, 0] = np.sum((self.pc[:,1:-1,:] - self.pc[:,0:-2,:])**2, -1)*0.5 + np.sum((self.pc[:,2:,:] - self.pc[:,1:-1,:])**2, -1)*0.5 
        self.area[:, 0, 0] = np.sum((self.pc[:,1,:] - self.pc[:,0,:])**2, -1)*0.5 + np.sum((self.pc[:,-1,:] - self.pc[:,0,:])**2, -1)*0.5 
        self.area[:,-1, 0] = np.sum((self.pc[:,-1,:] - self.pc[:,-2,:])**2, -1)*0.5 + np.sum((self.pc[:,-1,:] - self.pc[:,0,:])**2, -1)*0.5 
        self.domain = np.ones((domain_size,domain_size))
        self.domain_size = domain_size
        self.n_samples = 100

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'

        inputs = np.concatenate((self.pc[0:1,:,:], self.normals[0:1,:,:], self.area[0:1,:,:]), -1) # 2, Npoint, 2
        forcing = np.ones_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

def compute_winding_gpts(points, normals, area, q):
    points = points.view(points.size(0), points.size(2), points.size(3)).permute(0, 2, 1)
    normals = normals.view(normals.size(0), normals.size(2), normals.size(3)).permute(0, 2, 1)
    area = area.view(area.size(0), area.size(2), area.size(3)).permute(0, 2, 1)
    points = points.unsqueeze(-2).unsqueeze(0)
    normals = normals.unsqueeze(-2).unsqueeze(0)
    area = area.unsqueeze(-2).unsqueeze(0)
    q = q.unsqueeze(-1)
    # print(points.shape, normals.shape, area.shape, q.shape)
    # print(torch.sum((area*(points - q[:, :, 0, 0, :])*normals),2).shape)

    winding_number = torch.sum((torch.stack([torch.stack([torch.sum(area*(points - q[:, :, q_idx, q_idy, :])*normals,2)
        for q_idx in range(q.size(2))], 0) for q_idy in range(q.size(3))],
        0)) / (torch.stack([torch.stack([torch.sum(torch.sqrt((points - q[:, :, q_idx, q_idy, :])**2),2)
        for q_idx in range(q.size(2))], 0) for q_idy in range(q.size(3))], 0))**2, -1)
    # winding_number_numerator = torch.stack([torch.stack([torch.sum(area*(points - q[:, :, q_idx, q_idy, :])*normals,2)
    #     for q_idx in range(q.size(2))], 0) for q_idy in range(q.size(3))], 0)
    # winding_number_denominator = (torch.stack([torch.stack([torch.sum(torch.sqrt((points - q[:, :, q_idx, q_idy, :])**2),2)
    #     for q_idx in range(q.size(2))], 0) for q_idy in range(q.size(3))], 0))**2

    winding_number = winding_number.permute(3,2,1,0,4)
    return winding_number


def compute_winding_nodes(points, normals, area, q):
    points = points.view(points.size(0), points.size(2), points.size(3)).permute(0, 2, 1)
    normals = normals.view(normals.size(0), normals.size(2), normals.size(3)).permute(0, 2, 1)
    area = area.view(area.size(0), area.size(2), area.size(3)).permute(0, 2, 1)
    points = points.unsqueeze(-2).unsqueeze(0)
    normals = normals.unsqueeze(-2).unsqueeze(0)
    area = area.unsqueeze(-2).unsqueeze(0)
    q = q.unsqueeze(-1)
    # print(points.shape, normals.shape, area.shape, q.shape)
    # print(torch.sum((area*(points - q[:, :, 0, :])*normals),2).shape)

    winding_number = torch.sum((torch.stack([torch.sum((points - q[:, :, q_idx, :])*normals,2)
        for q_idx in range(q.size(2))],
        0)) / (4*math.pi*torch.stack([torch.sum(torch.sqrt((points - q[:, :, q_idx, :])**2),2)
        for q_idx in range(q.size(2))], 0))**3, -1)
    # winding_number_numerator = torch.stack([torch.stack([torch.sum(area*(points - q[:, :, q_idx, q_idy, :])*normals,2)
    #     for q_idx in range(q.size(2))], 0) for q_idy in range(q.size(3))], 0)
    # winding_number_denominator = (torch.stack([torch.stack([torch.sum(torch.sqrt((points - q[:, :, q_idx, q_idy, :])**2),2)
    #     for q_idx in range(q.size(2))], 0) for q_idy in range(q.size(3))], 0))**2

    winding_number = winding_number.permute(2,1,0,3)
    return winding_number


class Poisson(DiffNet2DFEM):
    """docstring for Poisson"""
    def __init__(self, network, dataset, **kwargs):
        super(Poisson, self).__init__(network, dataset, **kwargs)

    def loss(self, u, inputs_tensor, forcing_tensor):

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        pc = inputs_tensor[:,:,:,0:2] # b, 1, npoints, 2
        normals = inputs_tensor[:,:,:,2:4] # b, 1, npoints, 2
        area = inputs_tensor[:,:,:,4:5] # b, 1, npoints, 1

        # init bin widths
        hx = self.h
        hy = self.h

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

        # nu_gp = self.gauss_pt_evaluation(nu)
        f_gp = self.gauss_pt_evaluation(f)
        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)



        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(f)
        res_elmwise = transformation_jacobian * (1.0 * (u_x_gp**2 + u_y_gp**2) - (u_gp * f_gp))
        res_elmwise = torch.sum(res_elmwise, 1)

        # gpts = torch.stack((self.xgp, self.ygp), 1).type_as(pc)
        nodes = torch.stack((self.xx, self.yy),0).type_as(pc)

        # winding_number = compute_winding_torch(pc, normals, area, gpts)
        winding_number = compute_winding_nodes(pc, normals, area, nodes)

        u_cut_elms = torch.where(winding_number > 0.1, u, u*0.0)

        loss = torch.mean(res_elmwise) + torch.sum(area*(u_pts-1.0)**2) + torch.sum(u_cut_elms**2)
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        return self.network[0], inputs_tensor, forcing_tensor

    def training_step(self, batch, batch_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
        return {"loss": loss_val}

    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs["loss"]
        self.log('PDE_loss', loss.item())
        self.log('loss', loss.item())
        return training_step_outputs

    def configure_optimizers(self):
        lr = self.learning_rate
        opts = [torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        # opts = [torch.optim.Adam(self.network, lr=lr)]
        return opts, []

    def on_epoch_end(self):
        self.network.eval()
        inputs, forcing = self.dataset[0]
        u, winding_number = self.do_query(inputs, forcing)
        self.plot_contours(u.squeeze().detach().cpu(), winding_number.squeeze().detach().cpu())

    def do_query(self, inputs, forcing):
        u, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

        f = forcing_tensor # renaming variable
        # extract boundary conditions here
        pc = inputs_tensor[:,:,:,0:2] # b, 1, npoints, 2
        normals = inputs_tensor[:,:,:,2:4] # b, 1, npoints, 2
        area = inputs_tensor[:,:,:,4:5] # b, 1, npoints, 1
        nodes = torch.stack((self.xx, self.yy),0).type_as(u)

        # winding_number = compute_winding_torch(pc, normals, area, gpts)
        winding_number = compute_winding_nodes(pc, normals, area, nodes)

        # extract diffusivity and boundary conditions here
        # nu = inputs_tensor[:,0:1,:,:]
        # bc1 = inputs_tensor[:,1:2,:,:]
        # bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        # u = torch.where(bc1>0.5,1.0+u*0.0,u)
        # u = torch.where(bc2>0.5,u*0.0,u)

        # nu = nu.squeeze().detach().cpu()
        # u = u.squeeze().detach().cpu()

        return u, winding_number

    def plot_contours(self, u, winding_number):
        fig, axs = plt.subplots(1, 2, figsize=(2*2,1.2),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])


        im0 = axs[0].imshow(u,cmap='jet')
        fig.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(winding_number,cmap='jet', vmin=0.0, vmax=1.0)
        fig.colorbar(im1, ax=axs[1])  
        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    filename = 'bunny-18.png'
    dataset = PCVox(filename, domain_size=128)
    u_tensor = np.ones((1,1,128,128))
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    basecase = Poisson(network, dataset, batch_size=2, domain_size=128)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="pc_complex_immersed_background")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=5, deterministic=True, profiler='simple')

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