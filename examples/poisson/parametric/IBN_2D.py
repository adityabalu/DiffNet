import os
import sys
import json
import torch
import argparse
import math, random
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
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
seed_everything(42)
from torch.utils import data
from torch.utils.data import DataLoader

import DiffNet
from DiffNet.networks.wgan_old import GoodGenerator
from DiffNet.networks.unets import UNet
from DiffNet.networks.autoencoders import AE
from DiffNet.DiffNetFEM import DiffNet2DFEM
from DiffNet.datasets.parametric.klsum import KLSumStochastic
from DiffNet.datasets.single_instances.klsum import Dataset



class PointClouds(data.Dataset):
    def __init__(self, data_path, type='train', domain_size=32):
        """
        Initialization
        """
        # load point cloud data
        points = np.load(data_path + 'point_cloud.npz')['arr_0']
        normals = np.load(data_path + 'normals.npz')['arr_0']
        if type == 'val':
            points = points[:1250]
            normals = normals[:1250]
        elif type == 'train':
            points = points[1250:]
            normals = normals[1250:]
        # need to shift the point cloud into the center of the domain *** handcrafted for 32^2 domain
        points[:,:,0] = ((points[:,:,0]/np.max(points[:,:,0]).any()))
        points[:,:,1] = ((points[:,:,1]/np.max(points[:,:,1]).any()))
        points = points * 0.5
        points[:,:,0] += 0.25
        points[:,:,1] += 0.5

        # normals = np.load(data_path + 'normals.npz')['arr_0']
        # other inits
        self.domain = np.ones((domain_size,domain_size))
        self.domain_size = domain_size
        self.n_samples = points.shape[0]
        # bc1 will be source, u will be set to 1 at these locations
        self.normals = normals[:,:,:2]
        self.pc = points
        self.area = np.zeros((self.pc.shape[0], self.pc.shape[1], 1))
        self.area[:, 1:-1, 0] = np.sum((self.pc[:,1:-1,:] - self.pc[:,0:-2,:])**2, -1)*0.5 + np.sum((self.pc[:,2:,:] - self.pc[:,1:-1,:])**2, -1)*0.5 
        self.area[:, 0, 0] = np.sum((self.pc[:,1,:] - self.pc[:,0,:])**2, -1)*0.5 + np.sum((self.pc[:,-1,:] - self.pc[:,0,:])**2, -1)*0.5 
        self.area[:,-1, 0] = np.sum((self.pc[:,-1,:] - self.pc[:,-2,:])**2, -1)*0.5 + np.sum((self.pc[:,-1,:] - self.pc[:,0,:])**2, -1)*0.5 
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros_like(self.domain)
        self.bc2[-1,:] = 1
        self.bc2[0,:] = 1
        self.bc2[:,0] = 1
        self.bc2[:,-1] = 1

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.concatenate((self.pc[index,:,:], self.normals[index,:,:], self.area[index,:,:]), -1) # 2, Npoint, 2
        sink = self.bc2
        forcing = np.zeros_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0), torch.FloatTensor(sink).unsqueeze(0)



# Winding number computation for in-outs
def compute_winding_nodes(points, normals, area, q):
    points = points.view(points.size(0), points.size(2), points.size(3)).permute(0, 2, 1)
    normals = normals.view(normals.size(0), normals.size(2), normals.size(3)).permute(0, 2, 1)
    area = area.view(area.size(0), area.size(2), area.size(3)).permute(0, 2, 1)
    points = points.unsqueeze(-2).unsqueeze(0)
    normals = normals.unsqueeze(-2).unsqueeze(0)
    area = area.unsqueeze(-2).unsqueeze(0)
    q = q.unsqueeze(-1)

    winding_number = torch.sum((torch.stack([torch.sum((points - q[:, :, q_idx, :])*normals,2)
        for q_idx in range(q.size(2))],
        0)) / (4*math.pi*torch.stack([torch.sum(torch.sqrt((points - q[:, :, q_idx, :])**2),2)
        for q_idx in range(q.size(2))], 0))**3, -1)

    winding_number = winding_number.permute(2,1,0,3)
    return winding_number






class Poisson(DiffNet2DFEM):
    """docstring for Poisson"""
    def __init__(self, network, **kwargs):
        super(Poisson, self).__init__(network, **kwargs)

    def loss(self, u, source_tensor, f, sink_tensor):
        # apply boundary conditions
        # apply immersed boundary conditions - source
        u = torch.where(source_tensor > 0.5, 1.+u*0., u)
        # apply exterior boundary conditions - sink
        u = torch.where(sink_tensor > 0.5, u*0., u)

        nu_gp = self.gauss_pt_evaluation(torch.ones_like(u))
        f_gp = self.gauss_pt_evaluation(f)
        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(u)
        res_elmwise = transformation_jacobian * (nu_gp * (u_x_gp**2 + u_y_gp**2) - (u_gp * f_gp))
        res_elmwise = torch.sum(res_elmwise, 1) 

        loss = torch.mean(res_elmwise)
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor, sink_tensor = batch
        pc = inputs_tensor[:,:,0:2].unsqueeze(1) # b, 1, npoints, 2
        normals = inputs_tensor[:,:,2:4].unsqueeze(1) # b, 1, npoints, 2
        area = inputs_tensor[:,:,4:5].unsqueeze(1) # b, 1, npoints, 1
        nodes = torch.stack((self.xx, self.yy),0).type_as(pc)
        source_tensor = compute_winding_nodes(pc, normals, area, nodes)
        ones = torch.ones_like(source_tensor)
        zeros = torch.zeros_like(source_tensor)
        source_tensor = torch.where(source_tensor > 0.005, ones, zeros)

        u = self.network(source_tensor)
        return u, source_tensor, forcing_tensor, sink_tensor

    def training_step(self, batch, batch_idx):
        u, source_tensor, forcing_tensor, sink_tensor = self.forward(batch)
        loss = self.loss(u, source_tensor, forcing_tensor, sink_tensor).mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        u, source_tensor, forcing_tensor, sink_tensor = self.forward(batch)
        loss = self.loss(u, source_tensor, forcing_tensor, sink_tensor).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        lr = self.learning_rate
        # opts = [torch.optim.LBFGS(self.network.parameters(), lr=lr, max_iter=5)]
        opts = [torch.optim.Adam(self.network.parameters(), lr=lr)]
        # schd = []
        schd = [torch.optim.lr_scheduler.MultiStepLR(opts[0], milestones=[10,15,30], gamma=0.1)]
        return opts, schd




def main(args):
    domain_size = 32
    LR = 3e-4
    batch_size = args.batch_size
    max_epochs = 1
    print("Max_epochs = ", max_epochs)
    
    data_path = 'Insert correct data path'
    train_dataset = PointClouds(data_path, 'train', domain_size)
    val_dataset = PointClouds(data_path, 'val', domain_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    network = AE(in_channels=1, out_channels=1, n_downsample=2)
    basecase = Poisson(network, batch_size=batch_size, domain_size=domain_size, learning_rate=LR)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    
    # Insert desired logger (optional)
    # wandb_logger = WandbLogger(project='IBN', 
    #                            log_model='all')
    
    checkpoint = ModelCheckpoint(monitor='train_loss',
                                 mode='min', 
                                 save_last=True)

    trainer = pl.Trainer(devices=1,
                         callbacks=[checkpoint],
                         # logger=wandb_logger, # uncomment or replace with desired logger
                         max_epochs=max_epochs,
                         fast_dev_run=args.debug)

    # ------------------------
    # 4 Training
    # ------------------------

    trainer.fit(basecase, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D IBN example problem - Poisson Eqn')
    parser.add_argument('-b','--batch_size', default=512, type=int,
                        help='Batch size')
    parser.add_argument('--debug', default=False, type=bool,
                        help='fast_dev_run argument')
    hparams = parser.parse_args()
    main(hparams)