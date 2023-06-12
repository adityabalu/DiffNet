import os
import sys
import json
import torch
import math
import argparse
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
from DiffNet.networks.wgan3d import GoodGenerator
from DiffNet.DiffNetFEM import DiffNet3DFEM

import os
from trimesh import Trimesh
from skimage import measure




def visMC(VDtarget, Threshold=0.5, path='MCs'):
    # Padding required to remove artifacts in the isosurfaces..
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
        return vector
    
    # necessary for distributed training 
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except FileExistsError:
        pass

    VDtarget = np.pad(VDtarget,2,pad_with)
    try:
        # verts, faces, normals, _ = measure.marching_cubes_lewiner(VDtarget, Threshold, allow_degenerate=False)
        verts, faces, normals, _ = measure.marching_cubes(VDtarget, 
                                                          Threshold, 
                                                          allow_degenerate=False, 
                                                          method='lewiner')
    except ValueError:
        Threshold = (VDtarget.max()-VDtarget.min())*0.5 + VDtarget.min()
        # verts, faces, normals, _ = measure.marching_cubes_lewiner(VDtarget, Threshold, allow_degenerate=False)
        verts, faces, normals, _ = measure.marching_cubes(VDtarget, 
                                                          Threshold, 
                                                          allow_degenerate=False, 
                                                          method='lewiner')
    
    mesh = Trimesh(vertices=verts,
                    faces=faces,
                    vertex_normals=normals)
    mesh.export(file_obj=os.path.join(path,'target_k.obj'))


################################
#   3D Dataset - Structurally optimized topologies via SIMP with random loading conditions 
################################

class TopoDataset3D(data.Dataset):
    def __init__(self, data_path, domain_size=32, mode='train'):
        self.samples = data_path
        list_IDs = os.listdir(self.samples)
        if mode == 'train':
            self.list_IDs = list_IDs[:100]
        else:
            self.list_IDs = list_IDs[100:125]
        
        self.domain = np.ones((domain_size, domain_size, domain_size))
        self.domain_size = domain_size

        self.bc2 = np.zeros_like(self.domain)
        self.bc2[-1,:,:] = 1
        self.bc2[0,:,:] = 1
        self.bc2[:,0,:] = 1
        self.bc2[:,-1,:] = 1
        self.bc2[:,:,0] = 1
        self.bc2[:,:,-1] = 1

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        sample = np.load(os.path.join(self.samples, str(self.list_IDs[index])))['arr_0']
        source = torch.FloatTensor(sample)
        sink = torch.FloatTensor(np.expand_dims(self.bc2, axis=0))
        forcing = torch.zeros_like(source)
        return source, sink, forcing




class Poisson(DiffNet3DFEM):
    """docstring for Poisson"""
    def __init__(self, network, **kwargs):
        super(Poisson, self).__init__(network, **kwargs)

    def loss(self, u, source_tensor, sink_tensor, forcing_tensor):

        f = forcing_tensor # renaming variable
    
        # apply boundary conditions
        u = torch.where(source_tensor > 0.5, 1.+(u*0.), u) # source
        source_tensor = torch.where(source_tensor > 0.5, 1.+(source_tensor*0.), source_tensor*0.)
        sink_tensor = torch.where(source_tensor == sink_tensor, sink_tensor*0., sink_tensor) # adjust for source on boundaries
        u = torch.where(sink_tensor > 0.5, u*0., u) # sink

        # nu_gp = self.gauss_pt_evaluation(torch.ones_like(u))
        f_gp = self.gauss_pt_evaluation(f)
        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)
        u_z_gp = self.gauss_pt_evaluation_der_z(u)

        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(u)
        res_elmwise = transformation_jacobian * (1. * (u_x_gp**2 + u_y_gp**2 + u_z_gp**2) - (u_gp * f_gp))
        res_elmwise = torch.sum(res_elmwise, 1) 

        loss = torch.mean(res_elmwise)
        return loss

    def forward(self, batch):
        source_tensor, sink_tensor, forcing_tensor = batch

        u = self.network(source_tensor)
        return u, source_tensor, sink_tensor, forcing_tensor

    def training_step(self, batch, batch_idx):
        u, source_tensor, sink_tensor, forcing_tensor = self.forward(batch)
        loss = self.loss(u, source_tensor, sink_tensor, forcing_tensor).mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        u, source_tensor, sink_tensor, forcing_tensor = self.forward(batch)
        loss = self.loss(u, source_tensor, sink_tensor, forcing_tensor).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        lr = self.learning_rate
        # opts = [torch.optim.LBFGS(self.network.parameters(), lr=lr, max_iter=5)]
        opts = [torch.optim.Adam(self.network.parameters(), lr=lr)]
        # schd = []
        schd = [torch.optim.lr_scheduler.MultiStepLR(opts[0], milestones=[10,15,30], gamma=0.1)]
        return opts, schd


def main(args):
    domain_size = 32
    LR = 3e-5
    batch_size = args.batch_size
    max_epochs = 10
    print("Max_epochs = ", max_epochs)

    data_path = 'insert correct data path here'
    train_dataset = TopoDataset3D(data_path, domain_size, mode='train')
    val_dataset = TopoDataset3D(data_path, domain_size, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    network = GoodGenerator(in_channels=1, out_channels=1)
    basecase = Poisson(network, batch_size=batch_size, domain_size=domain_size, learning_rate=LR, nsd=3)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------

    # wandb logger if applicable
    # wandb_logger = WandbLogger(project='IBN_3D', 
    #                            log_model='all')
    
    checkpoint = ModelCheckpoint(monitor='train_loss',
                                 mode='min', 
                                 save_last=True)

    trainer = pl.Trainer(devices=args.gpu, 
                         accelerator='gpu', 
                         strategy='ddp',
                         callbacks=[checkpoint],
                        #  logger=wandb_logger, # uncomment or replace with desired logger
                         max_epochs=max_epochs,
                         fast_dev_run=args.debug)

    # ------------------------
    # 4 Training
    # ------------------------

    trainer.fit(basecase, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CAD IBN Results Compilation')
    parser.add_argument('-g','--gpu', default=2, type=int,
                        help='num gpus')
    parser.add_argument('-b','--batch_size', default=8, type=int,
                        help='Batch size')
    parser.add_argument('--debug', default=False, type=bool,
                        help='fast_dev_run argument')
    hparams = parser.parse_args()
    main(hparams)