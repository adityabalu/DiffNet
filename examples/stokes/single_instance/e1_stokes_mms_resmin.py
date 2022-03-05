import os
import sys
import math
import json
import torch
import numpy as np

import scipy.io
from scipy import ndimage
import matplotlib
# from skimage import io
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
from DiffNet.DiffNetFEM import DiffNet2DFEM
from torch.utils import data
from e1_stokes_resmin_base import Stokes2D

class Stokes_MMS_Dataset(data.Dataset):
    'PyTorch dataset for Stokes_MMS_Dataset'
    def __init__(self, domain_size=64, Re=1):
        """
        Initialization
        """

        x = np.linspace(0, 1, domain_size)
        y = np.linspace(0, 1, domain_size)

        xx , yy = np.meshgrid(x, y)
        self.x = xx
        self.y = yy
        # bc1 for fixed boundaries
        self.bc1 = np.zeros_like(xx)
        self.bc1[ 0, :] = 1.0
        self.bc1[-1, :] = 1.0
        self.bc1[ :, 0] = 1.0
        self.bc1[ :,-1] = 1.0

        self.bc2 = np.zeros_like(xx)
        self.bc2[ 0, :] = 1.0
        self.bc2[-1, :] = 1.0
        self.bc2[ :, 0] = 1.0
        self.bc2[ :,-1] = 1.0

        self.bc3 = np.zeros_like(xx)
        self.bc3[0:1,0:1] = 1.0

        self.Re = Re
        self.n_samples = 100

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.x, self.y, self.bc1, self.bc2, self.bc3])

        forcing = np.ones_like(self.x)*(1/self.Re)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

class Stokes_MMS(Stokes2D):
    """docstring for Stokes_MMS"""
    def __init__(self, network, dataset, **kwargs):
        super(Stokes_MMS, self).__init__(network, dataset, **kwargs)

        self.u_bc = self.u_exact
        self.v_bc = self.v_exact
        self.p_bc = self.p_exact

    def exact_solution(self, x, y):
        print("exact_solution -- child class called")
        pi = math.pi
        sin = np.sin
        cos = np.cos        
        u_exact =  sin(pi*x)*cos(pi*y)
        v_exact = -cos(pi*x)*sin(pi*y)
        p_exact =  sin(pi*x)*sin(pi*y)
        return u_exact, v_exact, p_exact

    def forcing(self, x, y):
        print("forcing -- child class called")
        pi = math.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp        
        fx =  2*pi**2*sin(pi*x)*cos(pi*y) + pi*sin(pi*y)*cos(pi*x)
        fy = -2*pi**2*sin(pi*y)*cos(pi*x) + pi*sin(pi*x)*cos(pi*y)
        return fx, fy

def main():
    domain_size = 16
    dir_string = "stokes_mms"
    max_epochs = 1000

    x = np.linspace(0, 1, domain_size)
    y = np.linspace(0, 1, domain_size)
    xx , yy = np.meshgrid(x, y)

    dataset = Stokes_MMS_Dataset(domain_size=domain_size)
    v1 = np.zeros_like(dataset.x) # np.sin(math.pi*xx)*np.cos(math.pi*yy)
    v2 = np.zeros_like(dataset.x) # -np.cos(math.pi*xx)*np.sin(math.pi*yy)
    p  = np.zeros_like(dataset.x) # np.sin(math.pi*xx)*np.sin(math.pi*yy)
    u_tensor = np.expand_dims(np.array([v1,v2,p]),0)
    print(u_tensor.shape)
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    basecase = Stokes_MMS(network, dataset, domain_size=domain_size, batch_size=1, fem_basis_deg=1)

    # Initialize trainer
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

    # Training
    trainer.fit(basecase)
    # Save network
    torch.save(basecase.network, os.path.join(logger.log_dir, 'network.pt'))


if __name__ == '__main__':
    main()