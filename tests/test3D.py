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
    'font.size':8,
})
from matplotlib import pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
seed_everything(42)

import DiffNet
from DiffNet.DiffNetFEM import DiffNet3DFEM
from torch.utils import data



class Test3D(DiffNet3DFEM):
    """docstring for Test"""
    def __init__(self, **kwargs):
        super(Test3D, self).__init__(None, None, **kwargs)


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

    def calc_residuals(self, output, input):
        N_values = self.Nvalues.type_as(input)
        dN_x_values = self.dN_x_values.type_as(input)
        dN_y_values = self.dN_y_values.type_as(input)
        dN_z_values = self.dN_z_values.type_as(input)
        gpw = self.gpw.type_as(input)

        input_pad = torch.nn.functional.pad(input, (1, 1, 1, 1, 1, 1), 'replicate')
        output_pad = torch.nn.functional.pad(output, (0,0,1,1,1,1), 'replicate')
        output_pad = torch.nn.functional.pad(output_pad, (1,0,0,0,0,0), 'constant', value=1)
        output_pad = torch.nn.functional.pad(output_pad, (0,1,0,0,0,0), 'constant', value=0)
        
        u_gp = (self.gauss_pt_evaluation(output_pad)).unsqueeze(1)
        nu_gp = (self.gauss_pt_evaluation(input_pad)).unsqueeze(1)
        u_x_gp = (self.gauss_pt_evaluation_der_x(output_pad)).unsqueeze(1)
        u_y_gp = (self.gauss_pt_evaluation_der_y(output_pad)).unsqueeze(1)
        u_z_gp = (self.gauss_pt_evaluation_der_z(output_pad)).unsqueeze(1)

        trnsfrm_jac = (0.5*self.h)**3
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        # CALCULATION STARTS
        # lhs
        vxux = dN_x_values*u_x_gp*JxW
        vyuy = dN_y_values*u_y_gp*JxW
        vzuz = dN_z_values*u_z_gp*JxW

        # integrated values on lhs & rhs
        v_lhs = torch.sum(nu_gp*(vxux+vyuy+vzuz), 2) # sum across all GP

        # unassembled residual
        R_split = v_lhs
        # assembly
        R = torch.zeros_like(input_pad); R = self.Q1_3D_vector_assembly(R, R_split)

        loss = torch.sum(R**2, (-1, -2, -3, -4))

        loss = torch.mean(loss)
        return loss


def main():
    bs = 1
    domain_size = 64
    x = torch.linspace(0.0, 1.0, domain_size)
    y = torch.linspace(0.0, 1.0, domain_size)
    z = torch.linspace(0.0, 1.0, domain_size)

    xx, yy, zz = torch.meshgrid(x,y,z)
    xx = xx.permute(2,1,0)
    yy = yy.permute(2,1,0)
    zz = zz.permute(2,1,0)

    xx = xx.unsqueeze(0).unsqueeze(1)
    yy = yy.unsqueeze(0).unsqueeze(1)
    zz = zz.unsqueeze(0).unsqueeze(1)

    seed = (1-xx)**3
    u = torch.repeat_interleave(seed, bs, 0)
    k = torch.ones((bs, 1, domain_size, domain_size, domain_size))

    testcase = Test3D(domain_size=domain_size,nsd=3)
    loss = testcase.calc_residuals(u,k)

    print(loss)


if __name__ == '__main__':
    main()