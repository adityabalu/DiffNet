import os
import math
import torch
import numpy as np
from torch.utils import data
from ...cuboid_mesh import CuboidMesh

class Cuboid(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """
        self.domain = np.ones((domain_size, domain_size, domain_size))
        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size, domain_size))
        self.bc1[0,:,:] = 1
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size, domain_size))
        self.bc2[-1,:,:] = 1
        self.n_samples = 100
        

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = np.zeros_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)



class CuboidManufactured(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """
        self.domain = np.ones((domain_size, domain_size, domain_size))
        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size, domain_size))
        # self.bc1[0,:,:] = 1
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size, domain_size))
        self.bc2[ 0,:,:] = 1
        self.bc2[-1,:,:] = 1
        self.bc2[:, 0,:] = 1
        self.bc2[:,-1,:] = 1
        self.bc2[:,:, 0] = 1
        self.bc2[:,:,-1] = 1
        self.n_samples = 100
        x = np.linspace(0,1,domain_size)
        y = np.linspace(0,1,domain_size)
        z = np.linspace(0,1,domain_size)
        # xx, yy, zz = np.meshgrid(x,y,z)
        self.xx, self.yy, self.zz = CuboidMesh.meshgrid_3d(x,y,z)
        # self.forcing = 3. * math.pi**2 * np.sin(math.pi * self.xx) * np.sin(math.pi * self.yy) * np.sin(math.pi * self.zz)
        self.forcing = self.forcing(self.xx, self.yy, self.zz)

    def forcing(self,x,y,z):
        pi = math.pi
        sin = np.sin
        cos = np.cos
        # f = 3. * pi**2 * sin(pi*x) * sin(pi*y) * sin(pi*z)
        f = 19. * pi**2 * sin(pi*x)*sin(3.*pi*y)*sin(3*pi*z)
        return f

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = self.forcing
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

