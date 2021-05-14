import os
import math
import torch
import numpy as np
from torch.utils import data


class CircleIMBack(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """

        self.domain = np.zeros((domain_size, domain_size))
        circ_params = [15,40,15] # x, y, r
        x = np.linspace(0,1,domain_size)*domain_size
        y = np.linspace(0,1,domain_size)*domain_size
        xx, yy = np.meshgrid(x,y)
        zz = (xx - circ_params[0])**2 + (yy - circ_params[1])**2 - circ_params[2]**2
        self.domain[zz>0.0] = 1.0

        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        self.bc1[zz<0.0] = 1
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        self.bc2[:,0] = 1
        self.bc2[:,-1] = 1
        self.bc2[0,:] = 1
        self.bc2[-1,:] = 1
        self.n_samples = 100        

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = np.zeros_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)