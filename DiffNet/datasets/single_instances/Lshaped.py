import os
import math
import copy
import torch
import numpy as np
from torch.utils import data

class LShaped(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """
        self.domain = np.zeros((domain_size, domain_size))
        params = [5,5,50,20,50,20]
        self.domain[params[0]:params[0]+params[2],params[1]:params[1]+params[3]] = 1.0
        self.domain[params[0]:params[0]+params[5],params[1]:params[1]+params[4]] = 1.0
        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        self.bc2[params[0]:params[0]+params[2],params[1]] = 1
        self.bc2[params[0]+params[2],params[1]:params[1]+params[3]] = 1
        self.bc2[params[0]+params[5]:params[0]+params[2],params[1]+params[3]] = 1
        self.bc2[params[0]+params[5],params[1]+params[3]:params[1]+params[4]] = 1
        self.bc2[params[0]:params[0]+params[5],params[1]+params[4]] = 1
        self.bc2[params[0],params[1]:params[1]+params[4]] = 1
        self.n_samples = 200
        self.forcing = copy.deepcopy(self.domain)*10
        

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        return torch.FloatTensor(inputs), torch.FloatTensor(self.forcing).unsqueeze(0)



