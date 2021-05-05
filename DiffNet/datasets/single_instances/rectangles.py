import os
import torch
import numpy as np
from torch.utils import data

class Dataset(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """
        self.domain = np.zeros(domain_size,domain_size)
        rect_params = (10, 10, 50, 30)
        self.domain[rect_params[0]:rect_params[0]+rect_params[2],rect_params[1]:rect_params[1]+rect_params[3]] = 1.0
        self.n_samples = 1
        

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        coeff = self.coeff

        return coeff
