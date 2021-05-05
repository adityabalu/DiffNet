import os
import torch
import numpy as np
from torch.utils import data
from DiffNet.datasets.

class Dataset(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, coeff_file):
        """
        Initialization
        """
        if os.path.exists(coeff_file):
            self.coeff = np.loadtxt(coeff_file, dtype=np.float32)
        else:
            raise FileNotFoundError("Single instance: Wrong path to coefficient file.")
        self.n_samples = 1
        self.nu = 
        

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'

        return coeff
