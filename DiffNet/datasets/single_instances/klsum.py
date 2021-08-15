import os
import torch
import numpy as np
from torch.utils import data
from DiffNet.gen_input_calc import generate_diffusivity_tensor

class Dataset(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, coeff_file, domain_size=64):
        """
        Initialization
        """
        if os.path.exists(coeff_file):
            self.coeff = np.loadtxt(coeff_file, dtype=np.float32)
        else:
            raise FileNotFoundError("Single instance: Wrong path to coefficient file.")
        self.domain_size = domain_size
        self.n_samples = 1000
        self.nu = generate_diffusivity_tensor(self.coeff, output_size=self.domain_size).squeeze()
        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        self.bc1[:,0] = 1
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        self.bc2[:,-1] = 1
        self.n_samples = 100        

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.nu, self.bc1, self.bc2])
        forcing = np.zeros_like(self.nu)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)        
