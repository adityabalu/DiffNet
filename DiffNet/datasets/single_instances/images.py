import os
import math
import torch
import numpy as np
import PIL
from torch.utils import data


class ImageIMBack(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, filename, domain_size=64):
        """
        Initialization
        """
        file, ext = os.path.splitext(filename)
        if ext in ['.png', '.jpg', '.bmp', '.tiff']:
            img = PIL.Image.open(filename).convert('L')
            # img = PIL.Image.open(filename).convert('L').resize((700, 300))
            img = (np.asarray(img)>0).astype('float')
        else:
            raise ValueError('invalid extension; extension not supported')
        self.domain = (1-img)

        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros_like(self.domain)
        self.bc1[(1-self.domain).astype('bool')] = 1
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros_like(self.domain)
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

class Disk(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, filename, domain_size=64):
        """
        Initialization
        """
        file, ext = os.path.splitext(filename)
        if ext in ['.png', '.jpg', '.bmp', '.tiff']:
            img = PIL.Image.open(filename).convert('L')
            # img = PIL.Image.open(filename).convert('L').resize((700, 300))
            img = (np.asarray(img)>0).astype('float')
        else:
            raise ValueError('invalid extension; extension not supported')
        self.domain = (1-img)

        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros_like(self.domain)
        self.bc1[(1-self.domain).astype('bool')] = 1
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros_like(self.domain)
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
        forcing = np.ones_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)
