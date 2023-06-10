import os
import math
import torch
import numpy as np
import PIL
from torch.utils import data


class ImageIMBack(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, dirname, domain_size=64):
        """
        Initialization
        """
        filenames = sorted(os.listdir(dirname))
        self.dataset = []
        for fname in filenames:
            filename = os.path.join(dirname, fname)
            file, ext = os.path.splitext(filename)
            if ext in ['.png', '.jpg', '.bmp', '.tiff']:
                img = PIL.Image.open(filename).convert('L')
                # img = PIL.Image.open(filename).convert('L').resize((700, 300))
                img = (np.asarray(img)>0).astype('float')
            else:
                raise ValueError('invalid extension; extension not supported')
            domain = (1-img)

            # bc1 will be source, u will be set to 1 at these locations
            bc1 = np.zeros_like(domain)
            bc1[(1-domain).astype('bool')] = 1
            # bc2 will be sink, u will be set to 0 at these locations
            bc2 = np.zeros_like(domain)
            bc2[:,0] = 1
            bc2[:,-1] = 1
            bc2[0,:] = 1
            bc2[-1,:] = 1
            self.dataset.append(np.array([domain,bc1,bc2]))
        self.dataset = np.array(self.dataset)
        self.n_samples = self.dataset.shape[0]

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = self.dataset[index]
        forcing = np.zeros_like(self.dataset[index][0])
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

class ImageIMBackObject(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, dirname, domain_size=64):
        """
        Initialization
        """
        filenames = sorted(os.listdir(dirname))
        self.dataset = []
        for fname in filenames:
            filename = os.path.join(dirname, fname)
            file, ext = os.path.splitext(filename)
            if ext in ['.png', '.jpg', '.bmp', '.tiff']:
                img = PIL.Image.open(filename).convert('L')
                # img = PIL.Image.open(filename).convert('L').resize((700, 300))
                img = (np.asarray(img)>0).astype('float')
            else:
                raise ValueError('invalid extension; extension not supported')
            domain = (1-img)

            # bc1 will be source, u will be set to 1 at these locations
            bc1 = np.zeros_like(domain)
            bc1[(1-domain).astype('bool')] = 1
            # bc2 will be sink, u will be set to 0 at these locations
            bc2 = np.zeros_like(domain)
            bc2[:,0] = 1
            bc2[:,-1] = 1
            bc2[0,:] = 1
            bc2[-1,:] = 1
            self.dataset.append(np.array([domain,bc1,bc2]))
        self.dataset = np.array(self.dataset)
        self.n_samples = self.dataset.shape[0]

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = self.dataset[index]
        forcing = np.ones_like(self.dataset[index][0])
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

class ImageIMBackNeumann(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, dirname, domain_size=64):
        """
        Initialization
        """
        filenames = sorted(os.listdir(dirname))
        self.dataset = []
        for fname in filenames:
            filename = os.path.join(dirname, fname)
            file, ext = os.path.splitext(filename)
            if ext in ['.png', '.jpg', '.bmp', '.tiff']:
                img = PIL.Image.open(filename).convert('L')
                # img = PIL.Image.open(filename).convert('L').resize((700, 300))
                img = (np.asarray(img)>0).astype('float')
            else:
                raise ValueError('invalid extension; extension not supported')
            domain = (1-img)

            # bc1 will be source, u will be set to 1 at these locations
            bc1 = np.zeros_like(domain)
            bc1[(1-domain).astype('bool')] = 1
            # bc2 will be sink, u will be set to 0 at these locations
            bc2 = np.zeros_like(domain)
            bc2[:,0] = 1
            bc2[0,:] = 1
            bc3 = np.zeros_like(domain)
            bc3[-1,:] = 1
            bc3[:,-1] = 1
            self.dataset.append(np.array([domain,bc1,bc2,bc3]))
        self.dataset = np.array(self.dataset)
        self.n_samples = self.dataset.shape[0]

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = self.dataset[index]
        forcing = np.zeros_like(self.dataset[index][0])
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)