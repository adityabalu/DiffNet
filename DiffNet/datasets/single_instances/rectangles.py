import os
import math
import torch
import numpy as np
from torch.utils import data

class Rectangle(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """
        self.domain = np.ones((domain_size, domain_size))
        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        self.bc1[0,:] = 1
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        self.bc2[-1,:] = 1
        self.n_samples = 6000
        

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = np.zeros_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)



class RectangleManufactured(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """
        self.domain = np.ones((domain_size, domain_size))
        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        self.bc2[-1,:] = 1
        self.bc2[0,:] = 1
        self.bc2[:,0] = 1
        self.bc2[:,-1] = 1
        self.n_samples = 100
        x = np.linspace(0,1,domain_size)
        y = np.linspace(0,1,domain_size)
        xx, yy = np.meshgrid(x,y)
        self.forcing = 2. * math.pi**2 * np.sin(math.pi * xx) * np.sin(math.pi * yy)
        

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = self.forcing
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

class RectangleManufacturedNonZeroBC(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """
        self.domain = np.ones((domain_size, domain_size))
        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        self.bc1[:,0] = 1
        self.bc1[:,-1] = 1
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        self.bc2[-1,:] = 1
        self.bc2[0,:] = 1
        self.n_samples = 100
        x = np.linspace(0,1,domain_size)
        y = np.linspace(0,1,domain_size)
        xx, yy = np.meshgrid(x,y)
        self.xx = xx
        self.yy = yy
        self.om = np.pi
        self.u_exact = np.exp(-self.om*xx)*np.sin(self.om*yy)
        self.forcing = np.zeros_like(xx)


    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = self.forcing
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

class RectangleHelmholtzManufactured(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """
        self.khh = 0.5
        self.domain = np.ones((domain_size, domain_size))
        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        self.bc2[-1,:] = 1
        self.bc2[0,:] = 1
        self.bc2[:,0] = 1
        self.bc2[:,-1] = 1
        self.n_samples = 100
        x = np.linspace(0,1,domain_size)
        y = np.linspace(0,1,domain_size)
        xx, yy = np.meshgrid(x,y)
        self.forcing = (2. * math.pi**2 - self.khh**2) * np.sin(math.pi * xx) * np.sin(math.pi * yy)


    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = self.forcing
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

class RectangleHelmholtzDeltaForce(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """
        self.khh = 1./8.
        self.domain = np.ones((domain_size, domain_size))
        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        self.bc2[-1,:] = 1
        self.bc2[0,:] = 1
        self.bc2[:,0] = 1
        self.bc2[:,-1] = 1
        self.n_samples = 100
        x = np.linspace(0,1,domain_size)
        y = np.linspace(0,1,domain_size)
        xx, yy = np.meshgrid(x,y)

        mu1 = 0.1875
        mu2 = 0.1875
        sigma1 = 0.05
        sigma2 = 0.05
        self.forcing = np.exp(-0.5*((xx-mu1)/sigma1)**2 - 0.5*((yy-mu2)/sigma2)**2) / (2*np.pi*sigma1*sigma2)

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = self.forcing
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

class RectangleManufacturedStokes(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """
        self.domain = np.ones((domain_size, domain_size))
        # bc1 will be source, ux will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        # bc2 will be sink, ux will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        # bc3 will be source, uy will be set to 1 at these locations
        self.bc3 = np.zeros((domain_size, domain_size))
        # bc4 will be sink, uy will be set to 0 at these locations
        self.bc4 = np.zeros((domain_size, domain_size))
        self.bc2[-1,:] = 1
        self.bc2[0,:] = 1
        self.n_samples = 100
        x = np.linspace(0,1,domain_size)
        y = np.linspace(0,1,domain_size)
        xx, yy = np.meshgrid(x,y)
        self.forcing = 2. * math.pi**2 * np.sin(math.pi * xx) * np.sin(math.pi * yy)
        

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = self.forcing
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)




class RectangleIM(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """

        self.domain = np.zeros((domain_size, domain_size))
        rect_params = [10,10,30,50] # x, y, w, h
        self.domain[rect_params[1]:rect_params[1]+rect_params[3],rect_params[0]:rect_params[0]+rect_params[2]] = 1.0

        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        self.bc1[rect_params[1],rect_params[0]:rect_params[0]+rect_params[2]] = 1
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        self.bc2[rect_params[1]+rect_params[3],rect_params[0]:rect_params[0]+rect_params[2]] = 1
        self.n_samples = 200        

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = np.zeros_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

class RectangleIMBack(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """
        
        self.domain = np.ones((domain_size, domain_size))
        rect_params = [10,10,30,20] # x, y, w, h
        self.domain[rect_params[1]:rect_params[1]+rect_params[3],rect_params[0]:rect_params[0]+rect_params[2]] = 0.0

        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        self.bc1[rect_params[1]:rect_params[1]+rect_params[3],rect_params[0]:rect_params[0]+rect_params[2]] = 1.0
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        self.bc2[0,:] = 1
        self.bc2[-1,:] = 1
        self.bc2[:,0] = 1
        self.bc2[:,-1] = 1
        self.n_samples = 200

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = np.zeros_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)
