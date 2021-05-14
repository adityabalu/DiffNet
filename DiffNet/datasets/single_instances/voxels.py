import os
import math
import torch
import numpy as np
import PIL
from torch.utils import data

def load_raw(fileName, **kwargs):
    def _configParser(cName):
        with open(cName, 'r') as configFile:
            configFile.readline()
            line1 = configFile.readline().split()
            bBoxMin = np.array([float(i) for i in line1])
            line2 = configFile.readline().split()
            bBoxMax = np.array([float(i) for i in line2])
            line3 = configFile.readline().split()
            numDiv = np.array([int(i) for i in line3])
            line4 = configFile.readline().split()
            gridSize = np.array([float(i) for i in line4])
            inOutVoxelNum = int(configFile.readline())
            boundaryVoxelNum = int(configFile.readline())
        return bBoxMax, bBoxMin, numDiv, gridSize

    inOutName = fileName + 'inouts.raw'
    configName = fileName + 'VoxelConfig.txt'
    inOut = np.fromfile(inOutName, dtype=np.dtype('uint8'))
    inOut = (inOut / 254.0 > 0.5).astype(float)
    bBoxMax, bBoxMin, numDiv, gridSize = _configParser(configName)
    inOut = np.reshape(inOut, numDiv, order='F')
    return inOut, numDiv, gridSize, bBoxMin

class VoxelIMBackRAW(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, filename, domain_size=64):
        """
        Initialization
        """

        vox, _, _ , _  = load_raw(filename)
        self.domain = 1-vox

        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros_like(self.domain)
        self.bc1[(1-self.domain).astype('bool')] = 1
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros_like(self.domain)
        self.bc2[-1,:,:] = 1
        self.bc2[0,:,:] = 1
        self.bc2[:,0,:] = 1
        self.bc2[:,-1,:] = 1
        self.bc2[:,:,0] = 1
        self.bc2[:,:,-1] = 1
        self.n_samples = 100

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = np.zeros_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)