import torch
import numpy as np
from torch import nn
from .base import PDE

def get_deriv_kernels(nsd, ktype, num_pt, output_dim):
    # First derivative kernels
    if ktype == 'fs':
        k1 = np.array([[0.104550, 0.292315, 0.0000, -0.292315, -0.104550]], dtype=np.float32)
        k2 = np.array([[0.25, 0.5, 1.0, 0.5, 0.25]], dtype=np.float32).transpose(1,0) 
        ker_t= np.matmul(k2, k1)/output_dim * 5
        ker_x = ker_t.T/5.0
        padding_d1=2
        order = 1
    else:
        if num_pt == 3:
            padding_d1 = 1
            stencil = np.array([-1.0, 0.0, 1.0], dtype=np.float32)*((output_dim-1)/2.0) 
            if ktype == 'fdm':
                weights = np.array([1,1,1],dtype=np.float32)                           
            elif ktype == 'sobel':
                weights = np.array([1,2,1],dtype=np.float32)
                # weights = np.array([[3,10,3]],dtype=np.float32)
        elif num_pt == 5:
            padding_d1 = 2
            stencil = np.array([1.0, -8.0, 0.0, 8.0, -1.0], dtype=np.float32)*((output_dim-1)/12.0)
            if ktype == 'fdm':
                weights = np.array([1,1,1,1,1],dtype=np.float32)                
            elif ktype == 'sobel':
                weights = np.array([1,4,6,4,1],dtype=np.float32)

    # Second derivative stencil and weights
    if num_pt == 3:        
        pading_d2 = 1
        d2_stencil = ((output_dim-1)**2) * np.array([1,-2,1],dtype=np.float32)
        d2_weights = np.array([1,1,1],dtype=np.float32) 
    elif num_pt == 5:
        pading_d2 = 1
        raise NotImplementedError
    
    if nsd == 2:
        ker_x = np.kron(weights, stencil) / np.sum(weights)
        ker_x = ker_x.reshape((num_pt, num_pt))
        ker_y = ker_x.T
        ker_z = np.zeros(ker_x.shape)
        ker_xx = np.kron(d2_weights, d2_stencil) / np.sum(d2_weights)
        ker_xx = ker_xx.reshape((num_pt, num_pt))
        ker_yy = ker_xx.T
        ker_zz = np.zeros(ker_xx.shape)
    elif nsd == 3:
        ker_x = np.kron(weights, np.kron(weights, stencil)) / (np.sum(weights) ** 2)
        ker_x = ker_x.reshape((num_pt, num_pt, num_pt))
        ker_y = ker_x.transpose((0,2,1))
        ker_z = ker_x.transpose((2,1,0))
        ker_xx = np.kron(d2_weights, np.kron(d2_weights, d2_stencil)) / (np.sum(d2_weights)**2)
        ker_xx = ker_xx.reshape((num_pt, num_pt, num_pt))
        ker_yy = ker_xx.transpose((0,2,1))
        ker_zz = ker_xx.transpose((2,1,0))

    return padding_d1, ker_x, ker_y, ker_z, pading_d2, ker_xx, ker_yy, ker_zz


def get_sobel_correction_matrix(nsd, size, padding_xy, padding_xy_d2):
    w = size
    nz = size

    # Correction for First derivative
    corr_mat = np.eye(w, dtype=np.float32)
    if padding_xy == 1:
        corr_mat[0,0] = 4.0
        corr_mat[w-1, w-1] = 4.0
        corr_mat[1,0] = -1.0
        corr_mat[w-2, w-1] = -1.0
    elif padding_xy == 2:
        corr_mat[0,0] = 7.469077911720371
        corr_mat[1,0] = -3.617376998526026
        corr_mat[2,0] = 1.523414436571198
        
        corr_mat[0,1] = -1.715859601067273
        corr_mat[1,1] = 2.053315601134080
        corr_mat[2,1] = -0.484817674298193
        
        corr_mat[w-1, w-1] = 7.469077911720371
        corr_mat[w-2, w-1] = -3.617376998526026
        corr_mat[w-3, w-1] = 1.523414436571198
        
        corr_mat[w-1, w-2] = -1.715859601067273
        corr_mat[w-2, w-2] = 2.053315601134080
        corr_mat[w-3, w-2] = -0.484817674298193 

    # Correction for Second derivative
    corr_mat_d2 = np.eye(w, dtype=np.float32)
    if padding_xy_d2 == 1:
        corr_mat_d2[0,0] = 0.0
        corr_mat_d2[w-1, w-1] = 0.0
        corr_mat_d2[1,0] = 1.0
        corr_mat_d2[w-2, w-1] = 1.0
    elif padding_xy_d2 == 2:
        raise NotImplementedError 

    if nsd == 2:
        corr_matX = corr_mat
        corr_matY = corr_matX.T

        corr_matX_d2 = corr_mat_d2
        corr_matY_d2 = corr_matX_d2.T
    elif nsd == 3:
        corr_mat = corr_mat[np.newaxis,:,:]
        corr_mat = np.repeat(corr_mat, nz, axis=0)        
        corr_matX = corr_mat
        corr_matY = np.transpose(corr_mat, (0,2,1))

        corr_mat_d2 = corr_mat_d2[np.newaxis,:,:]
        corr_mat_d2 = np.repeat(corr_mat_d2, nz, axis=0)        
        corr_matX_d2 = corr_mat_d2
        corr_matY_d2 = np.transpose(corr_mat_d2, (0,2,1))
    print("corr_matX = ", corr_matX)
    print("corr_matY = ", corr_matY)
    return corr_matX, corr_matY, corr_matX_d2, corr_matY_d2



class DiffNetFDM(PDE):
    """docstring for DiffNetFDM"""
    def __init__(self, network, dataset, **kwargs):
        super(DiffNetFDM, self).__init__(network, dataset, **kwargs)

        padding_d1, ker_x, ker_y, ker_z, padding_d2, ker_xx, ker_yy, ker_zz = get_deriv_kernels(self.args.nsd, self.args.ktype, self.args.stencil_len, self.args.output_size)
        corr_matX, corr_matY, corr_matX_d2, corr_matY_d2 = get_sobel_correction_matrix(self.args.nsd, self.args.output_size, padding_d1, padding_d2)

        self.stencil_len = self.args.stencil_len
        self.d2_calc_type = self.args.d2_calc_type

        self.sobelx = nn.Parameter(torch.tensor(ker_x).unsqueeze(0).unsqueeze(1), requires_grad=False)
        self.sobely = nn.Parameter(torch.tensor(ker_y).unsqueeze(0).unsqueeze(1), requires_grad=False)
        self.sobelz = nn.Parameter(torch.tensor(ker_z).unsqueeze(0).unsqueeze(1), requires_grad=False)

        self.sobelxx = nn.Parameter(torch.tensor(ker_xx).unsqueeze(0).unsqueeze(1), requires_grad=False)
        self.sobelyy = nn.Parameter(torch.tensor(ker_yy).unsqueeze(0).unsqueeze(1), requires_grad=False)
        self.sobelzz = nn.Parameter(torch.tensor(ker_zz).unsqueeze(0).unsqueeze(1), requires_grad=False)

        # self.laplacian = nn.Parameter(torch.tensor(ker_laplacian).unsqueeze(0).unsqueeze(1), requires_grad=False)
        if self.args.nsd == 2:
            self.pad = nn.ReplicationPad2d(padding=padding_d1)
            self.pad_d2 = nn.ReplicationPad2d(padding=padding_d2)
        else:
            self.pad = nn.ReplicationPad3d(padding=padding_d1)
            self.pad_d2 = nn.ReplicationPad3d(padding=padding_d2)
        self.h_corr = nn.Parameter((torch.tensor(corr_matX)).unsqueeze(0).unsqueeze(1), requires_grad=False)
        self.v_corr = nn.Parameter((torch.tensor(corr_matY)).unsqueeze(0).unsqueeze(1), requires_grad=False)
        
        self.h_corr_d2 = nn.Parameter(torch.tensor(corr_matX_d2).unsqueeze(0).unsqueeze(1), requires_grad=False)
        self.v_corr_d2 = nn.Parameter(torch.tensor(corr_matY_d2).unsqueeze(0).unsqueeze(1), requires_grad=False)
        
        self.c1 = self.args.l1_coeff
        self.c2 = self.args.l2_coeff

    def derivative_x(self, g):
        # print("size of g = ", g.shape, ", size of sobelx = ", self.sobelx.shape, ", size of h_corr = ", self.h_corr.shape)
        if self.nsd == 2:
            d = torch.matmul(nn.functional.conv2d(g, self.sobelx), self.h_corr)
        elif self.nsd == 3:
            d = torch.matmul(nn.functional.conv3d(g, self.sobelx), self.h_corr)
        return d

    def derivative_y(self, g):
        if self.nsd == 2:
            d = torch.matmul(self.v_corr, nn.functional.conv2d(g, self.sobely))
        elif self.nsd == 3:
            d = torch.matmul(self.v_corr, nn.functional.conv3d(g, self.sobely))
        return d

    def derivative_z(self, g):
        d = nn.functional.conv3d(g, self.sobelz)
        if self.stencil_len == 3:
            d[:,:,0,:,:]  = 4 * d[:,:,0,:,:]  - d[:,:,1,:,:]
            d[:,:,-1,:,:] = 4 * d[:,:,-1,:,:] - d[:,:,-2,:,:]
        return d

    def derivative_xx(self, g):
        if self.nsd == 2:
            d = torch.matmul(nn.functional.conv2d(g, self.sobelxx), self.h_corr_d2)
        elif self.nsd == 3:
            d = torch.matmul(nn.functional.conv3d(g, self.sobelxx), self.h_corr_d2)
        return d

    def derivative_yy(self, g):
        if self.nsd == 2:
            d = torch.matmul(self.v_corr_d2, nn.functional.conv2d(g, self.sobelyy))
        elif self.nsd == 3:
            d = torch.matmul(self.v_corr_d2, nn.functional.conv3d(g, self.sobelyy))
        return d

    def derivative_zz(self, g):
        d = nn.functional.conv3d(g, self.sobelzz)
        if self.stencil_len == 3:
            d[:,:,0,:,:]  = d[:,:,1,:,:]
            d[:,:,-1,:,:] = d[:,:,-2,:,:]
        return d

    def calc_laplacian(self, g):
        dL = nn.functional.conv2d(g, self.laplacian)
        return dL