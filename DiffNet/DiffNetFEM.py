import torch
import numpy as np
from torch import nn
from .base import PDE

def gauss_pt_eval(tensor, N, nsd=2, stride=1):
    if nsd == 2:
        conv_gp = nn.functional.conv2d
    elif nsd == 3:
        conv_gp = nn.functional.conv3d

    result_list = []
    for i in range(len(N)):
        result_list.append(conv_gp(tensor, N[i], stride=stride))
    return torch.cat(result_list, 1)


class DiffNetFEM(PDE):
    """docstring for DiffNetFEM"""
    def __init__(self, network, dataset, **kwargs):
        super(DiffNetFEM, self).__init__(network, dataset, **kwargs)
        self.ngp_1d = kwargs.get('ngp_1d', 2)
        self.fem_basis_deg = kwargs.get('fem_basis_deg', 1)

        # Gauss quadrature setup
        if self.fem_basis_deg == 1:
            ngp_1d = 2
        elif self.fem_basis_deg == 2:
            ngp_1d = 3
        elif self.fem_basis_deg == 3:
            ngp_1d = 3

        if ngp_1d > self.ngp_1d:
            self.ngp_1d = ngp_1d
        
        self.ngp_total = ngp_total = self.ngp_1d**self.nsd
        self.gpx_1d, self.gpw_1d = self.gauss_guadrature_scheme(self.ngp_1d)

        # Basis functions setup
        if self.fem_basis_deg == 1:
            self.nelem = nelem = self.domain_size - 1
            self.nbf_1d = nbf_1d = 2
            self.nbf_total = nbf_total = self.nbf_1d**self.nsd
            self.h = 1. / self.nelem
            
            self.bf_1d = lambda x: np.array([0.5*(1.-x), 0.5*(1.+x)])
            self.bf_1d_der = lambda x: np.array([0.5*(0.-1.), 0.5*(0.+1.)])
            self.bf_1d_der2 = lambda x: np.array([0.0, 0.0])

        elif self.fem_basis_deg == 2:
            assert (self.domain_size- 1)%2 == 0
            self.nelem = nelem = int((self.domain_size - 1)/2)
            self.nbf_1d = nbf_1d = 3
            self.nbf_total = nbf_total = self.nbf_1d**self.nsd
            self.h = 1. / self.nelem

            self.bf_1d = lambda x: np.array([
                                        0.5 * x * (x-1.),
                                        (1. - x**2),
                                        0.5 * x * (x+1.)
                                        ], dtype=np.float32)
            self.bf_1d_der = lambda x: np.array([
                                        0.5 * (2.*x-1.),
                                        (- 2.*x),
                                        0.5 * (2.*x+1.)
                                        ], dtype=np.float32)
            self.bf_1d_der2 = lambda x: np.array([
                                        0.5 * (2.),
                                        (- 2.),
                                        0.5 * (2.)
                                        ], dtype=np.float32)
        
        elif self.fem_basis_deg == 3:
            assert (self.domain_size- 1)%3 == 0
            self.nelem = nelem = int((self.domain_size - 1)/3)
            self.nbf_1d = nbf_1d = 4
            self.nbf_total = nbf_total = self.nbf_1d**self.nsd
            self.h = 1. / self.nelem

            self.bf_1d = lambda x: np.array([
                                        (-9. / 16.) * (x**3- x**2 - (1. / 9.) * x + (1. / 9.)),
                                        (27. / 16.) * (x**3 - (1. / 3.) * x**2 - x + (1. / 3.)),
                                        (-27. / 16.) * (x**3 + (1. / 3.) * x**2 - x - (1. / 3.)),
                                        (9. / 16.) * (x**3 + x**2 - (1. / 9.) * x - (1. / 9.))
                                        ], dtype=np.float32)
            self.bf_1d_der = lambda x: np.array([
                                        (-9. / 16.) * (3 * x**2 - 2 * x - (1. / 9.)),
                                        (27. / 16.) * (3 * x**2 - (2. / 3.) * x - 1),
                                        (-27. / 16.) * (3 * x**2 + (2. / 3.) * x - 1),
                                        (9. / 16.) * (3 * x**2 + 2 * x - (1. / 9.))
                                        ], dtype=np.float32)

            self.bf_1d_der2 = lambda x: np.array([
                                        (-9. / 16.) * (6. * x - 2.),
                                        (27. / 16.) * (6. * x - (2. / 3.)),
                                        (-27. / 16.) * (6. * x + (2. / 3.)),
                                        (9. / 16.) * (6. * x + 2.)
                                        ], dtype=np.float32)

    def gauss_guadrature_scheme(self, ngp_1d):
        if ngp_1d == 1:
            gpx_1d = np.array([0.])
            gpw_1d = np.array([2.])
        elif ngp_1d == 2:
            gpx_1d = np.array([-0.577, 0.577])
            gpw_1d = gpw = np.array([1., 1.])
        elif ngp_1d == 3:
            gpx_1d = np.array([-0.774596669, 0., +0.774596669])
            gpw_1d = np.array([5./9., 8./9., 5./9.])
        elif ngp_1d == 4:
            gpx_1d = np.array([-0.861136, -0.339981, +0.339981, +0.861136])
            gpw_1d = np.array([0.347855, 0.652145, 0.652145, 0.347855])
        return gpx_1d, gpw_1d

    def gauss_pt_evaluation(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.N_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der_x(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.dN_x_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der_y(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.dN_y_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der_z(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.dN_z_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der2_x(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_x_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der2_y(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_y_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der2_z(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_z_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der2_xy(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_xy_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der2_yz(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_yz_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der2_zx(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_zx_gp, nsd=self.nsd, stride=(self.nbf_1d-1))
   


class DiffNet2DFEM(DiffNetFEM):
    """docstring for DiffNet2DFEM"""
    def __init__(self, network, dataset, **kwargs):
        super(DiffNet2DFEM, self).__init__(network, dataset, **kwargs)
        assert self.nsd==2
        self.gpw = torch.zeros(self.ngp_total)
        self.N_gp = nn.ParameterList() 
        self.dN_x_gp = nn.ParameterList()
        self.dN_y_gp = nn.ParameterList()
        self.d2N_x_gp = nn.ParameterList()
        self.d2N_y_gp = nn.ParameterList() 
        self.d2N_xy_gp = nn.ParameterList()
        for jgp in range(self.ngp_1d):
            for igp in range(self.ngp_1d):
                N_gp = torch.zeros((self.nbf_1d, self.nbf_1d))
                dN_x_gp = torch.zeros((self.nbf_1d, self.nbf_1d))
                dN_y_gp = torch.zeros((self.nbf_1d, self.nbf_1d))
                d2N_x_gp = torch.zeros((self.nbf_1d, self.nbf_1d))
                d2N_y_gp = torch.zeros((self.nbf_1d, self.nbf_1d))
                d2N_xy_gp = torch.zeros((self.nbf_1d, self.nbf_1d))

                IGP = self.ngp_1d * jgp + igp # tensor product id or the linear id of the gauss point
                self.gpw[IGP] = self.gpw_1d[igp] * self.gpw_1d[jgp]
                for jbf in range(self.nbf_1d):
                    for ibf in range(self.nbf_1d):
                        N_gp[ibf,jbf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf]
                        dN_x_gp[ibf,jbf] = self.bf_1d_der(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * (2 / self.h)
                        dN_y_gp[ibf,jbf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d_der(self.gpx_1d[jgp])[jbf] * (2 / self.h)
                        d2N_x_gp[ibf,jbf] = self.bf_1d_der2(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * (2 / self.h)**2
                        d2N_y_gp[ibf,jbf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d_der2(self.gpx_1d[jgp])[jbf] * (2 / self.h)**2
                        d2N_xy_gp[ibf,jbf] = self.bf_1d_der(self.gpx_1d[igp])[ibf] * self.bf_1d_der(self.gpx_1d[jgp])[jbf] * (2 / self.h)**2
                self.N_gp.append(nn.Parameter(N_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                self.dN_x_gp.append(nn.Parameter(dN_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                self.dN_y_gp.append(nn.Parameter(dN_y_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                self.d2N_x_gp.append(nn.Parameter(d2N_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                self.d2N_y_gp.append(nn.Parameter(d2N_y_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                self.d2N_xy_gp.append(nn.Parameter(d2N_xy_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))



class DiffNet3DFEM(DiffNetFEM):
    """docstring for DiffNet2DFEM"""
    def __init__(self, network, dataset, **kwargs):
        super(DiffNet3DFEM, self).__init__(network, dataset, **kwargs)
        assert self.nsd==3
        self.gpw = torch.zeros(self.ngp_total)
        self.N_gp = nn.ParameterList() 
        self.dN_x_gp = nn.ParameterList()
        self.dN_y_gp = nn.ParameterList()
        self.dN_z_gp = nn.ParameterList()
        self.d2N_x_gp = nn.ParameterList()
        self.d2N_y_gp = nn.ParameterList() 
        self.d2N_z_gp = nn.ParameterList() 
        self.d2N_xy_gp = nn.ParameterList()
        self.d2N_yz_gp = nn.ParameterList()
        self.d2N_zx_gp = nn.ParameterList()
        for kgp in range(self.ngp_1d):
            for jgp in range(self.ngp_1d):
                for igp in range(self.ngp_1d):
                    N_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    dN_x_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    dN_y_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    dN_z_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_x_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_y_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_z_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_xy_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_yz_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_zx_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))

                    IGP = kgp * self.ngp_1d**2 + jgp * self.ngp_1d + igp # tensor product id or the linear id of the gauss point
                    self.gpw[IGP] = self.gpw_1d[igp] * self.gpw_1d[jgp]

                    for kbf in range(self.nbf_1d):
                        for jbf in range(self.nbf_1d):
                            for ibf in range(self.nbf_1d):
                                N_gp[ibf,jbf,kbf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf]
                                dN_x_gp[ibf,jbf,kbf] = self.bf_1d_der(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf] * (2 / self.h)
                                dN_y_gp[ibf,jbf,kbf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d_der(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf] * (2 / self.h)
                                dN_z_gp[ibf,jbf,kbf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d_der(self.gpx_1d[kgp])[kbf] * (2 / self.h)
                                d2N_x_gp[ibf,jbf,kbf] = self.bf_1d_der2(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf] * (2 / self.h)**2
                                d2N_y_gp[ibf,jbf,kbf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d_der2(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf] * (2 / self.h)**2
                                d2N_z_gp[ibf,jbf,kbf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d_der2(self.gpx_1d[kgp])[kbf] * (2 / self.h)**2
                                d2N_xy_gp[ibf,jbf,kbf] = self.bf_1d_der(self.gpx_1d[igp])[ibf] * self.bf_1d_der(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf] * (2 / self.h)**2
                                d2N_yz_gp[ibf,jbf,kbf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d_der(self.gpx_1d[jgp])[jbf] * self.bf_1d_der(self.gpx_1d[kgp])[kbf] * (2 / self.h)**2
                                d2N_zx_gp[ibf,jbf,kbf] = self.bf_1d_der(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d_der(self.gpx_1d[kgp])[kbf] * (2 / self.h)**2

                    self.N_gp.append(nn.Parameter(N_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.dN_x_gp.append(nn.Parameter(dN_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.dN_y_gp.append(nn.Parameter(dN_y_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.dN_z_gp.append(nn.Parameter(dN_z_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_x_gp.append(nn.Parameter(d2N_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_y_gp.append(nn.Parameter(d2N_y_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_z_gp.append(nn.Parameter(d2N_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_xy_gp.append(nn.Parameter(d2N_xy_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_yz_gp.append(nn.Parameter(d2N_yz_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_zx_gp.append(nn.Parameter(d2N_zx_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
