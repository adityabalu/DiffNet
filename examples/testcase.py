import torch
from torch import nn
import numpy as np
from .fem import FEM
from .fdm import FDM
import time

from .gen_input_calc import create_grid, get_fields_of_coefficients, create_forcing_tensor

class TestCase(FEM,FDM):
    def __init__(self, args):
        if args.loss_calc_type == 'fem':
            FEM.__init__(args)
        else:
            FDM.__init__(args)
        self.x, self.y, self.z = create_grid(args)

    def loss(self, g, nu, forcing):
        if self.loss_calc_type == 'fem':
            if self.nsd == 2:
                return self.loss_2d_fem(g, nu, forcing)
            elif self.nsd == 3:
                return self.loss_3d_fem(g, nu, forcing)
        elif self.loss_calc_type == 'fdm':
            if self.nsd == 2:
                return self.loss_2d(g, nu, forcing)
            elif self.nsd == 3:
                return self.loss_3d(g, nu, forcing)

    def boundary_loss(self, g, list_of_bc, output_dim):
        if self.nsd == 2:
            return self.boundary_loss_2d(g, list_of_bc, output_dim)
        elif self.nsd == 3:
            return self.boundary_loss_3d(g, list_of_bc, output_dim)

    def loss_2d(self, g, nu, forcing):
        advection = 1.0 
        
        g_pd = self.apply_bc(g)
        nu_pd = self.apply_padding_on(nu)
        
        g_x = nn.functional.conv2d(g_pd, self.sobelx)
        g_y = nn.functional.conv2d(g_pd, self.sobely)
        
        res = (advection * nu * g_x + g_y) - forcing # nu is constant and is also the input to gen

        res_flat = res.view(g.shape[0], -1)
        res_norm_1 = torch.norm(res_flat, p=1, dim=1) #/(output_dim*output_dim)
        res_norm_2 = torch.norm(res_flat, p=2, dim=1) #/(output_dim*output_dim)

        c1 = self.c1
        c2 = self.c2

        res_norm = c1 * res_norm_1 + c2 * res_norm_2
        return res_norm

    def loss_2d_fem(self, g, nu, forcing):        
        # g = self.apply_bc(args, g)
        # nu = self.apply_padding_on(args, nu)
        # f = self.apply_padding_on(args, forcing)

        # stride = args.fem_basis_deg

        # nu_gp = self.gauss_pt_evaluations(nu, stride=stride)
        
        # f_gp = self.gauss_pt_evaluations(f, stride=stride)
        
        # u_gp = self.gauss_pt_evaluations(g, stride=stride)        
        # u_x_gp = self.gauss_pt_evaluations_der_x(g, stride=stride)
        # u_y_gp = self.gauss_pt_evaluations_der_y(g, stride=stride)

        # advection = 1.0
        # transformation_jacobian = (0.25 * self.h ** 2)
        
        # res_elmwise = transformation_jacobian * sum(
        #                 self.gpw[i] * (advection * nu_gp[i] * u_x_gp[i] + u_y_gp[i])**2
        #                 for i in range(self.ngp_total))

        # res_flat = res_elmwise.view(g.shape[0], -1)
        # loss = torch.sum(res_flat, 1)        
        # return loss

        ####################################################################################################

        # v = torch.zeros(forcing.shape).to(device)
        # v[:,:,self.test_j,self.test_i] = 1.0
        # v = nn.functional.pad(v, (1,1,1,1), 'constant', value=0)

        # g = self.apply_bc(args, g)
        # g = nn.functional.pad(g, (1,0,0,0), 'constant', value=0)
        # g = nn.functional.pad(g, (0,1,1,1), 'replicate')
        # x = torch.linspace(0,1,args.output_size + 2)
        # g[0,0,0,:] = torch.exp(-(x-0.4)**2/0.01)
        # print(g, flush=True)
        # nu = self.apply_padding_on(args, nu)
        # f = self.apply_padding_on(args, forcing)


        stride = self.args.fem_basis_deg

        sinx = (torch.sin(np.pi * self.x))**2
        x_sq = self.x**2

        nu = self.apply_padding_on(nu)
        sinx = self.apply_padding_on(sinx)
        x_sq = self.apply_padding_on(x_sq)

        sinx_gp = self.gauss_pt_evaluations(sinx, stride=stride)
        x_sq_gp = self.gauss_pt_evaluations(x_sq, stride=stride)

        nu_gp = self.gauss_pt_evaluations(nu, stride=stride)
        
        # f_gp = self.gauss_pt_evaluations(f, stride=stride)
        
        u_gp = self.gauss_pt_evaluations(g, stride=stride)        
        # u_x_gp = self.gauss_pt_evaluations_der_x(g, stride=stride)
        # u_y_gp = self.gauss_pt_evaluations_der_y(g, stride=stride)
        # u_xx_gp = self.gauss_pt_evaluations_der_xx(g, stride=stride)
        # u_yy_gp = self.gauss_pt_evaluations_der_yy(g, stride=stride)

        # v_gp = self.gauss_pt_evaluations(v, stride=stride)
        # v_x_gp = self.gauss_pt_evaluations_der_x(v, stride=stride)

        # print(v, flush=True)
        # print(v_gp)
        # print(v_x_gp)
        # print("sizes = ", v.shape, v_gp[0].shape, v_x_gp[0].shape)
        # exit()


        # print("f_gp = ", f_gp)
        # print("nu_gp = ", nu_gp)
        # print("u_gp = ", u_gp)
        # print("f_gp shape = ", len(f_gp))
        # print("nu_gp shape = ", len(nu_gp))
        # exit()


        Pe = 0.5
        nu_coeff = 1./(2. * self.args.output_size * Pe)
        advection = 1.0
        transformation_jacobian = (0.25 * self.h ** 2)

        stab = (self.args.fem_basis_deg / self.args.output_size) / (2. * advection) / np.sqrt(1. + 9. / Pe**2)

        

        # res_elmwise = transformation_jacobian * sum(
        #                 self.gpw[i] * (nu_gp[i] * u_gp[i] - 0.15 * torch.ones_like(nu_gp[0]))**2
        #                 for i in range(self.ngp_total))

        res_test_elm = transformation_jacobian * sum(
                        self.gpw[i] * sinx_gp[i] * nu_gp[i]
                        for i in range(self.ngp_total))

        res_test_flat = res_test_elm.view(g.shape[0], -1)
        loss_testcase = torch.sum(res_test_flat, 1)
        print("loss_testcase = ", loss_testcase, flush=True)
        print("size of gp = ", len(nu_gp))
        print("self.ngp_total = ", self.ngp_total)

        exit()

        # x_gp = self.gauss_pt_evaluations(self.x, stride=stride)
        # y_gp = self.gauss_pt_evaluations(self.y, stride=stride)

        # res_elmwise = transformation_jacobian * sum(
        #                 self.gpw[i] * (x_gp[i]**4)
        #                 for i in range(self.ngp_total))


        # # Alternative way, without pythonic generator
        # res_elmwise = torch.zeros(g_gp_list[0].size()).to(device)
        # for i in range(self.ngp_total):
        #     res_elmwise += (0.25 * self.h ** 2) * self.gpw[i] * (nu_gp_list[i] * (g_x_gp_list[i]**2 + g_y_gp_list[i]**2) - 2. * (g_gp_list[i] * f_gp_list[i])) 

        res_flat = res_elmwise.view(g.shape[0], -1)
        loss = torch.sum(res_flat, 1)
        # print(res_flat)
        # print(loss)
        # exit()
        
        return loss

    def loss_3d(self, g, nu, forcing):
        raise NotImplementedError()

    def loss_3d_fem(self, g, nu, forcing):
        raise NotImplementedError()

    def boundary_loss_2d(self, g, list_of_bc, output_dim):     
        raise NotImplementedError()

    def boundary_loss_3d(self, g, list_of_bc, output_dim):     
        raise NotImplementedError()

    def prepare_inputs(self, coeff_mini_batch):
        all_coeff_tensors = get_fields_of_coefficients(self.args, coeff_mini_batch, self.args.output_size)        
        forcing_4d_tensor = torch.zeros_like(self.x)
        nu_4d_tensor = torch.ones_like(self.x)
        gen_input = nu_4d_tensor
        return gen_input, nu_4d_tensor, forcing_4d_tensor

    def apply_bc(self, g):
        if args.fem_basis_deg == 1:
            g = nn.functional.pad(g, (1,0,0,0), 'constant', value=0)
            g = nn.functional.pad(g, (0,1,1,1), 'replicate')
            x = torch.linspace(0,1,args.output_size + 2)
            mu = 0.4
            sigma = 0.01
            g[0,0,0,:] = torch.exp(-(x-mu)**2/sigma) # a guassian pulse
        elif args.fem_basis_deg == 2:
            g = nn.functional.pad(g, (0,1,1,1), 'replicate')
            g = nn.functional.pad(g, (1,0,0,0), 'constant', value=0)
            g = nn.functional.pad(g, (0,1,0,1), 'replicate')
            # g = nn.functional.pad(g, (0,1,0,0), 'constant', value=0)                    
            mu = 0.4
            sigma = 0.01
            x = torch.linspace(0,1,args.output_size + 3)
            g[0,0,0,:] = torch.exp(-(x-mu)**2/sigma) # a guassian pulse
        elif args.fem_basis_deg == 3:
            g = nn.functional.pad(g, (1,1,1,1), 'replicate')
            g = nn.functional.pad(g, (0,0,1,1), 'replicate')
            g = nn.functional.pad(g, (0,1,0,0), 'constant', value=0)
            g = nn.functional.pad(g, (1,0,0,0), 'constant', value=0)
        return g


    def apply_padding_on(self, field):
        if self.args.fem_basis_deg == 1:
            field = nn.functional.pad(field, (1, 1, 1, 1), 'replicate')
            return field
        elif self.args.fem_basis_deg == 2:
            field = nn.functional.pad(field, (1,1,1,1), 'replicate')
            field = nn.functional.pad(field, (0,1,0,1), 'replicate') 
            return field                
        elif self.args.fem_basis_deg == 3:
            field = nn.functional.pad(field, (2, 2, 2, 2), 'replicate')
            return field
