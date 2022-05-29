import os
import sys
import json
import torch
import numpy as np
import scipy.io
from scipy import ndimage
import matplotlib
from skimage import io
# matplotlib.use("pgf")
matplotlib.rcParams.update({
    # 'font.family': 'serif',
    'font.size':12,
})
from matplotlib import pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.base import Callback
seed_everything(42)

import DiffNet
from DiffNet.networks.autoencoders import AE
from DiffNet.DiffNetFEM import DiffNet2DFEM
from DiffNet.DiffNetFDM import DiffNetFDM
import PIL
from torch.utils import data
import torchvision

# sys.path.append('/work/baskarg/bkhara/diffnet/examples/poisson/single_instance')
# from pc_complex_immersed_background import PCVox as PCVoxPoisson
# from pc_complex_immersed_background import Poisson

# def load_pc_normals_from_poisson():
#     version_id = 4
#     domain_size = 256
#     poisson_path = '/work/baskarg/bkhara/diffnet/examples/poisson/single_instance/'
#     cib_path = poisson_path + 'pc_complex_immersed_background/'
#     case_dir = cib_path + 'version_'+str(version_id)
#     # filename = 'bunny-18.png'
#     filename = poisson_path + 'bonefishes-1.png'
#     dataset = PCVoxPoisson(filename, domain_size=domain_size)
#     network = torch.load(os.path.join(case_dir, 'network.pt'))
#     equation = Poisson(network, dataset, batch_size=1, domain_size=domain_size)
#     # Query
#     inputs, forcing = equation.dataset[0:1]
#     up = equation.do_query(inputs, forcing)
#     pc = inputs[0:1,:,:].squeeze()
#     normals = inputs[1:2,:,:].squeeze()
#     u_pts, u_x_pts, u_y_pts = equation.loss_calc_inspect(up, inputs.unsqueeze(0), forcing.unsqueeze(0))
#     grad_vec = torch.stack((u_x_pts.detach().squeeze(), u_y_pts.detach().squeeze()), dim=0)
#     grad_vec = grad_vec.T
#     grad_mag = torch.sqrt(torch.sum(grad_vec**2, dim=1, keepdim=True))
#     grad_vec_unit = grad_vec/grad_mag
#     averaged_normals = np.load(os.path.join(poisson_path, 'renormal.npy'))
#     # return pc.numpy(), grad_vec_unit.numpy(), averaged_normals
#     return pc.numpy(), averaged_normals

class OptimSwitchLBFGS(Callback):
    def __init__(self, epochs=50):
        self.switch_epoch = epochs
        self.print_declaration = False

    def on_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.switch_epoch:
            if not self.print_declaration:
                print("======================Switching to LBFGS after {} epochs ======================".format(self.switch_epoch))
                self.print_declaration = True
            opts = [torch.optim.LBFGS(pl_module.network.parameters(), lr=pl_module.learning_rate, max_iter=5),
                        ]
            trainer.optimizers = opts


def im_io(filepath):
    image = io.imread(filepath).astype(bool).astype(float)
    
    return im2pc(image)
def im2pc(image, nx, ny):
    pc = []
    normals = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] == 1.0:
                boundary = 0
                if i < image.shape[0] - 1:
                    if image[i+1,j] == 0:
                        boundary = 1
                if j < image.shape[1] - 1:
                    if image[i,j+1] == 0:
                        boundary = 1
                if i > 0:
                    if image[i-1,j] == 0:
                        boundary = 1
                if j > 0:
                    if image[i,j-1] == 0:
                        boundary = 1
                if i < image.shape[0] - 1  and j < image.shape[1] - 1:
                    if image[i+1,j+1] == 0:
                        boundary = 1
                if i < image.shape[0] - 1  and j > 0:
                    if image[i+1,j-1] == 0:
                        boundary = 1
                if i > 0 and j < image.shape[1] - 1:
                    if image[i-1,j+1] == 0:
                        boundary = 1
                if i > 0 and j > 0:
                    if image[i-1,j-1] == 0:
                        boundary = 1
                if boundary == 1:
                    pc.append([i+0.5,j+0.5])
                    normals.append([nx[i,j]/(nx[i,j]**2 + ny[i,j]**2), ny[i,j]/(nx[i,j]**2 + ny[i,j]**2)])
    return np.array(pc), np.array(normals)



class PCVox(data.Dataset):
    'PyTorch dataset for PCVox'
    def __init__(self, filename, domain_size=128):
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
        # Define kernel for x differences
        # kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        # # Define kernel for y differences
        # ky = np.array([[1,2,1] ,[0,0,0], [-1,-2,-1]])
        # # Perform x convolution
        # nx = ndimage.convolve(img,kx)
        # # Perform y convolution
        # ny = ndimage.convolve(img,ky)
        # nx = np.divide(nx,(nx**2 + ny**2), out=np.zeros_like(nx), where=((nx**2 + ny**2)!=0))
        # ny = np.divide(ny,(nx**2 + ny**2), out=np.zeros_like(ny), where=((nx**2 + ny**2)!=0))
        # # bc1 will be source, sdf will be set to 0.5 at these locations
        # self.pc, _ = im2pc(img,nx,ny)
        # _, self.normals = load_pc_normals_from_poisson()
        # self.pc = self.pc/(img.shape[0])
        pt_cloud = []
        for _ in range(1000):
            vec = np.random.randn(2)
            vec /= 4*np.linalg.norm(vec)
            pt_cloud.append(vec)
        pt_cloud = np.array(pt_cloud)
        self.normals = pt_cloud*4.0
        self.pc = pt_cloud + 0.5
        self.domain = np.ones((domain_size,domain_size))
        self.domain_size = domain_size
        self.n_samples = 100
    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples
    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.pc, self.normals]) # 2, Npoint, 2
        forcing = np.ones_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)



class Eiqonal(DiffNet2DFEM,DiffNetFDM):
    """docstring for Eiqonal"""

    def __init__(self, network, dataset, **kwargs):
        super(Eiqonal, self).__init__(network, dataset, **kwargs)
        self.mapping_type = kwargs.get('mapping_type', 'no_network')
        self.save_frequency = kwargs.get('save_frequency', 5)
        self.loss_type = kwargs.get('loss_type', 'FDM')

        self.network = network
        self.tau = 0.25

        # for plotting
        self.domain_loss = 0
        self.boundary_loss = 0
        self.reg_loss = 0
        self.normals_loss = 0


    def Q1_vector_assembly(self, Aglobal, Aloc_all):
        Aglobal[:,0, 0:-1, 0:-1] += Aloc_all[:,0, :, :]
        Aglobal[:,0, 0:-1, 1:  ] += Aloc_all[:,1, :, :]
        Aglobal[:,0, 1:  , 0:-1] += Aloc_all[:,2, :, :]
        Aglobal[:,0, 1:  , 1:  ] += Aloc_all[:,3, :, :]
        return Aglobal


    def lossFDM(self, u, inputs_tensor, forcing_tensor):
        f = forcing_tensor # renaming variable
        # init bin widths
        hx = self.h
        hy = self.h
        u_x = self.derivative_x(self.pad(u))
        u_y = self.derivative_y(self.pad(u))
        R1 = (u_x**2 + u_y**2) - 1.0
        # extract diffusivity and boundary conditions here
        pc = inputs_tensor[:,0:1,:,:].type_as(f)
        normals = inputs_tensor[:,1:2,:,:].type_as(f)
        # apply boundary conditions
        nidx = (pc[:,:,:,0]/self.hx).type(torch.LongTensor).to(pc.device)
        nidy = (pc[:,:,:,1]/self.hy).type(torch.LongTensor).to(pc.device)
        u_pts_grid =  torch.stack([
                torch.stack([
                    torch.stack([u[b,0,nidx[b,0,:],nidy[b,0,:]] for b in range(u.size(0))]),
                    torch.stack([u[b,0,nidx[b,0,:]+1,nidy[b,0,:]] for b in range(u.size(0))])]),
                torch.stack([
                    torch.stack([u[b,0,nidx[b,0,:],nidy[b,0,:]+1] for b in range(u.size(0))]),
                    torch.stack([u[b,0,nidx[b,0,:]+1,nidy[b,0,:]+1] for b in range(u.size(0))])])
                ]).unsqueeze(2)
        x_pts = pc[:,:,:,0] - nidx.type_as(pc)*self.hx 
        y_pts = pc[:,:,:,1] - nidy.type_as(pc)*self.hy
        xi_pts = (x_pts*2)/self.hx - 1
        eta_pts = (y_pts*2)/self.hy - 1
        # print(xi_pts, eta_pts)
        N_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        dN_x_values_pts = self.bf_1d_der_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        dN_y_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_der_th(eta_pts).unsqueeze(1)
        u_pts = torch.sum(torch.sum(N_values_pts*u_pts_grid,0),0)
        u_x_pts = torch.sum(torch.sum(dN_x_values_pts*u_pts_grid,0),0)
        u_y_pts = torch.sum(torch.sum(dN_y_values_pts*u_pts_grid,0),0)
        # Second loss - boundary loss
        sdf_recon_loss = torch.sum(u_pts**2)
        # Third loss - boundary reconstruction
        normals_loss = torch.nn.functional.mse_loss(u_x_pts, normals[:,:,:,0], reduction='sum') + torch.nn.functional.mse_loss(u_y_pts, normals[:,:,:,1], reduction='sum')
        # res_elmwise2 = JxW * normals
        # res_elmwise2 = JxW * (normals_lhs - normals_rhs)
        # Assemble 
        # print('***'*10)
        # print(torch.sum(res_elmwise1, 1).shape)
        # print('***'*10)
        # R1 = torch.zeros_like(u); R1 = self.Q1_vector_assembly(R1, torch.sum(res_elmwise1, 1))
        # reg_loss = torch.exp(-100*(u_reg**2))
        loss = torch.mean(R1**2) + sdf_recon_loss + normals_loss
        # loss = torch.norm(R1, 'fro') + 1000*sdf_recon_loss + 10*normals_loss
        # loss = torch.mean(res_elmwise1**2) + sdf_recon_loss
        return loss


    def loss(self, u, inputs_tensor, forcing_tensor):
        f = forcing_tensor # renaming variable
        # init bin widths
        hx = self.h
        hy = self.h
        # init vars for weak formulation
        N_values = self.Nvalues.type_as(u)
        gpw = self.gpw.type_as(u)
        
        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)
        # Eikonal Residual on the domain
        trnsfrm_jac = (0.5*hx)*(0.5*hy)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        # eikonal_lhs = N_values*(u_x_gp**2 + u_y_gp**2)
        # eikonal_rhs = 1.0
        # res_elmwise1 = torch.sum(JxW * (eikonal_lhs), 2)# \nabla \phi - 1 = 0  JxW addresses discretization of domain
        # R1 = torch.zeros_like(u); R1 = self.Q1_vector_assembly(R1, res_elmwise1)

        R1 = torch.sum(u_x_gp**2 + u_y_gp**2, 2) - 1.0
        # First loss - eikonal eqn ie   grad(u) = 1
        eikonal_lhs = (N_values * u_x_gp)**2 + (N_values * u_y_gp)**2
        eikonal_rhs = N_values * 1.0 
        res_elmwise1 = JxW * (eikonal_lhs - eikonal_rhs) # \nabla \phi - 1 = 0  JxW addresses discretization of domain
        R_split_1 = torch.sum(res_elmwise1, 2) # sum across all GP
        R1 = torch.zeros_like(u); R1 = self.Q1_vector_assembly(R1, R_split_1)

        # add boundary conditions to R <---- this step is very important
        # R1 = torch.where(bc1>0.5,u_bc,R1)

        # return R1
        # return torch.norm(R1, 'fro')

        # First loss regularizer
        # res_elmwise1_regularizer = torch.exp(-100*torch.abs(u_gp))

        # Second loss - normals
        # normals = 1.0 - (u_x_gp*bc2_gp + u_y_gp*bc3_gp) 
        # normals_lhs =  (N_values**2 * u_x_gp * bc2_gp) + (N_values**2 * u_y_gp * bc3_gp) # treat this as a directional eikonal??
        # normals_rhs = N_values * 1. 
        # normals = (u_x_gp - bc2_gp) + (u_y_gp - bc3_gp) # this makes more sense ie mse(pred_u_normals, normals)
        # res_elmwise2 = JxW * normals
        # res_elmwise2 = JxW * (normals_lhs - normals_rhs)

        # Third loss - boundary reconstruction
        sdf_recon_loss = torch.sum(sdf_recon**2)

        # res_elmwise2 = transformation_jacobian * (1.0 - u_x_gp*bc2_gp - u_y_gp*bc3_gp) 
        # res_elmwise = torch.sum(res_elmwise1, 1) + 10*torch.sum(res_elmwise2, 1) + torch.sum(res_elmwise1_regularizer,1)# + sdf_boundary_recon
        res_elmwise = torch.norm(R1, 'fro') + sdf_recon_loss # + torch.sum(res_elmwise1_regularizer, 1)
        loss = torch.mean(res_elmwise) 
        return loss

    def loss1(self, u, inputs_tensor, forcing_tensor):
        # J = \int ( |grad u|^2-1)^2 d\Omega

        f = forcing_tensor # renaming variable

        # init bin widths
        hx = self.h
        hy = self.h

        # init vars for weak formulation
        N_values = self.Nvalues.type_as(f)
        gpw = self.gpw.type_as(f)

        # extract diffusivity and boundary conditions here
        bc1 = inputs_tensor[:,1:2,:,:]  # sdf = 0.5 at boundary else 0.
        bc2 = inputs_tensor[:,2:3,:,:]  # normals in the x-direction
        bc3 = inputs_tensor[:,3:4,:,:]  # normals in the y-direction
        u = torch.unsqueeze(u, 0)

        # apply boundary conditions
        # build reconstruction boundary loss
        zeros = torch.zeros_like(u)
        sdf_recon = torch.where(bc1>0.5, u, zeros)
        u = torch.where(bc1>0.5, zeros, u)

        # u = torch.where(bc1>0.5, u*0.0, u) # this may give misleading gradient if we are trying to learn B.C.s

        u_gp = self.gauss_pt_evaluation(u)
        # bc2_gp = self.gauss_pt_evaluation(bc2)
        # bc3_gp = self.gauss_pt_evaluation(bc3)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        mag_grad_u = u_x_gp**2 + u_y_gp**2
        integrand = (mag_grad_u - 1)**2

        # bin width (reimann sum)
        trnsfrm_jac = (0.5*hx)*(0.5*hy)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        J = torch.sum(integrand*JxW, 1)

        # Third loss - boundary reconstruction
        sdf_recon_loss = torch.sum(sdf_recon**2)

        # res_elmwise2 = transformation_jacobian * (1.0 - u_x_gp*bc2_gp - u_y_gp*bc3_gp) 
        # res_elmwise = torch.sum(res_elmwise1, 1) + 10*torch.sum(res_elmwise2, 1) + torch.sum(res_elmwise1_regularizer,1)# + sdf_boundary_recon
        res_elmwise = J # + sdf_recon_loss # + torch.sum(res_elmwise1_regularizer, 1)
        loss = torch.sum(res_elmwise) 
        return loss

    def loss3(self, u, inputs_tensor, forcing_tensor):
        # -(\grad v, u\grad u) + (1+\tau) * (\grad (vu), \grad u) = (v,1)
        tau = self.tau

        f = forcing_tensor # renaming variable

        # init bin widths
        hx = self.h
        hy = self.h

        # init vars for weak formulation
        N_values = self.Nvalues.type_as(f)
        dN_x_values = self.dN_x_values.type_as(f)
        dN_y_values = self.dN_y_values.type_as(f)
        gpw = self.gpw.type_as(f)

        # extract diffusivity and boundary conditions here
        bc1 = inputs_tensor[:,1:2,:,:]  # sdf = 0.5 at boundary else 0.
        bc2 = inputs_tensor[:,2:3,:,:]  # normals in the x-direction
        bc3 = inputs_tensor[:,3:4,:,:]  # normals in the y-direction
        u = torch.unsqueeze(u, 0)

        # apply boundary conditions
        # build reconstruction boundary loss
        zeros = torch.zeros_like(u)
        sdf_recon = torch.where(bc1>0.5, u, zeros)

        # u = torch.where(bc1>0.5, u*0.0, u) # this may give misleading gradient if we are trying to learn B.C.s

        u_gp = self.gauss_pt_evaluation(u)
        # bc2_gp = self.gauss_pt_evaluation(bc2)
        # bc3_gp = self.gauss_pt_evaluation(bc3)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        # bin width (reimann sum)
        trnsfrm_jac = (0.5*hx)*(0.5*hy)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        # First loss - eikonal eqn ie   grad(u) = 1
        eikonal_lhs = tau*u_gp*(dN_x_values*u_x_gp + dN_y_values*u_y_gp) + (1+tau)*N_values*(u_x_gp**2+u_y_gp**2)
        eikonal_rhs = N_values * 1.0 

        res_elmwise1 = JxW * (eikonal_lhs - eikonal_rhs) # \nabla \phi - 1 = 0  JxW addresses discretization of domain        
        R_split_1 = torch.sum(res_elmwise1, 2) # sum across all GP
        R1 = torch.zeros_like(u); R1 = self.Q1_vector_assembly(R1, R_split_1)

        # add boundary conditions to R <---- this step is very important
        # R1 = torch.where(bc1>0.5,u_bc,R1)

        # return R1
        # return torch.norm(R1, 'fro')
        boundary_gaussian = torchvision.transforms.functional.gaussian_blur(bc1, (3,3))
        psi_func = torch.sin(u)
        directionality_loss = torch.sum((boundary_gaussian*psi_func - boundary_gaussian*u)**2)

        # res_elmwise2 = transformation_jacobian * (1.0 - u_x_gp*bc2_gp - u_y_gp*bc3_gp) 
        # res_elmwise = torch.sum(res_elmwise1, 1) + 10*torch.sum(res_elmwise2, 1) + torch.sum(res_elmwise1_regularizer,1)# + sdf_boundary_recon
        res_elmwise = torch.norm(R1, 'fro') + sdf_recon_loss + directionality_loss # + torch.sum(res_elmwise1_regularizer, 1)
        loss = torch.mean(res_elmwise) 
        return loss

    def loss4(self, u, inputs_tensor, forcing_tensor):
        # -(\grad v, u\grad u) + (1+\tau) * (\grad (vu), \grad u) = (v,1)
        tau = self.tau

        f = forcing_tensor # renaming variable

        # init bin widths
        hx = self.h
        hy = self.h

        # init vars for weak formulation
        N_values = self.Nvalues.type_as(u)
        dN_x_values = self.dN_x_values.type_as(f)
        dN_y_values = self.dN_y_values.type_as(f)
        gpw = self.gpw.type_as(u)

        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        # Eikonal Residual on the domain
        trnsfrm_jac = (0.5*hx)*(0.5*hy)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        # First loss - eikonal eqn ie   grad(u) = 1
        # print('^ '*10)
        # print('u_gp ', u_gp.shape)
        # print('dN_x_values ',dN_x_values.shape)
        # print('u_x_gp ', u_x_gp.shape)
        # print('N_values ', N_values.shape)
        
        # exit()
        eikonal_lhs = tau*u_gp*(dN_x_values*u_x_gp + dN_y_values*u_y_gp) + (1+tau)*N_values*(u_x_gp**2+u_y_gp**2)
        eikonal_rhs = N_values * 1.0 

        res_elmwise1 = JxW * (eikonal_lhs - eikonal_rhs) # \nabla \phi - 1 = 0  JxW addresses discretization of domain        
        R_split_1 = torch.sum(res_elmwise1, 2) # sum across all GP
        R1 = torch.zeros_like(u); R1 = self.Q1_vector_assembly(R1, R_split_1)

        # extract diffusivity and boundary conditions here
        pc = inputs_tensor[:,0:1,:,:].type_as(f)
        normals = inputs_tensor[:,1:2,:,:].type_as(f)
        # apply boundary conditions
        nidx = (pc[:,:,:,0]/self.hx).type(torch.LongTensor).to(pc.device)
        nidy = (pc[:,:,:,1]/self.hy).type(torch.LongTensor).to(pc.device)


        # print('* '*10)
        # print(u.shape)
        # print(nidx.shape)
        # exit()
        u_pts_grid =  torch.stack([
                torch.stack([
                    torch.stack([u[b,0,nidx[b,0,:],nidy[b,0,:]] for b in range(u.size(0))]),
                    torch.stack([u[b,0,nidx[b,0,:]+1,nidy[b,0,:]] for b in range(u.size(0))])]),
                torch.stack([
                    torch.stack([u[b,0,nidx[b,0,:],nidy[b,0,:]+1] for b in range(u.size(0))]),
                    torch.stack([u[b,0,nidx[b,0,:]+1,nidy[b,0,:]+1] for b in range(u.size(0))])])
                ]).unsqueeze(2)

        print('* '*10)

        x_pts = pc[:,:,:,0] - nidx.type_as(pc)*self.hx 
        y_pts = pc[:,:,:,1] - nidy.type_as(pc)*self.hy
        xi_pts = (x_pts*2)/self.hx - 1
        eta_pts = (y_pts*2)/self.hy - 1
        # print(xi_pts, eta_pts)
        N_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        dN_x_values_pts = self.bf_1d_der_th(xi_pts).unsqueeze(0)*self.bf_1d_th(eta_pts).unsqueeze(1)
        dN_y_values_pts = self.bf_1d_th(xi_pts).unsqueeze(0)*self.bf_1d_der_th(eta_pts).unsqueeze(1)
        u_pts = torch.sum(torch.sum(N_values_pts*u_pts_grid,0),0)
        u_x_pts = torch.sum(torch.sum(dN_x_values_pts*u_pts_grid,0),0)
        u_y_pts = torch.sum(torch.sum(dN_y_values_pts*u_pts_grid,0),0)

        # Second loss - boundary loss
        sdf_recon_loss = torch.sum((u_pts - 0.0)**2)
        # Second loss - normals
        # normals = 1.0 - (u_x_gp*bc2_gp + u_y_gp*bc3_gp) 
        # normals_lhs =  (N_values**2 * u_x_gp * bc2_gp) + (N_values**2 * u_y_gp * bc3_gp) # treat this as a directional eikonal??
        # normals_rhs = N_values * 1. 
        # normals = (u_x_gp - bc2_gp) + (u_y_gp - bc3_gp) # this makes more sense ie mse(pred_u_normals, normals)
        # res_elmwise2 = JxW * normals
        # res_elmwise2 = JxW * (normals_lhs - normals_rhs)
        # Third loss - boundary reconstruction
        # normals_loss = torch.nn.functional.mse_loss(u_x_pts, normals[:,:,:,0]) + torch.nn.functional.mse_loss(u_y_pts, normals[:,:,:,1])
        normals_loss = torch.sum((u_x_pts*normals[:,:,:,0]+u_y_pts*normals[:,:,:,1] - 1.0)**2)

        # res_elmwise2 = JxW * normals
        # res_elmwise2 = JxW * (normals_lhs - normals_rhs)
        # Assemble 
        # print('***'*10)
        # print(torch.sum(res_elmwise1, 1).shape)
        # print('***'*10)
        # R1 = torch.zeros_like(u); R1 = self.Q1_vector_assembly(R1, torch.sum(res_elmwise1, 1))

        reg_loss = torch.exp(-100*(torch.abs(u)))
        # if self.current_epoch < 3:
        #     loss = 10*torch.norm(R1, 'fro') + 1000*sdf_recon_loss  + 5*reg_loss
        # else:
        #     loss = 100*torch.norm(R1, 'fro') + 1000*sdf_recon_loss 

        # for plotting purposes
        self.domain_loss = R1.detach().squeeze().cpu().clone()

        loss = torch.norm(R1, 'fro')
        loss = loss + sdf_recon_loss
        loss = loss + normals_loss
        # if self.current_epoch < 400: # TODO: make it better
        #     loss = loss + 0.01*reg_loss
        # print()
        # print('R1:', torch.norm(R1, 'fro').item())
        # print('SDF loss:', sdf_recon_loss.item())
        # print('normals loss:', normals_loss.item())
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        if self.mapping_type == 'no_network':
            return self.network[0], inputs_tensor, forcing_tensor
        elif self.mapping_type == 'network':
            nu = inputs_tensor[:,0:1,:,:]
            return self.network(nu), inputs_tensor, forcing_tensor
    def configure_optimizers(self):
        """
        Configure optimizer for network parameters
        """
        lr = self.learning_rate
        # opts = [torch.optim.LBFGS(self.network.parameters(), lr=1.0, max_iter=5)]
        opts = [torch.optim.Adam(self.network.parameters(), lr=lr)]
        return opts

    def training_step(self, batch, batch_idx):
    # def training_step(self, batch, batch_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        if self.loss_type == 'FEM':
            loss_val = self.loss4(u, inputs_tensor, forcing_tensor).mean()
        elif self.loss_type == 'FDM':
            loss_val = self.lossFDM(u, inputs_tensor, forcing_tensor).mean()
        # loss_val = self.loss_resmin(u, inputs_tensor, forcing_tensor).mean()
        # loss_val = self.loss_resmin_mass(u, inputs_tensor, forcing_tensor).mean()
        # loss_val = self.loss_matvec(u, inputs_tensor, forcing_tensor).mean()
        # self.log('PDE_loss', loss_val.item())
        # self.log('loss', loss_val.item())
        return {"loss": loss_val}

    def do_query(self, inputs, forcing):
        u, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), 
                                                         forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

        u = u.squeeze().detach().cpu()
        return u

    # def calc_res_for_plot(self, u):
    #     tau = self.tau

    #     # init bin widths
    #     hx = self.h
    #     hy = self.h

    #     # init vars for weak formulation
    #     N_values = self.Nvalues.type_as(u)
    #     dN_x_values = self.dN_x_values.type_as(u)
    #     dN_y_values = self.dN_y_values.type_as(u)
    #     gpw = self.gpw.type_as(u)


    #     u_gp = self.gauss_pt_evaluation(u)
    #     u_x_gp = self.gauss_pt_evaluation_der_x(u)
    #     u_y_gp = self.gauss_pt_evaluation_der_y(u)

    #     # bin width (reimann sum)
    #     trnsfrm_jac = (0.5*hx)*(0.5*hy)
    #     JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

    #     # First loss - eikonal eqn ie   grad(u) = 1
    #     eikonal_lhs = tau*u_gp*(dN_x_values*u_x_gp + dN_y_values*u_y_gp) + (1+tau)*N_values*(u_x_gp**2+u_y_gp**2)
    #     eikonal_rhs = N_values * 1.0 

    #     res_elmwise1 = JxW * (eikonal_lhs - eikonal_rhs) # \nabla \phi - 1 = 0  JxW addresses discretization of domain        
    #     R_split_1 = torch.sum(res_elmwise1, 2) # sum across all GP
    #     R1 = torch.zeros_like(u); R1 = self.Q1_vector_assembly(R1, R_split_1)

    #     return R1

    def on_epoch_end(self):     
        if self.current_epoch % self.save_frequency == 0:
            self.network.eval()
            inputs, forcing = self.dataset[0]
            u = self.do_query(inputs, forcing)  
            # R1 = self.calc_res_for_plot(u)  
            # R1 = R1.squeeze().detach().cpu()    
            R1 = self.domain_loss             
            self.plot_contours((u,R1))

    def plot_contours(self, field_tuple):
        # unpack
        u = field_tuple[0]
        R1 = field_tuple[1]

        fig, axs = plt.subplots(1, 5, figsize=(2*7,2),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        im0 = axs[0].imshow(u,cmap='jet')
        fig.colorbar(im0, ax=axs[0])
        axs[0].set_title('u')

        im1 = axs[1].imshow(R1, vmin=0, vmax=1, cmap='gray')
        fig.colorbar(im1, ax=axs[1])
        axs[1].set_title('Eikonal residual', fontsize=8)

        im2 = axs[2].imshow(abs(u), cmap='jet')
        fig.colorbar(im2, ax=axs[2])
        axs[2].set_title('Unsigned Distance Field', fontsize=8)

        # im3 = axs[3].imshow(bc1,cmap='jet')
        # fig.colorbar(im3, ax=axs[3])  
        # axs[3].set_title('source')
        # im4 = axs[4].imshow(bc2,cmap='jet')
        # fig.colorbar(im4, ax=axs[4])
        # axs[4].set_title('nrmls x')
        # im5 = axs[5].imshow(bc3,cmap='jet')
        # fig.colorbar(im5, ax=axs[5])  
        # axs[5].set_title('nrmls y')
        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    Nx = Ny = 256 # equal
    LR=3e-4
    max_epochs = 1
    save_frequency = 25
    opt_switch_epochs = max_epochs
    domain_size = Nx
    mapping_type = 'no_network'
    load_from_prev = False
    load_version_id = 34
    dir_string = "curve_reconstruction_debug"

    if load_from_prev:
        print("LOADING FROM PREVIOUS VERSION: ", load_version_id)
        case_dir = os.path.join('.', dir_string, 'version_'+str(load_version_id))
        network = torch.load(os.path.join(case_dir, 'network.pt'))
    else:
        print("INITIALIZING PARAMETERS: ZERO / RANDOM")
        if mapping_type == 'no_network':
            u_tensor = np.random.randn(1,1,Ny, Nx)
            network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
        elif mapping_type == 'network':
            network = AE(in_channels=1, out_channels=1, dims=Nx, n_downsample=3)
    dataset = PCVox('./images/bonefishes-1.png', domain_size=domain_size)
    basecase = Eiqonal(network, dataset, batch_size=1, fem_basis_deg=1, domain_size=Nx, domain_length=1.0, learning_rate=LR, mapping_type=mapping_type, loss_type='FEM', save_frequency=save_frequency)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name=dir_string)
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=max_epochs, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    lbfgs_switch = OptimSwitchLBFGS(epochs=opt_switch_epochs)
    trainer = Trainer(gpus=[0],callbacks=[lbfgs_switch],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=max_epochs, deterministic=True, profiler="simple")

    # ------------------------
    # 4 Training
    # ------------------------
    trainer.fit(basecase)
    # ------------------------
    # 5 SAVE NETWORK
    # ------------------------
    torch.save(basecase.network, os.path.join(logger.log_dir, 'network.pt'))
if __name__ == '__main__':
    main()