import os
import json
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule
from collections import OrderedDict

import matplotlib
# matplotlib.use("pgf")
matplotlib.rcParams.update({
    # 'font.family': 'serif',
    'font.size':12,
})
from matplotlib import pyplot as plt
import os
import sys
import json


def gauss_pt_eval(self, tensor, N, nsd=2, stride=1):
    if nsd == 2:
        conv_gp = nn.functional.conv2d
    elif nsd == 3:
        conv_gp = nn.functional.conv3d

    result_list = []
    for i in range(len(N)):
        result_list.append(conv_gp(tensor, N[i], stride=stride))
    return torch.cat(result_list, 1)


class PDE(LightningModule):
    """
    PDE Base Class
    """

    def __init__(self, dataset, network, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.nsd = args.nsd

        # if args.nsd == 2:
        #     print("2D gen")
        #     self.network = GoodGenerator2D(in_channels=3, out_channels=1)
        # elif args.nsd == 3:
        #     print("3D gen")
        #     self.network = GoodGenerator3D(in_channels=1, out_channels=1)
        self.dataset = dataset
        self.network = network
        self.vis_sample_ids = [0,1]
        if self.args.dataset.sampling_method == 'parametric':
            self.vis_sample_ids = [0,1,2,3,4,5,6]
            self.validation_batch_size = len(self.vis_sample_ids)

        nsd = self.nsd
        self.output_size = args.output_size
        self.ngp_1d = args.pde.ngp_1d
        self.fem_basis_deg = args.pde.fem_basis_deg

        # Gauss quadrature setup
        if self.fem_basis_deg == 1:
            ngp_1d = 2
        elif self.fem_basis_deg == 2:
            ngp_1d = 3
        elif self.fem_basis_deg == 3:
            ngp_1d = 3

        if ngp_1d > self.ngp_1d:
            ngp_1d = self.ngp_1d
            self.ngp_1d = self.ngp_1d
        
        self.ngp_total = ngp_total = self.ngp_1d**self.nsd
        gpx_1d, gpw_1d = self.gauss_guadrature_scheme(self.ngp_1d)

        # Basis functions setup
        if self.fem_basis_deg == 1:
            self.nelem = nelem = self.output_size - 1
            self.nbf_1d = nbf_1d = 2
            self.nbf_total = nbf_total = self.nbf_1d**self.nsd
            self.h = 1. / self.nelem
            
            bf_1d = lambda x: np.array([0.5*(1.-x), 0.5*(1.+x)])
            bf_1d_der = lambda x: np.array([0.5*(0.-1.), 0.5*(0.+1.)])
            bf_1d_der2 = lambda x: np.array([0.0, 0.0])

        elif self.fem_basis_deg == 2:
            assert (self.output_size- 1)%2 == 0
            self.nelem = nelem = int((self.output_size - 1)/2)
            self.nbf_1d = nbf_1d = 3
            self.nbf_total = nbf_total = self.nbf_1d**self.nsd
            self.h = 1. / self.nelem

            bf_1d = lambda x: np.array([
                                        0.5 * x * (x-1.),
                                        (1. - x**2),
                                        0.5 * x * (x+1.)
                                        ], dtype=np.float32)
            bf_1d_der = lambda x: np.array([
                                        0.5 * (2.*x-1.),
                                        (- 2.*x),
                                        0.5 * (2.*x+1.)
                                        ], dtype=np.float32)
            bf_1d_der2 = lambda x: np.array([
                                        0.5 * (2.),
                                        (- 2.),
                                        0.5 * (2.)
                                        ], dtype=np.float32)
        
        elif self.fem_basis_deg == 3:
            assert (self.output_size- 1)%3 == 0
            self.nelem = nelem = int((self.output_size - 1)/3)
            self.nbf_1d = nbf_1d = 4
            self.nbf_total = nbf_total = self.nbf_1d**self.nsd
            self.h = 1. / self.nelem

            bf_1d = lambda x: np.array([
                                        (-9. / 16.) * (x**3- x**2 - (1. / 9.) * x + (1. / 9.)),
                                        (27. / 16.) * (x**3 - (1. / 3.) * x**2 - x + (1. / 3.)),
                                        (-27. / 16.) * (x**3 + (1. / 3.) * x**2 - x - (1. / 3.)),
                                        (9. / 16.) * (x**3 + x**2 - (1. / 9.) * x - (1. / 9.))
                                        ], dtype=np.float32)
            bf_1d_der = lambda x: np.array([
                                        (-9. / 16.) * (3 * x**2 - 2 * x - (1. / 9.)),
                                        (27. / 16.) * (3 * x**2 - (2. / 3.) * x - 1),
                                        (-27. / 16.) * (3 * x**2 + (2. / 3.) * x - 1),
                                        (9. / 16.) * (3 * x**2 + 2 * x - (1. / 9.))
                                        ], dtype=np.float32)

            bf_1d_der2 = lambda x: np.array([
                                        (-9. / 16.) * (6. * x - 2.),
                                        (27. / 16.) * (6. * x - (2. / 3.)),
                                        (-27. / 16.) * (6. * x + (2. / 3.)),
                                        (9. / 16.) * (6. * x + 2.)
                                        ], dtype=np.float32)

        if self.args.nsd == 2:
            self.gpw = torch.zeros(ngp_total)
            self.N_gp = nn.ParameterList() 
            self.dN_x_gp = nn.ParameterList()
            self.dN_y_gp = nn.ParameterList()
            self.d2N_x_gp = nn.ParameterList()
            self.d2N_y_gp = nn.ParameterList() 
            self.d2N_xy_gp = nn.ParameterList()
            for jgp in range(ngp_1d):
                for igp in range(ngp_1d):
                    N_gp = torch.zeros((nbf_1d, nbf_1d))
                    dN_x_gp = torch.zeros((nbf_1d, nbf_1d))
                    dN_y_gp = torch.zeros((nbf_1d, nbf_1d))
                    d2N_x_gp = torch.zeros((nbf_1d, nbf_1d))
                    d2N_y_gp = torch.zeros((nbf_1d, nbf_1d))
                    d2N_xy_gp = torch.zeros((nbf_1d, nbf_1d))

                    IGP = ngp_1d * jgp + igp # tensor product id or the linear id of the gauss point
                    self.gpw[IGP] = gpw_1d[igp] * gpw_1d[jgp]
                    for jbf in range(nbf_1d):
                        for ibf in range(nbf_1d):
                            N_gp[ibf,jbf] = bf_1d(gpx_1d[igp])[ibf] * bf_1d(gpx_1d[jgp])[jbf]
                            dN_x_gp[ibf,jbf] = bf_1d_der(gpx_1d[igp])[ibf] * bf_1d(gpx_1d[jgp])[jbf] * (2 / self.h)
                            dN_y_gp[ibf,jbf] = bf_1d(gpx_1d[igp])[ibf] * bf_1d_der(gpx_1d[jgp])[jbf] * (2 / self.h)
                            d2N_x_gp[ibf,jbf] = bf_1d_der2(gpx_1d[igp])[ibf] * bf_1d(gpx_1d[jgp])[jbf] * (2 / self.h)**2
                            d2N_y_gp[ibf,jbf] = bf_1d(gpx_1d[igp])[ibf] * bf_1d_der2(gpx_1d[jgp])[jbf] * (2 / self.h)**2
                            d2N_xy_gp[ibf,jbf] = bf_1d_der(gpx_1d[igp])[ibf] * bf_1d_der(gpx_1d[jgp])[jbf] * (2 / self.h)**2
                self.N_gp.append(nn.Parameter(N_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                self.dN_x_gp.append(nn.Parameter(dN_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                self.dN_y_gp.append(nn.Parameter(dN_y_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                self.d2N_x_gp.append(nn.Parameter(d2N_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                self.d2N_y_gp.append(nn.Parameter(d2N_y_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                self.d2N_xy_gp.append(nn.Parameter(d2N_xy_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))

        elif self.nsd == 3:

            self.gpw = torch.zeros(ngp_total)
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
            for kgp in range(ngp_1d):
                for jgp in range(ngp_1d):
                    for igp in range(ngp_1d):
                        N_gp = torch.zeros((nbf_1d, nbf_1d, nbf_1d))
                        dN_x_gp = torch.zeros((nbf_1d, nbf_1d, nbf_1d))
                        dN_y_gp = torch.zeros((nbf_1d, nbf_1d, nbf_1d))
                        dN_z_gp = torch.zeros((nbf_1d, nbf_1d, nbf_1d))
                        d2N_x_gp = torch.zeros((nbf_1d, nbf_1d, nbf_1d))
                        d2N_y_gp = torch.zeros((nbf_1d, nbf_1d, nbf_1d))
                        d2N_z_gp = torch.zeros((nbf_1d, nbf_1d, nbf_1d))
                        d2N_xy_gp = torch.zeros((nbf_1d, nbf_1d, nbf_1d))
                        d2N_yz_gp = torch.zeros((nbf_1d, nbf_1d, nbf_1d))
                        d2N_zx_gp = torch.zeros((nbf_1d, nbf_1d, nbf_1d))

                        IGP = kgp * ngp_1d**2 + jgp * ngp_1d + igp # tensor product id or the linear id of the gauss point
                        self.gpw[IGP] = gpw_1d[igp] * gpw_1d[jgp]

                        for kbf in range(nbf_1d):
                            for jbf in range(nbf_1d):
                                for ibf in range(nbf_1d):
                                    N_gp[ibf,jbf,kbf] = bf_1d(gpx_1d[igp])[ibf] * bf_1d(gpx_1d[jgp])[jbf] * bf_1d(gpx_1d[kgp])[kbf]
                                    dN_x_gp[ibf,jbf,kbf] = bf_1d_der(gpx_1d[igp])[ibf] * bf_1d(gpx_1d[jgp])[jbf] * bf_1d(gpx_1d[kgp])[kbf] * (2 / self.h)
                                    dN_y_gp[ibf,jbf,kbf] = bf_1d(gpx_1d[igp])[ibf] * bf_1d_der(gpx_1d[jgp])[jbf] * bf_1d(gpx_1d[kgp])[kbf] * (2 / self.h)
                                    dN_z_gp[ibf,jbf,kbf] = bf_1d(gpx_1d[igp])[ibf] * bf_1d(gpx_1d[jgp])[jbf] * bf_1d_der(gpx_1d[kgp])[kbf] * (2 / self.h)
                                    d2N_x_gp[ibf,jbf,kbf] = bf_1d_der2(gpx_1d[igp])[ibf] * bf_1d(gpx_1d[jgp])[jbf] * bf_1d(gpx_1d[kgp])[kbf] * (2 / self.h)**2
                                    d2N_y_gp[ibf,jbf,kbf] = bf_1d(gpx_1d[igp])[ibf] * bf_1d_der2(gpx_1d[jgp])[jbf] * bf_1d(gpx_1d[kgp])[kbf] * (2 / self.h)**2
                                    d2N_z_gp[ibf,jbf,kbf] = bf_1d(gpx_1d[igp])[ibf] * bf_1d(gpx_1d[jgp])[jbf] * bf_1d_der2(gpx_1d[kgp])[kbf] * (2 / self.h)**2
                                    d2N_xy_gp[ibf,jbf,kbf] = bf_1d_der(gpx_1d[igp])[ibf] * bf_1d_der(gpx_1d[jgp])[jbf] * bf_1d(gpx_1d[kgp])[kbf] * (2 / self.h)**2
                                    d2N_yz_gp[ibf,jbf,kbf] = bf_1d(gpx_1d[igp])[ibf] * bf_1d_der(gpx_1d[jgp])[jbf] * bf_1d_der(gpx_1d[kgp])[kbf] * (2 / self.h)**2
                                    d2N_zx_gp[ibf,jbf,kbf] = bf_1d_der(gpx_1d[igp])[ibf] * bf_1d(gpx_1d[jgp])[jbf] * bf_1d_der(gpx_1d[kgp])[kbf] * (2 / self.h)**2

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

    def gauss_pt_evaluations(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.N_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluations_der_x(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.dN_x_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluations_der_y(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.dN_y_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluations_der_z(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.dN_z_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluations_der2_x(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_x_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluations_der2_y(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_y_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluations_der2_z(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_z_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluations_der2_xy(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_xy_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluations_der2_yz(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_yz_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluations_der2_zx(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_zx_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def loss(self):
        raise NotImplementedError

    def forward(self, batch):
        nu_4d_tensor, forcing_4d_tensor = batch
        nu_4d_tensor = nu_4d_tensor.type_as(next(self.network.parameters()))
        forcing_4d_tensor = forcing_4d_tensor.type_as(next(self.network.parameters()))
        u = self.network(nu_4d_tensor) 
        return u, nu_4d_tensor, forcing_4d_tensor

    def training_step(self, batch, batch_idx):
        u, nu_4d_tensor, forcing_4d_tensor = self.forward(batch)
        loss_val = self.loss(u, nu_4d_tensor, forcing_4d_tensor).mean()
        self.log('PDE_loss', loss_val.item())
        self.log('loss', loss_val.item())
        return loss_val

    def train_dataloader(self):
        """
        The data returned by DataLoader is on the same device that PL is using for network parameter
        """
        return DataLoader(self.dataset, batch_size=self.args.optimization.batch_size)

    def configure_optimizers(self):
        lr = self.args.optimization.learning_rate
        opts = [torch.optim.Adam(self.network.parameters(), lr=lr)]
        return opts, []

    def on_fit_start(self):
        """
        This is a PL callback that is called when fit function starts
        We need self.x,y,z to be on gpu since they are used in gen_input_calc KL construction
        where they are multiplied to data from gpu (rand_tensor_list)
        So they have to be moved to gpu
        """
        self.x = self.x.type_as(next(self.network.parameters()))
        self.y = self.y.type_as(next(self.network.parameters()))
        self.z = self.z.type_as(next(self.network.parameters()))

    def on_epoch_end(self):
        if self.args.nsd == 2:
            fig, axs = plt.subplots(len(self.vis_sample_ids), 4, figsize=(2*4,1.2*len(self.vis_sample_ids)),
                                subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
            fig.suptitle('Contour Plots')
            for ax_row in axs:
                for ax in ax_row:
                    ax.set_xticks([])
                    ax.set_yticks([])
            self.network.eval()
            for idx in range(len(self.vis_sample_ids)):
                grid, inputs, forcing = self.dataset[self.vis_sample_ids[idx]]

                # coeff is on cpu (because it is directly indexed from dataset)
                # so it has to be moved to the computation device (gpu/cpu) using type_as
                coeff = (torch.tensor(np.expand_dims(coeff, axis=0))).type_as(next(self.network.parameters()))
                u_gen, gen_input, nu_4d_tensor, forcing_4d_tensor = self.forward(coeff)

                u_gen = self.apply_padding_on(u_gen)
                k = nu_4d_tensor.squeeze().detach().cpu()
                u = u_gen.squeeze().detach().cpu()

                im0 = axs[idx][0].imshow(k,cmap='jet')
                fig.colorbar(im0, ax=axs[idx, 0])
                im1 = axs[idx][1].imshow(u,cmap='jet')
                fig.colorbar(im1, ax=axs[idx, 1])            
                if self.manufactured:
                    u_exact = self.calc_exact_solution(coeff).squeeze().detach().cpu()
                    print("u_exact.shape = ", u_exact.shape, "u.shape ", u.shape)
                    diff = u - u_exact
                    np.set_printoptions(precision=3)
                    print(np.around(np.linalg.norm(diff.flatten())/self.args.output_size, 4))
                    self.log('l2_norm w/ exact', np.around(np.linalg.norm(diff.flatten())/self.args.output_size, 4))

                    im2 = axs[idx][2].imshow(u_exact,cmap='jet')
                    fig.colorbar(im2, ax=axs[idx, 2])            
                    im3 = axs[idx][3].imshow(diff,cmap='jet')
                    fig.colorbar(im3, ax=axs[idx, 3])
        elif self.args.nsd == 3:
            fig, axs = plt.subplots(len(self.vis_sample_ids), 6, figsize=(2*4,1.2*len(self.vis_sample_ids)),
                                subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
            fig.suptitle('Contour Plots')
            for ax_row in axs:
                for ax in ax_row:
                    ax.set_xticks([])
                    ax.set_yticks([])
            self.network.eval()
            for idx in range(len(self.vis_sample_ids)):
                coeff = self.dataset[self.vis_sample_ids[idx]]

                # coeff is on cpu (because it is directly indexed from dataset)
                # so it has to be moved to the computation device (gpu/cpu) using type_as
                coeff = (torch.tensor(np.expand_dims(coeff, axis=0))).type_as(next(self.network.parameters()))
                u_gen, gen_input, nu_4d_tensor, forcing_4d_tensor = self.forward(coeff)

                u_gen = self.apply_padding_on(u_gen)
                k = nu_4d_tensor.squeeze().detach().cpu()
                u = u_gen.squeeze().detach().cpu()

                im = axs[idx][0].imshow(k[16,:,:],cmap='jet')
                fig.colorbar(im, ax=axs[idx, 0])
                im = axs[idx][1].imshow(k[32,:,:],cmap='jet')
                fig.colorbar(im, ax=axs[idx, 1])
                im = axs[idx][2].imshow(k[48,:,:],cmap='jet')
                fig.colorbar(im, ax=axs[idx, 2])
                im = axs[idx][3].imshow(u[16,:,:],cmap='jet')
                fig.colorbar(im, ax=axs[idx, 3])
                im = axs[idx][4].imshow(u[32,:,:],cmap='jet')
                fig.colorbar(im, ax=axs[idx, 4])
                im = axs[idx][5].imshow(u[48,:,:],cmap='jet')
                fig.colorbar(im, ax=axs[idx, 5])

        # logger[0] is the Tensorboard logger
        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')
