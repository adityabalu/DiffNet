import os
import sys
import json
import torch
import numpy as np

import matplotlib
# matplotlib.use("pgf")
matplotlib.rcParams.update({
    # 'font.family': 'serif',
    'font.size':12,
})
from matplotlib import pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
seed_everything(42)

import DiffNet
from DiffNet.networks.wgan import GoodNetwork
from DiffNet.networks.autoencoders import AE
from DiffNet.DiffNetFEM import DiffNet2DFEM
# from DiffNet.datasets.single_instances.klsum import Dataset
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
        self.nu = generate_diffusivity_tensor(self.coeff, output_size=self.domain_size).squeeze()
        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        self.bc1[:,0] = 1
        self.bc1[:,-1] = 1
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        self.bc2[:,-1] = 1
        self.n_samples = 1000

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.nu, self.bc1, self.bc2])
        forcing = np.zeros_like(self.nu)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

class Poisson(DiffNet2DFEM):
    """docstring for Poisson"""
    def __init__(self, network, dataset, **kwargs):
        super(Poisson, self).__init__(network, dataset, **kwargs)
        self.mapping_type = kwargs.get('mapping_type', 'no_network')
        self.loss_type = kwargs.get('loss_type', 'energy')

        u_bc = np.zeros_like(self.dataset.bc1); u_bc[:,0] = 1.; u_bc[:,-1] = 0.
        self.u_bc = torch.FloatTensor(u_bc)

    def Q1_vector_assembly(self, Aglobal, Aloc_all):
        Aglobal[:,0, 0:-1, 0:-1] += Aloc_all[:,0, :, :]
        Aglobal[:,0, 0:-1, 1:  ] += Aloc_all[:,1, :, :]
        Aglobal[:,0, 1:  , 0:-1] += Aloc_all[:,2, :, :]
        Aglobal[:,0, 1:  , 1:  ] += Aloc_all[:,3, :, :]
        return Aglobal

    def loss_EnergyMin(self, u, inputs_tensor, forcing_tensor):

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        # u = torch.where(bc1>0.5,1.0+u*0.0,u)
        # u = torch.where(bc2>0.5,u*0.0,u)
        u_bc = self.u_bc.type_as(u)
        u = torch.where(bc1>0.5,u_bc,u)

        nu_gp = self.gauss_pt_evaluation(nu)
        f_gp = self.gauss_pt_evaluation(f)
        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(nu_gp)
        res_elmwise = transformation_jacobian * (nu_gp * (u_x_gp**2 + u_y_gp**2) - (u_gp * f_gp))
        res_elmwise = torch.sum(res_elmwise, 1) 

        loss = torch.mean(res_elmwise)
        return loss

    def loss_ResMin(self, u, inputs_tensor, forcing_tensor):
        f = forcing_tensor # renaming variable

        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        u# u = torch.where(bc1>0.5,1.0+u*0.0,u)
        # u = torch.where(bc2>0.5,u*0.0,u)
        u_bc = self.u_bc.type_as(u)
        u = torch.where(bc1>0.5,u_bc,u)

        nu_gp = self.gauss_pt_evaluation(nu)
        f_gp = self.gauss_pt_evaluation(f)
        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        # transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(nu_gp)
        # res_elmwise = transformation_jacobian * (nu_gp * (u_x_gp**2 + u_y_gp**2) - (u_gp * f_gp))
        # res_elmwise = torch.sum(res_elmwise, 1)

        # loss = torch.mean(res_elmwise)
        # return loss

        hx = self.hx
        hy = self.hy

        N_values = self.Nvalues.type_as(u)
        dN_x_values = self.dN_x_values.type_as(u)
        dN_y_values = self.dN_y_values.type_as(u)
        gpw = self.gpw.type_as(u)

        # DERIVE NECESSARY VALUES
        trnsfrm_jac = (0.5*hx)*(0.5*hy)
        # JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        u_x = self.gauss_pt_evaluation_der_x(u)
        u_y = self.gauss_pt_evaluation_der_y(u)
        Wx_U1x = dN_x_values*u_x
        Wy_U1y = dN_y_values*u_y

        # integrated values on lhs & rhs
        temp1 = nu_gp*(Wx_U1x+Wy_U1y) # - Wx_P # - W_F1
        # unassembled residual
        R_split_1 = torch.sum(temp1*JxW, 2) # sum across all GP

        # assembly
        R1 = torch.zeros_like(u); R1 = self.Q1_vector_assembly(R1, R_split_1)

        # add boundary conditions to R <---- this step is very important
        u = torch.where(bc1>0.5,u_bc,R1)

        # return R1
        return torch.norm(R1, 'fro')

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        if self.mapping_type == 'network':
            return self.network(inputs_tensor[:,0:1,:,:]), inputs_tensor, forcing_tensor
        elif self.mapping_type == 'no_network':
            return self.network[0], inputs_tensor, forcing_tensor

    # def configure_optimizers(self):
    #     """
    #     Configure optimizer for network parameters
    #     """
    #     lr = self.learning_rate
    #     opts = [torch.optim.LBFGS(self.network.parameters(), lr=1.0, max_iter=5)]
    #     return opts, []

    def training_step(self, batch, batch_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        if self.loss_type == 'energy':
            loss_val = self.loss_EnergyMin(u, inputs_tensor, forcing_tensor).mean()
        elif self.loss_type == 'residual':
            loss_val = self.loss_ResMin(u, inputs_tensor, forcing_tensor).mean()
        self.log('PDE_loss', loss_val.item())
        self.log('loss', loss_val.item())
        return loss_val

    def on_epoch_end(self):
        fig, axs = plt.subplots(1, 2, figsize=(2*2,1.2),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        self.network.eval()
        inputs, forcing = self.dataset[0]

        u, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        k = nu.squeeze().detach().cpu()
        u = u.squeeze().detach().cpu()

        im0 = axs[0].imshow(k,cmap='jet')
        fig.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(u,cmap='jet')
        fig.colorbar(im1, ax=axs[1]) 
        fig.suptitle("({}, {}), LR = {:.1e}, mapping_type = {}, loss_type = {}".format(self.domain_sizeX, self.domain_sizeY, self.learning_rate, self.mapping_type, self.loss_type), fontsize=8) 
        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    LR=1e-2
    Nx=32
    Ny=32
    # loss_type = 'energy'
    loss_type = 'residual'
    # mapping_type = 'no_network'
    mapping_type = 'network'
    u_tensor = np.ones((1,1,Ny, Nx))
    if mapping_type == 'no_network':
        network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    elif mapping_type == 'network':
        network = AE(in_channels=1, out_channels=1, dims=Nx, n_downsample=3)
    # network = GoodNetwork(in_channels=3, out_channels=1, in_dim=64, out_dim=64)
    dataset = Dataset('example-coefficients.txt', domain_size=Nx)
    basecase = Poisson(network, dataset, batch_size=1, learning_rate=LR, mapping_type=mapping_type, loss_type=loss_type)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="klsum_network")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='min', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=50, deterministic=True, profiler="simple")

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