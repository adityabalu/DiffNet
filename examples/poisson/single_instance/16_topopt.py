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
from DiffNet.DiffNetFEM import DiffNet2DFEM
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
        self.n_samples = 100
        

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.bc1, self.bc2])
        forcing = np.zeros_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)



class Poisson(DiffNet2DFEM):
    """docstring for Poisson"""
    def __init__(self, network, dataset, **kwargs):
        super(Poisson, self).__init__(network, dataset, **kwargs)
        self.target_vf = 0.2

    def loss(self, network_inp, inputs_tensor, forcing_tensor):

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = torch.nn.functional.sigmoid(network_inp[1].type_as(f))
        bc1 = inputs_tensor[:,0:1,:,:]
        bc2 = inputs_tensor[:,1:2,:,:]

        # apply boundary conditions
        u = network_inp[0].type_as(f)
        u = nu*u
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)


        nu_gp = self.gauss_pt_evaluation(nu)
        f_gp = self.gauss_pt_evaluation(f)
        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(nu_gp)
        res_elmwise = 0.5 * transformation_jacobian * (nu_gp * (u_x_gp**2 + u_y_gp**2) - (u_gp * f_gp))
        res_elmwise = torch.sum(res_elmwise, 1) 

        loss = torch.mean(res_elmwise)
        return loss

    def compliance(self, network_inp, inputs_tensor, forcing_tensor):

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = torch.nn.functional.sigmoid(network_inp[1].type_as(f))
        bc1 = inputs_tensor[:,0:1,:,:]
        bc2 = inputs_tensor[:,1:2,:,:]

        # apply boundary conditions
        u = network_inp[0].type_as(f)
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        nu_gp = self.gauss_pt_evaluation(nu)
        f_gp = self.gauss_pt_evaluation(f)
        u_gp = self.gauss_pt_evaluation(u)

        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(nu_gp)
        res_elmwise = 0.5 * transformation_jacobian * (nu_gp * u_gp**2 - (u_gp * f_gp))
        res_elmwise = torch.sum(res_elmwise, 1) 

        loss = torch.mean(res_elmwise)
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        return [self.network[0][0], self.network[1][0]], inputs_tensor, forcing_tensor

    def configure_optimizers(self):
        """
        Configure optimizer for network parameters
        """
        lr = self.learning_rate
        opts = [torch.optim.Adam(self.network[0], lr=lr), torch.optim.Adam(self.network[1], lr=lr*5), torch.optim.Adam(self.network[1], lr=lr*10)]
        return opts, []

    def training_step(self, batch, batch_idx, optimizer_idx):
    # def training_step(self, batch, batch_idx):
        network_out, inputs_tensor, forcing_tensor = self.forward(batch)
        if (self.current_epoch%100) < 50:
            loss_val = self.loss(network_out, inputs_tensor, forcing_tensor).mean()
            if optimizer_idx is 0:
                self.log('PDE_loss', loss_val.item())
            else:
                loss_val = loss_val*0.0                    
        else:
            if optimizer_idx is 0:
                loss_val = self.loss(network_out, inputs_tensor, forcing_tensor).mean()
                self.log('PDE_loss', loss_val.item())
            elif optimizer_idx is 2:
                loss_val = self.compliance(network_out, inputs_tensor, forcing_tensor).mean()
                self.log('Compliance', loss_val.item())
            else:
                loss_val = (network_out[1].mean() - self.target_vf)**2
                self.log('VF_constraint', loss_val.item())

        return {"loss": loss_val}

    # def optimizer_closure():
    #     loss = self.training_step(batch, batch_idx, optimizer_idx)
    #     opt.zero_grad()
    #     loss.backward()
    #     return loss

    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu=False,
    #     using_native_amp=False,
    #     using_lbfgs=False,
    # ):

    #     # update generator every step
    #     if optimizer_idx == 0:
    #         optimizer.step(closure=optimizer_closure)

    #     # update discriminator every 2 steps
    #     if optimizer_idx == 1:
    #         if (batch_idx + 1) % 2 == 0:
    #             # the closure (which includes the `training_step`) will be executed by `optimizer.step`
    #             optimizer.step(closure=optimizer_closure)
    #         else:
    #             # call the closure by itself to run `training_step` + `backward` without an optimizer step
    #             optimizer_closure()

    def on_epoch_end(self):
        fig, axs = plt.subplots(1, 2, figsize=(2*2,1.2),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        inputs, forcing = self.dataset[0]

        network_inp, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network[0].parameters())), forcing.unsqueeze(0).type_as(next(self.network[0].parameters()))))

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = torch.nn.functional.sigmoid(network_inp[1].type_as(f))
        bc1 = inputs_tensor[:,0:1,:,:]
        bc2 = inputs_tensor[:,1:2,:,:]

        # apply boundary conditions
        u = network_inp[0].type_as(f)
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        k = nu.squeeze().detach().cpu()
        u = u.squeeze().detach().cpu()

        im0 = axs[0].imshow(k,cmap='jet')
        fig.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(u,cmap='jet')
        fig.colorbar(im1, ax=axs[1])  
        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    domain_size = 64
    u_tensor = np.random.randn(1,1,domain_size,domain_size)
    nu_tensor = np.random.randn(1,1,domain_size,domain_size)
    network1 = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    network2 = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(nu_tensor), requires_grad=True)])
    dataset = Rectangle(domain_size=domain_size)
    basecase = Poisson([network1, network2], dataset, batch_size=1)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="topopt")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='min', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=1000, deterministic=True, profiler='simple')

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