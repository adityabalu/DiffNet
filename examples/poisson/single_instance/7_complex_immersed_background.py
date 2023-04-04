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
from DiffNet.datasets.single_instances.images import ImageIMBack
import time


class Poisson(DiffNet2DFEM):
    """docstring for Poisson"""
    def __init__(self, network, dataset, **kwargs):
        super(Poisson, self).__init__(network, dataset, **kwargs)
        self.loss_type = kwargs.get('loss_type', 'energy') # 'energy' and 'resmin'

    def Q1_vector_assembly(self, Aglobal, Aloc_all):
        Aglobal[:,0, 0:-1, 0:-1] += Aloc_all[:,0, :, :]
        Aglobal[:,0, 0:-1, 1:  ] += Aloc_all[:,1, :, :]
        Aglobal[:,0, 1:  , 0:-1] += Aloc_all[:,2, :, :]
        Aglobal[:,0, 1:  , 1:  ] += Aloc_all[:,3, :, :]
        return Aglobal

    def loss(self, u, inputs_tensor, forcing_tensor):

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)


        nu_gp = self.gauss_pt_evaluation(nu)
        f_gp = self.gauss_pt_evaluation(f)
        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(nu_gp)
        res_elmwise = transformation_jacobian * (0.5*nu_gp * (u_x_gp**2 + u_y_gp**2) - (u_gp * f_gp))
        res_elmwise = torch.sum(res_elmwise, 1) 

        loss = torch.mean(res_elmwise)
        return loss

    def lossResMin(self, u, inputs_tensor, forcing_tensor):
        N_values = self.Nvalues.type_as(u)
        dN_x_values = self.dN_x_values.type_as(u)
        dN_y_values = self.dN_y_values.type_as(u)
        gpw = self.gpw.type_as(u)

        # extract diffusivity and boundary conditions here
        f = forcing_tensor # renaming variable
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # DERIVE NECESSARY VALUES
        trnsfrm_jac = (0.5*self.h)**2
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        # apply boundary conditions
        # NOTE: we do add the BC to the residual later, but adding the residual to u is also very important
        #       because ideally we want to calculate the values of A*u when A is BC adjusted. But since we
        #       are not altering the convolution kernel "Kmatrices" (i.e., effectively the values of A), thus
        #       we will end up with bad values in R at the interior points
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        # self.plot_contours(nu.squeeze().detach().cpu(), f.squeeze().detach().cpu(), u.squeeze().detach().cpu())
        # exit()

        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)
        f_gp = self.gauss_pt_evaluation(f)

        # CALCULATION STARTS
        # lhs
        vxux = dN_x_values*u_x_gp*JxW
        vyuy = dN_y_values*u_y_gp*JxW
        # rhs
        vf = N_values*f_gp*JxW

        # residual
        temp = vxux+vyuy-vf

        # unassembled residual
        R_split = torch.sum(temp*JxW, 2) # sum across all GP

        # assembly
        R = torch.zeros_like(u); R = self.Q1_vector_assembly(R, R_split)
        # add boundary conditions to R <---- this step is very important
        R = torch.where(bc1>0.5,R*0.0,R)
        R = torch.where(bc2>0.5,R*0.0,R)

        # loss = torch.sum(R**2)
        loss = torch.norm(R, 'fro')
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        return self.network[0], inputs_tensor, forcing_tensor

    def training_step(self, batch, batch_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        if self.loss_type == 'energy':
            loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
        elif self.loss_type == 'resmin':
            loss_val = self.lossResMin(u, inputs_tensor, forcing_tensor).mean()
        return {"loss": loss_val}

    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs["loss"]
        self.log('PDE_loss', loss.item())
        self.log('loss', loss.item())
        return training_step_outputs

    def configure_optimizers(self):
        lr = self.learning_rate
        # opts = [torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        opts = [torch.optim.Adam(self.network, lr=lr)]
        return opts, []

    def on_epoch_end(self):
        self.network.eval()
        inputs, forcing = self.dataset[0]
        if self.current_epoch % 100 == 0:
            nu, f, u = self.do_query(inputs, forcing)
            self.plot_contours(nu, f, u)

    def do_query(self, inputs, forcing):
        u, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        nu = nu.squeeze().detach().cpu()
        u = u.squeeze().detach().cpu()

        return nu, f, u

    def plot_contours(self,nu,f,u):
        fig, axs = plt.subplots(1, 2, figsize=(2*2,1.2),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        im0 = axs[0].imshow(1-nu,cmap='gray')
        fig.colorbar(im0, ax=axs[0], ticks=[0, 1])
        im1 = axs[1].imshow(u,cmap='jet')
        fig.colorbar(im1, ax=axs[1])  
        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

        np.save(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.npy'), u)

def main():
    domain_size = 257
    max_epochs = int(1e6)
    LR=1e-2
    loss_type = 'resmin' # 'energy'
    filename = '../selected-af/a18.png'
    dataset = ImageIMBack(filename, domain_size=domain_size)
    u_tensor = np.random.rand(domain_size, domain_size) # np.ones_like(dataset.domain) # 
    # u_tensor = np.load('/work/baskarg/bkhara/diffnet/examples/poisson/single_instance/complex_immersed_background/version_63/contour_13.npy')
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    basecase = Poisson(network, dataset, batch_size=1, domain_size=domain_size, learning_rate=LR, loss_type=loss_type)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="complex_immersed_background")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=max_epochs, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=max_epochs, deterministic=True, profiler='simple')

    # ------------------------
    # 4 Training
    # ------------------------

    t0 = time.time()
    trainer.fit(basecase)
    t1 = time.time()
    print("Time for training = ", t1-t0)

    # ------------------------
    # 5 SAVE NETWORK
    # ------------------------
    torch.save(basecase.network, os.path.join(logger.log_dir, 'network.pt'))


if __name__ == '__main__':
    main()