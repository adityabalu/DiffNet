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
from DiffNet.datasets.single_instances.images import Disk
from torch.utils.data import DataLoader

class Poisson(DiffNet2DFEM):
    """docstring for Poisson"""
    def __init__(self, network, dataset, **kwargs):
        super(Poisson, self).__init__(network, **kwargs)
        print("What is my domain size = ", self.domain_size)
        x = np.linspace(0,1,self.domain_size)
        y = np.linspace(0,1,self.domain_size)
        xx, yy = np.meshgrid(x,y)
        
        inputs = dataset[0][0]
        self.domain_mask = inputs[0,:,:]
        self.u_exact = self.exact_solution(torch.FloatTensor(xx), torch.FloatTensor(yy))
    
    def exact_solution(self, x,y):
        R = 0.25
        Z = 0.25*(R**2 - ((x-0.5)**2+(y-0.5)**2))
        Z = torch.where(Z>0, Z, torch.zeros_like(Z))
        return Z

    def loss(self, u, inputs_tensor, forcing_tensor):

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,u*0.0,u)
        # f = torch.where(bc1>0.5,f*0.0,u)
        # u = torch.where(bc2>0.5,u*0.0,u)


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

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        return self.network[0], inputs_tensor, forcing_tensor

    def training_step(self, batch, batch_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
        return {"loss": loss_val}

    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs["loss"]
        self.log('PDE_loss', loss.item())
        self.log('loss', loss.item())
        return training_step_outputs

    def configure_optimizers(self):
        lr = self.learning_rate
        opts = [torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        # opts = [torch.optim.Adam(self.network, lr=lr)]
        return opts, []

    # def on_epoch_end(self):
    #     self.network.eval()
    #     inputs, forcing = self.dataset[0]
    #     nu, f, u = self.do_query(inputs, forcing)
    #     self.plot_contours(nu, f, u)

    # def do_query(self, inputs, forcing):
    #     u, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

    #     f = forcing_tensor # renaming variable
        
    #     # extract diffusivity and boundary conditions here
    #     nu = inputs_tensor[:,0:1,:,:]
    #     bc1 = inputs_tensor[:,1:2,:,:]
    #     bc2 = inputs_tensor[:,2:3,:,:]

    #     # apply boundary conditions
    #     u = torch.where(bc1>0.5,1.0+u*0.0,u)
    #     u = torch.where(bc2>0.5,u*0.0,u)

    #     nu = nu.squeeze().detach().cpu()
    #     u = u.squeeze().detach().cpu()

    #     return nu, f, u

    # def plot_contours(self,nu,f,u):
    #     fig, axs = plt.subplots(1, 2, figsize=(2*2,1.2),
    #                         subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
    #     for ax in axs:
    #         ax.set_xticks([])
    #         ax.set_yticks([])

    #     im0 = axs[0].imshow(nu,cmap='jet')
    #     fig.colorbar(im0, ax=axs[0])
    #     im1 = axs[1].imshow(u,cmap='jet')
    #     fig.colorbar(im1, ax=axs[1])  
    #     plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
    #     self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
    #     plt.close('all')

class MyPrintingCallback(pl.callbacks.Callback):
    # def on_train_start(self, trainer, pl_module):
    #     print("Training is starting")
    # def on_train_end(self, trainer, pl_module):
    #     print("Training is ending")
    # def on_validation_epoch_end(self, trainer, pl_module):
    #     print("On validation epoch end")

    def __init__(self, **kwargs):
        super(MyPrintingCallback, self).__init__(**kwargs)
        self.num_query = 2

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.network.eval()
        inputs, forcing = trainer.train_dataloader.dataset[0:self.num_query]
        nu, f, u = self.do_query(trainer, pl_module, inputs, forcing)
        self.plot_contours(trainer, pl_module, nu, f, u)

    def do_query(self, trainer, pl_module, inputs, forcing):
        forcing = forcing.repeat(self.num_query,1,1,1)
        # print("\ninference for: ", trainer.train_dataloader.dataset.coeffs[0:num_query])

        u, inputs_tensor, forcing_tensor = pl_module.forward((inputs.type_as(next(pl_module.network.parameters())), forcing.type_as(next(pl_module.network.parameters()))))

        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[0:1,:,:]
        bc1 = inputs_tensor[1:2,:,:]
        bc2 = inputs_tensor[2:3,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,u*0.0,u)
        # u = torch.where(bc2>0.5,u*0.0,u)

        # loss = pl_module.loss(u, inputs_tensor, forcing_tensor[:,0:1,:,:])
        # print("loss incurred for this coeff:", loss)

        nu = nu.squeeze().detach().cpu()
        f = forcing_tensor.squeeze().detach().cpu()
        u = u.squeeze().detach().cpu()

        return  nu, f, u

    def plot_contours(self,trainer,pl_module,nu,f,u):
        fig, axs = plt.subplots(1, 4, figsize=(4*2,1.2),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        im0 = axs[0].imshow(nu,cmap='jet')
        fig.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(u,cmap='jet')
        fig.colorbar(im1, ax=axs[1])  
        # uexact = np.where(nu>0, pl_module.u_exact, np.zeros_like(nu))
        im2 = axs[2].imshow(pl_module.u_exact,cmap='jet')
        fig.colorbar(im2, ax=axs[2])  
        im3 = axs[3].imshow(u - pl_module.u_exact,cmap='jet')
        fig.colorbar(im3, ax=axs[3])  
        plt.savefig(os.path.join(trainer.logger.log_dir, 'contour_' + str(trainer.current_epoch) + '.png'))
        trainer.logger.experiment.add_figure('Contour Plots', fig, trainer.current_epoch)
        plt.close('all')

def main():
    # filename = 'img-0.png'
    domain_size = 64
    filename = '../disks-convergence/n_{}.png'.format(domain_size)
    batch_size = 1
    dataset = Disk(filename, domain_size=domain_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    u_tensor = np.ones_like(dataset.domain)
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    basecase = Poisson(network, dataset, batch_size=batch_size, domain_size=domain_size)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="disk")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)
    printsave = MyPrintingCallback()

    # trainer = Trainer(gpus=[0],callbacks=[early_stopping,printsave],
    #     checkpoint_callback=checkpoint, logger=[logger,csv_logger],
    #     max_epochs=5, deterministic=True, profiler='simple')
    trainer = pl.Trainer(accelerator='gpu',devices=1,
                     callbacks=[checkpoint,printsave],
                     logger=[logger,csv_logger], 
                     max_epochs=150,
                     fast_dev_run=False
                     )

    # ------------------------
    # 4 Training
    # ------------------------

    trainer.fit(basecase, dataloader)

    # ------------------------
    # 5 SAVE NETWORK
    # ------------------------
    torch.save(basecase.network, os.path.join(trainer.logger.log_dir, 'network.pt'))

    # L2 error calculation
    print("Calculating L2 error:")
    basecase.network.eval()
    inputs, forcing = trainer.train_dataloader.dataset[0]
    nu, f, u = printsave.do_query(trainer, basecase, inputs, forcing)
    basecase.calc_l2_err(u.detach().unsqueeze(0).unsqueeze(0))


if __name__ == '__main__':
    main()
