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
from DiffNet.networks.wgan_old import GoodGenerator
from DiffNet.networks.autoencoders import AE
from DiffNet.DiffNetFEM import DiffNet2DFEM
from DiffNet.datasets.parametric.klsum import KLSumStochastic
from DiffNet.datasets.single_instances.klsum import Dataset
from torch.utils.data import DataLoader


class Poisson(DiffNet2DFEM):
    """docstring for Poisson"""
    def __init__(self, network, **kwargs):
        super(Poisson, self).__init__(network, **kwargs)

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
        res_elmwise = transformation_jacobian * (nu_gp * (u_x_gp**2 + u_y_gp**2) - (u_gp * f_gp))
        res_elmwise = torch.sum(res_elmwise, 1) 

        # transformation_jacobian = (0.5 * self.h)**2 * self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(nu_gp)
        # res_elmwise = 0.5 * transformation_jacobian * (nu_gp * (u_x_gp**2 + u_y_gp**2) - (u_gp * f_gp))
        # res_elmwise = torch.sum(res_elmwise, 1) 

        loss = torch.mean(res_elmwise)
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        u = self.network(inputs_tensor[:,0:1,:,:])
        return u, inputs_tensor, forcing_tensor

    def training_step(self, batch, batch_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
        self.log("loss", loss_val)
        # return {"loss": loss_val}
        return loss_val

    # def training_step_end(self, training_step_outputs):
    #     loss = training_step_outputs #["loss"]
    #     self.log('PDE_loss', loss.item())
    #     self.log('loss', loss.item())
    #     return training_step_outputs

    def configure_optimizers(self):
        lr = self.learning_rate
        # opts = [torch.optim.LBFGS(self.network.parameters(), lr=lr, max_iter=5)]
        opts = [torch.optim.Adam(self.network.parameters(), lr=lr)]
        # schd = []
        schd = [torch.optim.lr_scheduler.MultiStepLR(opts[0], milestones=[10,15,30], gamma=0.1)]
        return opts, schd

class MyPrintingCallback(pl.callbacks.Callback):
    # def on_train_start(self, trainer, pl_module):
    #     print("Training is starting")
    # def on_train_end(self, trainer, pl_module):
    #     print("Training is ending")
    # def on_validation_epoch_end(self, trainer, pl_module):
    #     print("On validation epoch end")

    def __init__(self, **kwargs):
        super(MyPrintingCallback, self).__init__(**kwargs)
        self.num_query = 6

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.network.eval()
        nu, f, u = self.do_query(trainer, pl_module)
        self.plot_contours(trainer, pl_module, nu, f, u)

    def do_query(self, trainer, pl_module):
        inputs, forcing = trainer.train_dataloader.dataset[0:self.num_query]
        forcing = forcing.repeat(self.num_query,1,1,1)
        # print("\ninference for: ", trainer.train_dataloader.dataset.coeffs[0:num_query])

        u, inputs_tensor, forcing_tensor = pl_module.forward((inputs.type_as(next(pl_module.network.parameters())), forcing.type_as(next(pl_module.network.parameters()))))

        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        loss = pl_module.loss(u, inputs_tensor, forcing_tensor[:,0:1,:,:])
        print("loss incurred for this coeff:", loss)

        nu = nu.squeeze().detach().cpu()
        f = forcing_tensor.squeeze().detach().cpu()
        u = u.squeeze().detach().cpu()

        return  nu, f, u

    def plot_contours(self,trainer,pl_module,nu,f,u):
        plt_num_row = self.num_query
        plt_num_col = 2
        fig, axs = plt.subplots(plt_num_row, plt_num_col, figsize=(2*plt_num_col,1.2*plt_num_row),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax_row in axs:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])

        for idx in range(self.num_query):

            # extract diffusivity and boundary conditions here
            ki = nu[idx,:,:]
            ui = u[idx,:,:]

            im0 = axs[idx][0].imshow(ki,cmap='jet')
            fig.colorbar(im0, ax=axs[idx,0])
            im1 = axs[idx][1].imshow(ui,cmap='jet')
            fig.colorbar(im1, ax=axs[idx,1])  
        plt.savefig(os.path.join(trainer.logger.log_dir, 'contour_' + str(trainer.current_epoch) + '.png'))
        trainer.logger.experiment.add_figure('Contour Plots', fig, trainer.current_epoch)
        plt.close('all')
       


def main():
    kl_terms = 6
    domain_size = 32
    LR = 1e-3
    batch_size = 16
    sample_size = 65536
    sobol_file = 'sobol_d'+str(kl_terms)+'_N'+str(sample_size)+'.npy'
    max_epochs = int(np.ceil(200000 / (sample_size/batch_size)))
    print("Max_epochs = ", max_epochs)

    printsave = MyPrintingCallback()
    dataset = KLSumStochastic(sobol_file, domain_size=domain_size, kl_terms=kl_terms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # dataset = Dataset('../single_instance/example-coefficients.txt', domain_size=64)
    network = AE(in_channels=1, out_channels=1, dims=64, n_downsample=3)
    basecase = Poisson(network, batch_size=batch_size, domain_size=domain_size, learning_rate=LR)


    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="klsum_"+str(domain_size))
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    # trainer = Trainer(accelerator='gpu',devices=1,callbacks=[early_stopping,checkpoint],
    #     logger=[logger,csv_logger],
    #     max_epochs=max_epochs, deterministic=True, profiler='simple')

    ## Inits for updated lightning and wandb logger - EH
    # wandb_logger = WandbLogger(project='IBN', 
    #                            log_model='all')
    
    # checkpoint = ModelCheckpoint(monitor='train_loss',
    #                              mode='min', 
    #                              save_last=True)

    trainer = pl.Trainer(accelerator='gpu',devices=1,
                         callbacks=[early_stopping,checkpoint,printsave],
                         logger=[logger,csv_logger], 
                         max_epochs=max_epochs,
                         fast_dev_run=False
                         )

    # ------------------------
    # 4 Training
    # ------------------------

    trainer.fit(basecase, dataloader)


if __name__ == '__main__':
    main()