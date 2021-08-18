import os
import sys
import json
import torch
import numpy as np
from torch import nn

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
from DiffNet.DiffNetFDM import DiffNetFDM
from DiffNet.datasets.single_instances.klsum import Dataset


class Poisson(DiffNetFDM):
    """docstring for Poisson"""
    def __init__(self, network, dataset, **kwargs):
        super(Poisson, self).__init__(network, dataset, **kwargs)

    def test(self):
        x = torch.linspace(0, 1., 64)
        y = torch.linspace(0, 1., 64)
        xv, yv = torch.meshgrid(x, y)

        print("x = ", xv)
        print("y = ", yv)
        print("sobelx = ", self.sobelx)
        print("sobely = ", self.sobely)

        print("sobelxx = ", self.sobelxx)
        print("sobelyy = ", self.sobelyy)

        sinx = torch.sin(np.pi * xv).type_as(next(self.network.parameters()))
        dxsinx = nn.functional.conv2d(sinx.unsqueeze(0).unsqueeze(0), self.sobelx)
        dysinx = nn.functional.conv2d(sinx.unsqueeze(0).unsqueeze(0), self.sobely)

        fig, axs = plt.subplots(2, 2, figsize=(2*2,1.2*2),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax_row in axs:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])

        im0 = axs[0][0].imshow(sinx.squeeze().detach().cpu(),cmap='jet')
        fig.colorbar(im0, ax=axs[0,0])
        im1 = axs[1][0].imshow(dxsinx.squeeze().detach().cpu(),cmap='jet')
        fig.colorbar(im1, ax=axs[1,0])  
        im1 = axs[1][1].imshow(dysinx.squeeze().detach().cpu(),cmap='jet')
        fig.colorbar(im1, ax=axs[1,1])  
        
        plt.savefig(os.path.join(self.logger[0].log_dir, 'check_' + str(self.current_epoch) + '.png'))
        # self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')
        exit()


    def loss(self, u, inputs_tensor, forcing_tensor):

        # self.test()

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        u_x = nn.functional.conv2d(u, self.sobelx)
        u_y = nn.functional.conv2d(u, self.sobely)
        u_xx = nn.functional.conv2d(u, self.sobelxx)
        u_yy = nn.functional.conv2d(u, self.sobelyy)
        u_laplacian = u_xx + u_yy

        nu_x = nn.functional.conv2d(nu, self.sobelx)
        nu_y = nn.functional.conv2d(nu, self.sobely)

        gradU_DOT_gradNU = torch.mul(u_x, nu_x) + torch.mul(u_y, nu_y)

        # print("size of nu_x = ", nu_x.shape)
        # print("size of nu[:,:,1:-1,1:-1] = ", nu[:,:,1:-1,1:-1].shape)
        # print("size of u_x = ", u_x.shape)
        # print("size of u_laplacian = ", u_laplacian.shape)
        # exit()

        res = gradU_DOT_gradNU + torch.mul(nu[:,:,1:-1,1:-1], u_laplacian)
        # print("res size = ", (res.view(u.shape[0], -1)).shape)

        loss1 = torch.norm(res.view(u.shape[0], -1), p=1, dim=1)
        loss2 = torch.norm(res.view(u.shape[0], -1), p=2, dim=1)

        # print("loss1 = ", loss1, ", size = ", loss1.shape)
        # print("loss2 = ", loss2, ", size = ", loss2.shape)
        # exit()
        # return (0.1*loss1 + 0.9*loss2)
        return loss1

        # nu_gp = self.gauss_pt_evaluation(nu)
        # f_gp = self.gauss_pt_evaluation(f)
        # u_gp = self.gauss_pt_evaluation(u)
        # u_x_gp = self.gauss_pt_evaluation_der_x(u)
        # u_y_gp = self.gauss_pt_evaluation_der_y(u)

        # transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(nu_gp)
        # res_elmwise = transformation_jacobian * (nu_gp * (u_x_gp**2 + u_y_gp**2) - (u_gp * f_gp))
        # res_elmwise = torch.sum(res_elmwise, 1) 

        # loss = torch.mean(res_elmwise)
        # return loss
    def loss_nbc(self, u, inputs_tensor, forcing_tensor):
        # extract diffusivity and boundary conditions here
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        u_x = nn.functional.conv2d(u, self.sobelx)
        u_y = nn.functional.conv2d(u, self.sobely)

        nbc_down = u_x[:,:,0,:]-u_x[:,:,1,:]
        nbc_up = u_x[:,:,-1,:]-u_x[:,:,-2,:]
        
        # print("size of nu_x = ", nu_x.shape)
        # print("size of nu[:,:,1:-1,1:-1] = ", nu[:,:,1:-1,1:-1].shape)
        # print("size of u_x = ", u_x.shape)
        # print("size of u_laplacian = ", u_laplacian.shape)
        # exit()

        res = nbc_down**2 + nbc_up**2
        loss = res.sum(axis=-1)
        # print("res size = ", (res.view(u.shape[0], -1)).shape)

        # loss1 = torch.norm(res.view(u.shape[0], -1), p=1, dim=1)
        # loss2 = torch.norm(res.view(u.shape[0], -1), p=2, dim=1)

        # print("loss1 = ", loss1, ", size = ", loss1.shape)
        # print("loss2 = ", loss2, ", size = ", loss2.shape)
        # exit()
        # return (0.1*loss1 + 0.9*loss2)
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        return self.network[0], inputs_tensor, forcing_tensor

    def training_step(self, batch, batch_idx, optimizer_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        if optimizer_idx == 0:
            loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
            self.log('PDE_loss', loss_val.item())
        else:
            loss_val = self.loss_nbc(u, inputs_tensor, forcing_tensor).mean()
            self.log('NBC_loss', loss_val.item())
        self.log('loss', loss_val.item())
        return loss_val

    def configure_optimizers(self):
        """
        Configure optimizer for network parameters
        """
        # lr = self.learning_rate
        opts = [torch.optim.LBFGS(self.network, lr=1e-1, max_iter=3), torch.optim.LBFGS(self.network, lr=1e-1, max_iter=3)]
        # return opts, []
        # opts = [torch.optim.Adam(self.network.parameters(), lr=1e-4), torch.optim.Adam(self.network.parameters(), lr=1e-4)]
        schd = []
        # schd = [torch.optim.lr_scheduler.MultiStepLR(opts[0], milestones=[10,15,30], gamma=0.1)]
        return opts, schd

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
        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    u_tensor = np.ones((1,1,64,64))
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    dataset = Dataset('example-coefficients.txt', domain_size=64)
    basecase = Poisson(network, dataset, batch_size=1)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="klsum-fdm")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=2000, deterministic=True, profiler="simple")

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