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
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        # x = F.pad(x, self._padding(x), mode='reflect')
        # x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        # x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

class Rectangle(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """
        self.domain = np.ones((domain_size, domain_size))
        
        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        # self.bc1[0,:] = 1
        
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        # self.bc2[-1,int(0.4*domain_size):int(0.6*domain_size)] = 1
        self.bc2[:,0] = 1
        self.bc2[0,:] = 1
        self.n_samples = 100

        x = np.linspace(0,1,domain_size)
        y = np.linspace(0,1,domain_size)
        xx, yy = np.meshgrid(x,y)
        self.xx = xx
        self.yy = yy


    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.bc1, self.bc2, self.xx, self.yy])
        forcing = np.ones_like(self.domain)*1
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)

class Poisson(DiffNet2DFEM):
    """docstring for Poisson"""
    def __init__(self, network, dataset, **kwargs):
        super(Poisson, self).__init__(network, dataset, **kwargs)
        self.target_vf_sum = 0.4*self.domain_size**2
        self.median_filter = MedianPool2d(kernel_size=3,padding=1)
    

    def loss(self, network_inp, inputs_tensor, forcing_tensor):

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = self.median_filter(0.001 + (1.0 - 0.0)*torch.sigmoid(network_inp[1]).type_as(f)**3)
        # nu = torch.clip(torch.nn.functional.sigmoid(network_inp[1].type_as(f)), 0.001, 1.0)
        bc1 = inputs_tensor[:,0:1,:,:]
        bc2 = inputs_tensor[:,1:2,:,:]
        xx = inputs_tensor[:,2:3,:,:]
        yy = inputs_tensor[:,3:4,:,:]

        # apply boundary conditions
        u = network_inp[0].type_as(f)
        v = u
        # u = torch.where(bc1>0.5,1.0+u*0.0,u)
        # u = torch.where(bc2>0.5,u*0.0,u)        
        dbc1 = (bc1*(u -  (u*0+1))**2).mean()
        dbc2 = (bc2*(u -  (u*0))**2).mean()
        #v = (xx**2 + yy**2)*u
        #v = (xx**2 + yy**2)
        #v = (xx + yy)


        nu_gp = self.gauss_pt_evaluation(nu)
        f_gp = self.gauss_pt_evaluation(f)
        u_gp = self.gauss_pt_evaluation(u)
        v_gp = self.gauss_pt_evaluation(v)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        v_x_gp = self.gauss_pt_evaluation_der_x(v)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)
        v_y_gp = self.gauss_pt_evaluation_der_y(v)

        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(nu_gp)
        res_elmwise = (transformation_jacobian * (0.5*nu_gp * (u_x_gp*v_x_gp + u_y_gp*v_y_gp) - (v_gp*f_gp))) 
        res_elmwise = torch.sum(res_elmwise, 1) + dbc1 + dbc2

        loss = torch.mean(res_elmwise)
        return loss

    def compliance(self, network_inp, inputs_tensor, forcing_tensor):

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = self.median_filter(0.001 + (1.0 - 0.0)*torch.sigmoid(network_inp[1]).type_as(f)**3)
        # nu = torch.clip(torch.nn.functional.sigmoid(network_inp[1].type_as(f)), 0.001, 1.0)
        bc1 = inputs_tensor[:,0:1,:,:]
        bc2 = inputs_tensor[:,1:2,:,:]

        # apply boundary conditions
        u = network_inp[0].type_as(f)
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        nu_gp = self.gauss_pt_evaluation(nu)
        f_gp = self.gauss_pt_evaluation(f)
        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(nu_gp)
        #res_elmwise = -10  * transformation_jacobian * (f_gp*u_gp)
        #res_elmwise = -1  * transformation_jacobian * (f_gp*u_gp)
        #res_elmwise = (transformation_jacobian * (0.5*nu_gp * (u_x_gp*u_x_gp + u_y_gp*u_y_gp) + (u_gp*f_gp))) 
        #res_elmwise = transformation_jacobian * (u_gp*f_gp)
        

        #res_elmwise = transformation_jacobian * (0.5*nu_gp * (u_x_gp*u_x_gp + u_y_gp*u_y_gp))
        res_elmwise = -1*transformation_jacobian * (u_gp*f_gp)

        #res_elmwise = 

        res_elmwise = torch.sum(res_elmwise, 1) 

        loss = torch.mean(res_elmwise)
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        # return [self.network[0][0], self.network[1][0]], inputs_tensor, forcing_tensor
        # return [self.network[0](self.network[1][0]), self.network[1][0]], inputs_tensor, forcing_tensor #network[1] is a param list and network[1][0] is to access the actual params
        u_tensor = self.network[1]
        nu_tensor = self.network[2]
        u_nu_tensor = torch.squeeze(torch.stack([u_tensor, nu_tensor], dim=1),dim=2)
        
        u_nu_out = self.network[0](u_nu_tensor)
        # first dim is 0 because assuming only 1 batch
        u_out = u_nu_out[0,0,:,:].view(1, 1, 64, 64)
        nu_out = u_nu_out[0,1,:,:].view(1, 1, 64, 64)
        return [u_out, nu_out], inputs_tensor, forcing_tensor

    def configure_optimizers(self):
        """
        Configure optimizer for network parameters
        """
        lr = self.learning_rate #default is 3e-4
        lr = 0.001
        # opts = [torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        # opts = [torch.optim.Adam(self.network[0], lr=lr), torch.optim.Adam(self.network[1], lr=lr), torch.optim.Adam(self.network[1], lr=lr)]
        # opts = [torch.optim.Adam(self.network[0].parameters(), lr=lr), torch.optim.Adam(self.network[1], lr=lr), torch.optim.Adam(self.network[1], lr=lr)]

        # for joint network, 3 different optimizers on same network
        opts = [torch.optim.Adam(self.network[0].parameters(), lr=lr), torch.optim.Adam(self.network[0].parameters(), lr=lr), torch.optim.Adam(self.network[0].parameters(), lr=lr)]
        return opts, []


    def training_step(self, batch, batch_idx, optimizer_idx):
    # def training_step(self, batch, batch_idx):
        network_out, inputs_tensor, forcing_tensor = self.forward(batch)
        # if (self.current_epoch%20) < 10:
        #     loss_val = self.loss(network_out, inputs_tensor, forcing_tensor).mean()
        #     if optimizer_idx is 0:
        #         self.log('PDE_loss', loss_val.item())
        #     else:
        #         nu = self.median_filter(0.001 + (1.0 - 0.0)*network_out[1]**2)
        #         loss_val = loss_val + 100*(nu.mean() - self.target_vf)**2
        # else:
        # if self.current_epoch%100 == 0 and batch_idx ==  0:
        #     self.network[1][0].data = self.network[1][0].data + 0.1*torch.randn(self.network[1][0].data.size())

        if optimizer_idx is 0:
            loss_val = self.loss(network_out, inputs_tensor, forcing_tensor).mean()
            self.log('PDE_loss', loss_val.item())
            #print('PDE_loss', loss_val.item())
        # else:
        #     loss_val = torch.zeros((1), requires_grad=True)
        elif optimizer_idx is 1:
            nu = self.median_filter(0.001 + (1.0 - 0.0)*torch.sigmoid(network_out[1])**3)
            loss_val = self.compliance(network_out, inputs_tensor, forcing_tensor).mean()
            self.log('Compliance', loss_val.item())
            #print('Compliance', loss_val.item())
        else:
            nu = self.median_filter(0.001 + (1.0 - 0.0)*torch.sigmoid(network_out[1])**3)
            loss_val = (nu.sum() - self.target_vf_sum)**2
            self.log('VF_constraint', loss_val.item())
            #print('VF', loss_val.item())

        return {"loss": loss_val}
        # if  (self.current_epoch%20) < 10:
        #     nu = self.median_filter(0.001 + (1.0 - 0.0)*network_out[1]**3)
        #     loss_val = (nu.sum() - self.target_vf_sum)**2
        #     self.log('VF_constraint', loss_val.item())
        #     #print('VF', loss_val.item())
        #     return {"loss": loss_val}
        # else:
        #     if optimizer_idx is 2:
        #         loss_val = self.loss(network_out, inputs_tensor, forcing_tensor).mean()
        #         self.log('PDE_loss', loss_val.item())
        #         #print('PDE_loss', loss_val.item())
        #     # else:
        #     #     loss_val = torch.zeros((1), requires_grad=True)
        #     elif optimizer_idx is 1:
        #         nu = self.median_filter(0.001 + (1.0 - 0.0)*network_out[1]**3)
        #         loss_val = self.compliance(network_out, inputs_tensor, forcing_tensor).mean()
        #         self.log('Compliance', loss_val.item())
        #         #print('Compliance', loss_val.item())
        #     else:
        #         nu = self.median_filter(0.001 + (1.0 - 0.0)*network_out[1]**3)
        #         loss_val = (nu.sum() - self.target_vf_sum)**2
        #         self.log('VF_constraint', loss_val.item())
        #         #print('VF', loss_val.item())

        #     return {"loss": loss_val}
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
        fig, axs = plt.subplots(1, 3, figsize=(2*3,2),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        inputs, forcing = self.dataset[0]

        #network_inp, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network[0].parameters())), forcing.unsqueeze(0).type_as(next(self.network[0].parameters()))))

        single_batch = (inputs.unsqueeze(0), forcing.unsqueeze(0))
        network_out, inputs_tensor, forcing_tensor = self.forward(single_batch)

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        # nu = torch.clip(torch.nn.functional.sigmoid(network_inp[1].type_as(f)), 0.001, 1.0)
        nu = self.median_filter(0.001 + (1.0 - 0.0)*torch.sigmoid(network_out[1]).type_as(f)**3)
        # nu = torch.relu(network_inp[1].type_as(f)) + 0.001
        bc1 = inputs_tensor[:,0:1,:,:]
        bc2 = inputs_tensor[:,1:2,:,:]

        # apply boundary conditions
        u = network_out[0].type_as(f)
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        k = nu.squeeze().detach().cpu()
        u = u.squeeze().detach().cpu()
        print('max u:', u.max(), 'k mean:',k.mean())
        im0 = axs[0].imshow(k,cmap='jet', vmin=0.0, vmax=1.0)
        fig.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(u,cmap='jet',vmin=0.0, vmax=1.0)
        fig.colorbar(im1, ax=axs[1])  
        im2 = axs[2].imshow(u,cmap='jet')
        fig.colorbar(im2, ax=axs[2])  
        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

def main():
    domain_size = 64
    
    #nu_tensor = np.ones((1,1,domain_size,domain_size))
    nu_tensor = np.random.randn(1,1,domain_size,domain_size)*100

    #u_tensor = np.ones((1,1,domain_size,domain_size))
    u_tensor = np.random.randn(1,1,domain_size,domain_size)*100
    
    #network1 = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    #network1 = AE(in_channels=1, out_channels=1, dims=16, n_downsample=3)
    
    #network2 = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(nu_tensor), requires_grad=True)])
    #network2 = AE(in_channels=1, out_channels=1, dims=16, n_downsample=3)

    # joint_network = AE(in_channels=2, out_channels=2, dims=16, n_downsample=3)
    joint_network = torch.load('/data/EthanHerron/TopologyOptimization/DiffNet/DiffNet/pretrained_AE/microstructure_AE.pt')

    dataset = Rectangle(domain_size=domain_size)
    
    #basecase = Poisson([network1, network2], dataset, batch_size=1)
    basecase = Poisson([joint_network, torch.FloatTensor(u_tensor), torch.FloatTensor(nu_tensor)], dataset, batch_size=1)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="topopt")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('PDE_loss',
        min_delta=1e-8, patience=15, verbose=False, mode='min', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='PDE_loss',
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