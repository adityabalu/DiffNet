import os
import sys
import json
import math
import torch
import numpy as np

import matplotlib
# matplotlib.use("pgf")
matplotlib.rcParams.update({
    # 'font.family': 'serif',
    'font.size':12,
})
from matplotlib import pyplot as plt

from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
seed_everything(42)

import DiffNet
from DiffNet.networks.wgan import GoodNetwork
from DiffNet.DiffNetFEM import DiffNet2DFEM
# from DiffNet.datasets.single_instances.rectangles import RectangleManufactured
from DiffNet.datasets.single_instances.rectangles import SpaceTimeRectangleManufactured

def stiffness_vs_values_conv(tensor, N, nsd=2, stride=1):
    if nsd == 2:
        conv_gp = nn.functional.conv2d
    elif nsd == 3:
        conv_gp = nn.functional.conv3d

    result_list = []
    for i in range(len(N)):
        result_list.append(conv_gp(tensor, N[i], stride=stride))
    return torch.cat(result_list, 1)

class SpaceTimeHeat(DiffNet2DFEM):
    """docstring for SpaceTimeHeat"""
    def __init__(self, network, dataset, **kwargs):
        super(SpaceTimeHeat, self).__init__(network, dataset, **kwargs)

        self.diffusivity = 1.
        self.u_exact = self.exact_solution(self.xx.numpy(),self.yy.numpy())

        self.Kmatrices = nn.ParameterList()
        Aet = np.array([[-1.0,-0.5,1.0,0.5],[-0.5,-1.0,0.5,1.0],[-1.0,-0.5,1.0,0.5],[-0.5,-1.0,0.5,1.0]])/6.*self.h; 
        Aed = np.array([[2.0,-2.0, 1.0,-1.0],[-2.0, 2.0,-1.0, 1.0], [1.0,-1.0, 2.0,-2.0],[-1.0, 1.0,-2.0, 2.0]])/6.; 
        Kmx = torch.FloatTensor(Aet + self.diffusivity*Aed)
        for j in range(4):
            k = Kmx[j,:].reshape((2,2))
            print("k = ", k*6)
            self.Kmatrices.append(nn.Parameter(k.unsqueeze(0).unsqueeze(1), requires_grad=False))
        # print("self.Kmatrices[0] = ", self.Kmatrices[0])
        print("self.Kmatrices[0].shape = ", self.Kmatrices[0].shape)

    def exact_solution(self, x,y):
        return np.sin(math.pi*x)*np.exp(-y)
        # return np.sin(2.*math.pi*x)*np.sin(2.*math.pi*y)
        # return torch.sin(math.pi*x)*torch.sin(math.pi*y)

    def forcing(self, x, y):
        sin = torch.sin
        cos = torch.cos
        exp = torch.exp
        # return 2. * math.pi**2 * sin(math.pi * x) * cos(math.pi * y)
        # return 2. * math.pi**2 * sin(math.pi * x) * sin(math.pi * y)
        return sin(math.pi * x) * exp(-y) * (self.diffusivity * math.pi**2 - 1.)

    def loss(self, u, inputs_tensor, forcing_tensor):
        f = forcing_tensor # renaming variable
        ff = self.forcing(self.xgp, self.ygp)
        u0 = self.dataset.u0.unsqueeze(0).unsqueeze(0).type_as(u)
        # print("ff.shape = ", ff.shape)
        # print("u0.shape = ", u0.shape)
        
        transformation_jacobian = (0.5*self.h)**2
        JxW = (self.gpw*transformation_jacobian).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        Nf = self.Nvalues*ff.squeeze(0)*JxW
        Nf = torch.sum(Nf, 1).type_as(u) # sum across all gauss points

        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        # NOTE: we do add the BC to the residual later, but adding the residual to u is also very important
        #       because ideally we want to calculate the values of A*u when A is BC adjusted. But since we
        #       are not altering the convolution kernel "Kmatrices" (i.e., effectively the values of A), thus
        #       we will end up with bad values in R at the interior points
        u = torch.where(bc1>0.5,u*0.0+u0,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        R_split = stiffness_vs_values_conv(u, self.Kmatrices)
        R = torch.zeros_like(u)

        R[:,0, 0:-1, 0:-1] += R_split[:,0, :, :]-Nf[0,:,:]
        R[:,0, 0:-1, 1:  ] += R_split[:,1, :, :]-Nf[1,:,:]
        R[:,0, 1:  , 0:-1] += R_split[:,2, :, :]-Nf[2,:,:]
        R[:,0, 1:  , 1:  ] += R_split[:,3, :, :]-Nf[3,:,:]

        # add boundary conditions to R <---- this step is very important
        R = torch.where(bc1>0.5,R*0.0+u0,R)
        R = torch.where(bc2>0.5,R*0.0,R)

        #  DEBUG RELATED STUFF
        # utest0 = torch.FloatTensor(torch.sin(math.pi*self.xx)*torch.cos(math.pi*self.yy))
        # print("xx = n", self.xx)
        # print("yy = n", self.yy)
        # np.savetxt('debugging/xx.txt', self.xx.squeeze().reshape((-1,1)).numpy())
        # np.savetxt('debugging/yy.txt', self.yy.squeeze().reshape((-1,1)).numpy())
        # np.savetxt('debugging/utest0.txt', utest0.squeeze().reshape((-1,1)).numpy())
        # utest = utest0.unsqueeze(0).unsqueeze(0).type_as(next(self.network.parameters()))
        # R_split = stiffness_vs_values_conv(utest, self.Kmatrices)
        # R = torch.zeros_like(utest)

        # R[:,0, 0:-1, 0:-1] += -Nf[0,:,:]
        # R[:,0, 0:-1, 1:  ] += -Nf[1,:,:]
        # R[:,0, 1:  , 0:-1] += -Nf[2,:,:]
        # R[:,0, 1:  , 1:  ] += -Nf[3,:,:]
        # np.savetxt('debugging/R.txt', R.detach().cpu().squeeze().reshape((-1,1)).numpy(),fmt='%.10f')
        # exit()

        loss = torch.norm(R,'fro')**2
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        return self.network[0], inputs_tensor, forcing_tensor

    # def configure_optimizers(self):
    #     """
    #     Configure optimizer for network parameters
    #     """
    #     lr = self.learning_rate
    #     opts = [torch.optim.Adam(self.network, lr=lr)]
    #     return opts, []

    # def training_step(self, batch, batch_idx, optimizer_idx):
    # # def training_step(self, batch, batch_idx):
    #     if self.current_epoch % 2:
    #         opt = 1
    #     else:
    #         opt = 0
    #     if optimizer_idx==opt:
    #         u, inputs_tensor, forcing_tensor = self.forward(batch)
    #         loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
    #         # self.log('PDE_loss', loss_val.item())
    #         # self.log('loss', loss_val.item())
    #     else:
    #         loss_val = torch.zeros((1), requires_grad=True)
    #     return {"loss": loss_val}

    def training_step(self, batch, batch_idx):
    # def training_step(self, batch, batch_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
        # loss_val = self.loss_resmin(u, inputs_tensor, forcing_tensor).mean()
        # loss_val = self.loss_resmin_mass(u, inputs_tensor, forcing_tensor).mean()
        # loss_val = self.loss_matvec(u, inputs_tensor, forcing_tensor).mean()
        # self.log('PDE_loss', loss_val.item())
        # self.log('loss', loss_val.item())
        return {"loss": loss_val}

    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs["loss"]
        return training_step_outputs

    def configure_optimizers(self):
        lr = self.learning_rate
        opts = [torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        # opts = [torch.optim.Adam(self.network, lr=lr)]
        # opts = [torch.optim.Adam(self.network, lr=lr), torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        return opts, []

    def on_epoch_end(self):
        fig, axs = plt.subplots(1, 4, figsize=(2*4,1.2),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        self.network.eval()
        inputs, forcing = self.dataset[0]

        u, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

        f = forcing_tensor.squeeze().detach().cpu() # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # apply boundary conditions
        u = torch.where(bc1>0.5,1.0+u*0.0,u)
        u = torch.where(bc2>0.5,u*0.0,u)


        k = nu.squeeze().detach().cpu()
        u = u.squeeze().detach().cpu()
        self.u_curr = u

        u_exact = self.u_exact.squeeze()
        diff = u - u_exact
        # print(np.linalg.norm(diff.flatten())/self.domain_size)
        im0 = axs[0].imshow(f,cmap='jet', vmin=0.0, vmax=20.0)
        # fig.colorbar(im0, ax=axs[0], ticks=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0])
        fig.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(u,cmap='jet')
        fig.colorbar(im1, ax=axs[1])
        im2 = axs[2].imshow(u_exact,cmap='jet', vmin=0.0, vmax=1.0)
        fig.colorbar(im2, ax=axs[2])
        im3 = axs[3].imshow(diff,cmap='jet', vmin=0.0, vmax=0.5)
        fig.colorbar(im3, ax=axs[3])
        ff = self.forcing(self.xgp, self.ygp)
        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

    def calc_l2_err(self):
        cn = lambda j,n: [j,j+1,j+n,(j+1)+n]
        N = lambda x,y: (1./4.)*np.array([(1-x)*(1-y), (1+x)*(1-y), (1-x)*(1+y), (1+x)*(1+y)])
        transform = lambda a,b,x: ((a+b)/2. + (b-a)/2.*x)

        ngp = 4
        gpx = np.array([-0.577350269189626, 0.577350269189626, -0.577350269189626, 0.577350269189626])
        gpy = np.array([-0.577350269189626, -0.577350269189626, 0.577350269189626, 0.577350269189626])
        gpw = np.ones(4)

        nnodex = self.domain_size
        nnodey = self.domain_size
        nelmx = self.domain_size - 1
        nelmy = self.domain_size - 1
        hx = (1. / nelmx)
        hy = (1. / nelmy)
        J = (hx/2.)*(hy/2.)
        print("J = ", J)

        x = np.linspace(0,1,self.domain_size)
        y = np.linspace(0,1,self.domain_size)
        u_sol = self.u_curr.numpy()

        usolnorm = 0.
        uexnorm = 0.
        l2_err = 0.
        for j in range(nelmy):
            for i in range(nelmx):
                local_u = (u_sol[j:j+2,i:i+2].reshape(4,1)).squeeze()
                # print("local_u = ", local_u)
                for igp in range(ngp):
                    basis = N(gpx[igp], gpy[igp])
                    # print("basis = ", basis)
                    xp = transform(x[i],x[i+1],gpx[igp])
                    yp = transform(y[j],y[j+1],gpy[igp])

                    u1 = np.dot(local_u, basis)
                    u2 = self.exact_solution(xp,yp)

                    # print("(xp,yp,uex,usol) = ", xp,yp,u2,u1)
                    # print("u1 = ", u1)
                    # print("u2 = ", u2)
                    # print("gpw(igp) = ", gpw[igp])
                    # print("J = ", J)
                    l2_err += (u1 - u2)**2 * J
                    usolnorm += u1**2*J
                    uexnorm += u2**2*J

        l2_err = np.sqrt(l2_err)
        usolnorm = np.sqrt(usolnorm)
        uexnorm = np.sqrt(uexnorm)

        u_ex = self.u_exact.squeeze()


        print("usol.shape =", u_sol.shape)
        print("uex.shape =", u_ex.shape)
        print("||u_sol||, ||uex|| = ", usolnorm, uexnorm)
        print("||e||_{{L2}} = ", l2_err)
        # by taking vector norm
        print("||e|| (vector-norm) = ", np.linalg.norm(u_ex - u_sol, 'fro')/nnodex)


def main():
    # u_tensor = np.random.randn(1,1,256,256)

    caseId = 0
    domain_size = 64
    dir_string = "spacetime-heat"
    max_epochs = 25
    
    u_tensor = np.ones((1,1,domain_size,domain_size))
    network = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(u_tensor), requires_grad=True)])
    dataset = SpaceTimeRectangleManufactured(domain_size=domain_size)
    basecase = SpaceTimeHeat(network, dataset, batch_size=1, domain_size=domain_size, learning_rate=0.01)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name=dir_string)
    # logger = pl.loggers.TensorBoardLogger('.', name="simple-resmin")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
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

    basecase.calc_l2_err()


if __name__ == '__main__':
    main()