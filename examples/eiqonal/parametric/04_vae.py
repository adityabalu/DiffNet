import os, sys, json, torch, math
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
from DiffNet.networks.dgcnn import DGCNN2D
from DiffNet.networks.immdiff_networks import ImmDiff, IBN_DGCNN2d, ImmDiff_VAE
from DiffNet.DiffNetFEM import DiffNet2DFEM
import PIL
from torch.utils import data
import torchvision
import torch.nn.functional as F



class ParametricNURBS(data.Dataset):
    def __init__(self, data_path, domain_size=128):
        self.data_path = data_path

        points = np.load(os.path.join(self.data_path, 'point_cloud.npz'))['arr_0']
        normals = np.load(os.path.join(self.data_path, 'normals.npz'))['arr_0']
        points = points[:6]
        normals = normals[:6]
        self.normals = normals[:,:,:2]
        self.pc = points*0.25
        # self.pc[:,:,1:2] += 0.5
        # self.pc[:,:,0:1] += 0.25
        x_translation = np.random.uniform(low=0.1, high=0.7, size=(6,1))
        y_translation = np.random.uniform(low=0.3, high=0.7, size=(6,1))
        # x_translation = np.array((((0.7, 0.2),))).T
        # y_translation = np.array(((0.7, 0.2),)).T
        # print(x_translation.shape)
        # print(x_translation.T.shape)
        # exit()
        self.pc[:,:,0] += x_translation
        self.pc[:,:,1] += y_translation
        # print(self.pc.shape)
        # exit()
        self.area = np.zeros((self.pc.shape[0], self.pc.shape[1], 1))
        self.area[:, 1:-1, 0] = np.sum((self.pc[:,1:-1,:] - self.pc[:,0:-2,:])**2, -1)*0.5 + np.sum((self.pc[:,2:,:] - self.pc[:,1:-1,:])**2, -1)*0.5 
        self.area[:, 0, 0] = np.sum((self.pc[:,1,:] - self.pc[:,0,:])**2, -1)*0.5 + np.sum((self.pc[:,-1,:] - self.pc[:,0,:])**2, -1)*0.5 
        self.area[:,-1, 0] = np.sum((self.pc[:,-1,:] - self.pc[:,-2,:])**2, -1)*0.5 + np.sum((self.pc[:,-1,:] - self.pc[:,0,:])**2, -1)*0.5 
        self.domain = np.ones((domain_size,domain_size))
        self.domain_size = domain_size

    def __len__(self):
        return self.pc.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.concatenate((self.pc[index:index+1,:,:], self.normals[index:index+1,:,:], self.area[index:index+1,:,:]), -1) # 2, Npoint, 2
        forcing = np.ones_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)


def compute_winding_nodes(points, normals, area, q):
    points = points.view(points.size(0), points.size(2), points.size(3)).permute(0, 2, 1)
    normals = normals.view(normals.size(0), normals.size(2), normals.size(3)).permute(0, 2, 1)
    area = area.view(area.size(0), area.size(2), area.size(3)).permute(0, 2, 1)
    points = points.unsqueeze(-2).unsqueeze(0)
    normals = normals.unsqueeze(-2).unsqueeze(0)
    area = area.unsqueeze(-2).unsqueeze(0)
    q = q.unsqueeze(-1)

    winding_number = torch.sum((torch.stack([torch.sum((points - q[:, :, q_idx, :])*normals,2)
        for q_idx in range(q.size(2))],
        0)) / (4*math.pi*torch.stack([torch.sum(torch.sqrt((points - q[:, :, q_idx, :])**2),2)
        for q_idx in range(q.size(2))], 0))**3, -1)

    winding_number = winding_number.permute(2,1,0,3)
    return winding_number



class Eikonal(DiffNet2DFEM):
    """This is really a poisson solver, but uses eikonal for one of the boundary condition computations - will change later eh """

    def __init__(self, network, dataset, **kwargs):
        super(Eikonal, self).__init__(network, dataset, **kwargs)
        self.save_frequency = kwargs.get('save_frequency', 5)
        self.masking_type = kwargs.get('masking_type', 'eikonal')
        self.domain_size = kwargs.get('domain_size', 128)

        self.network = network
        self.tau = 0.25
        self.diffusivity = 1.

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


    def loss_poisson_windingnumber(self, u, inputs_tensor, forcing_tensor):
        N_values = self.Nvalues.type_as(u)
        dN_x_values = self.dN_x_values.type_as(u)
        dN_y_values = self.dN_y_values.type_as(u)
        gpw = self.gpw.type_as(u)

        # extract diffusivity and boundary conditions here
        f = forcing_tensor
        pc = inputs_tensor[:,:,:,0:2] # b, 1, npoints, 2
        normals = inputs_tensor[:,:,:,2:4] # b, 1, npoints, 2
        area = inputs_tensor[:,:,:,4:5] # b, 1, npoints, 1
        nodes = torch.stack((self.xx, self.yy),0).type_as(pc)

        # print('* '*10)
        # print(pc.shape)
        # print(normals.shape)
        # print(area.shape)
        # print(nodes.shape)

        # winding_number = compute_winding_torch(pc, normals, area, gpts)
        winding_number = compute_winding_nodes(pc, normals, area, nodes)
        # print(winding_number.shape)
        # exit()

        # DERIVE NECESSARY VALUES
        trnsfrm_jac = (0.5*self.h)*(0.5*self.h)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        # apply boundary conditions
        u = torch.where(winding_number>0.1, u*0.0, u)

        u_x_gp = self.gauss_pt_evaluation_der_x(u).unsqueeze(1)
        u_y_gp = self.gauss_pt_evaluation_der_y(u).unsqueeze(1)
        f_gp = self.gauss_pt_evaluation(f).unsqueeze(1)

        # CALCULATION STARTS
        # lhs
        vxux = dN_x_values*u_x_gp*JxW
        vyuy = dN_y_values*u_y_gp*JxW
        # rhs
        vf = N_values*f_gp*JxW

        # integrated values on lhs & rhs
        v_lhs = torch.sum(self.diffusivity*(vxux+vyuy), 2) # sum across all GP
        v_rhs = torch.sum(vf, 2) # sum across all gauss points

        # unassembled residual
        R_split = v_lhs - v_rhs
        # assembly
        R = torch.zeros_like(u); R = self.Q1_vector_assembly(R, R_split)
        R = torch.where(winding_number>0.1, R*0.0, R)

        loss = torch.sum(R**2) 

        return loss

    
    def loss_vae(self, mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)


    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        nu = inputs_tensor[:,0:1,:,:2].squeeze(1)
        u, mu, logvar = self.network(nu)
        return u, mu, logvar, inputs_tensor, forcing_tensor


    def configure_optimizers(self):
        """
        Configure optimizer for network parameters
        """
        lr = self.learning_rate
        # opts = [torch.optim.LBFGS(self.network.parameters(), lr=1.0, max_iter=5)]
        opts = [torch.optim.Adam(self.network.parameters(), lr=lr)]
        return opts


    def training_step(self, batch, batch_idx):
        u, mu, logvar, inputs_tensor, forcing_tensor = self.forward(batch)
        # if self.masking_type == 'eikonal':
        #     loss_eikonal = self.loss_eikonal(u[:,0:1,:,:], inputs_tensor, forcing_tensor).mean() # make sdf pred be second channel
        #     # loss_poisson = self.loss_poisson_eikonal(u, inputs_tensor, forcing_tensor).mean()
        #     loss_val = loss_eikonal# + loss_poisson
        if self.masking_type == 'WN':
            loss_poisson = self.loss_poisson_windingnumber(u, inputs_tensor, forcing_tensor).sum()
            loss_vae = self.loss_vae(mu, logvar)
            loss_val = loss_poisson + loss_vae
        return {"loss": loss_val}


    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs["loss"]
        self.log('PDE_loss', loss.item())
        self.log('loss', loss.item())
        return training_step_outputs


    def do_query(self, inputs, forcing):
        inputs = inputs.unsqueeze(1)
        forcing = forcing.unsqueeze(1)
        # print(inputs.shape)
        # exit()
        u, _, _, _, _ = self.forward((inputs.type_as(next(self.network.parameters())), 
                                                         forcing.type_as(next(self.network.parameters()))))

        pc = inputs[:,:,:,0:2] # b, 1, npoints, 2
        normals = inputs[:,:,:,2:4] # b, 1, npoints, 2
        area = inputs[:,:,:,4:5] # b, 1, npoints, 1
        nodes = torch.stack((self.xx, self.yy),0).type_as(pc)

        winding_number = compute_winding_nodes(pc, normals, area, nodes)
        winding_number = winding_number.squeeze().detach().cpu()

        u = u.squeeze().detach().cpu()
        if self.masking_type == 'WN':
            return u, winding_number
        elif self.masking_type == 'eikonal':
            return u


    def on_epoch_end(self):     
        if self.current_epoch % self.save_frequency == 0:
            self.network.eval()
            inputs_1, forcing_1 = self.dataset[3]
            # inputs, forcing = self.dataset[0]
            inputs_2, forcing_2 = self.dataset[5]
            inputs = torch.cat((inputs_1, inputs_2), 0)
            forcing = torch.cat((forcing_1, forcing_2), 0)
            if self.masking_type == 'WN':
                u, winding_number = self.do_query(inputs, forcing)
                inputs = winding_number
            elif self.masking_type == 'eikonal':
                u = self.do_query(inputs, forcing)
            self.plot_contours(u, inputs)


    def plot_contours(self, u, inputs):
        # unpack
        if self.masking_type == "eikonal":
            # u_sdf = u[0]
            # u = u[-1]

            fig, axs = plt.subplots(1, 3, figsize=(2*4,2),
                                subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])

            im0 = axs[0].imshow(u,cmap='jet')
            fig.colorbar(im0, ax=axs[0])
            axs[0].set_title('u')

            # im1 = axs[1].imshow(u_sdf,cmap='jet')
            # fig.colorbar(im1, ax=axs[1])
            # axs[1].set_title('SDF')

            # im2 = axs[2].imshow(abs(u_sdf),cmap='jet')
            # fig.colorbar(im2, ax=axs[2])
            # axs[2].set_title('UDF')

        elif self.masking_type == "WN":
            fig, axs = plt.subplots(2, 2, figsize=(6,4),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)

            for i in range(axs.shape[0]-1):
                for j in range(axs.shape[1]):
                    axs[i,j].set_xticks([])
                    axs[i,j].set_yticks([])

            im0 = axs[0,0].imshow(u[0],cmap='jet')
            fig.colorbar(im0, ax=axs[0,0])
            axs[0,0].set_title('u')

            im1 = axs[0,1].imshow(inputs[0],vmin=0.0, vmax=1.0,cmap='jet')
            fig.colorbar(im1, ax=axs[0,1])
            axs[0,1].set_title('WN') 

            im2 = axs[1,0].imshow(u[1],cmap='jet')
            fig.colorbar(im2, ax=axs[1,0])

            im3 = axs[1,1].imshow(inputs[1],vmin=0.0, vmax=1.0,cmap='jet')
            fig.colorbar(im3, ax=axs[1,1])

        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')




def main():
    dataset_path = '/data/EthanHerron/DiffNet/DiffNet/DiffNet/datasets/parametric/nurbs'
    dataset = ParametricNURBS(dataset_path)
    domain_size = 128
    # network = DGCNN2D(domain_size, 1000)
    masking_type = 'WN' # options: 'eikonal' OR 'WN' ie winding number
    batch_size = 3
    max_epochs = 1000
    if masking_type == 'WN':
        network = ImmDiff_VAE(out_channels=1)
        # network = IBN_DGCNN2d()
    elif masking_type == 'eikonal':
        network = ImmDiff(out_channels=2)
    basecase = Eikonal(network, dataset, batch_size=batch_size, masking_type=masking_type, domain_size=domain_size)
    log_string = 'vae'

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name=log_string)
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=max_epochs, deterministic=True, profiler='simple')

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