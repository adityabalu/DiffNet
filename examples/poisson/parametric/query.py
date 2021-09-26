import os
import sys
import json
import torch
import numpy as np
import argparse

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

class Poisson(DiffNet2DFEM):
    """docstring for Poisson"""
    def __init__(self, network, dataset, **kwargs):
        super(Poisson, self).__init__(network, dataset, **kwargs)

    def loss(self, u, inputs_tensor, forcing_tensor):
        return 0.0

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        u = self.network(inputs_tensor[:,0:1,:,:])
        return u, inputs_tensor, forcing_tensor

    def training_step(self, batch, batch_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
        return {"loss": loss_val}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        print("Printing U = ")
        print(u.shape)
        print(u)

    def query_and_plot(self, args):
        output_path = args.model_dir

        num_query = 10
        plt_num_row = num_query
        plt_num_col = 2
        fig, axs = plt.subplots(plt_num_row, plt_num_col, figsize=(2*plt_num_col,1.2*plt_num_row),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax_row in axs:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
        
        # self.network.eval()
        inputs, forcing = self.dataset[0:num_query]
        forcing = forcing.repeat(num_query,1,1,1)
        print("\ninference for: ", self.dataset.coeffs[0:num_query])

        ub, inputs_tensor, forcing_tensor = self.forward((inputs.type_as(next(self.network.parameters())), forcing.type_as(next(self.network.parameters()))))
        
        loss = self.loss(ub, inputs_tensor, forcing_tensor[:,0:1,:,:])
        print("loss incurred for this coeff:", loss)        

        for idx in range(num_query):
            f = forcing_tensor # renaming variable
            
            # extract diffusivity and boundary conditions here
            nu = inputs_tensor[idx,0:1,:,:]
            u = ub[idx,0:1,:,:]
            bc1 = inputs_tensor[idx,1:2,:,:]
            bc2 = inputs_tensor[idx,2:3,:,:]

            # apply boundary conditions
            u = torch.where(bc1>0.5,1.0+u*0.0,u)
            u = torch.where(bc2>0.5,u*0.0,u)

            k = nu.squeeze().detach().cpu()
            u = u.squeeze().detach().cpu()

            im0 = axs[idx][0].imshow(k,cmap='jet')
            fig.colorbar(im0, ax=axs[idx,0])
            im1 = axs[idx][1].imshow(u,cmap='jet')
            fig.colorbar(im1, ax=axs[idx,1])  
        plt.savefig(os.path.join(output_path, 'query_contour_' + '.png'))
        # self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

        npz_filepath = os.path.join(output_path, 'query_data_' + '.npz')
        np.savez(npz_filepath, u=u, k=k)

    def query_statistical(self, args):
        output_path = args.model_dir

        query_batch_size = 128
        num_query_sample = self.dataset.coeffs.shape[0]

        assert(query_batch_size < num_query_sample)

        num_batch = num_query_sample // query_batch_size
        if (num_query_sample % query_batch_size) > 0:
            num_batch += 1

        all_inf = np.zeros((num_query_sample, self.domain_size, self.domain_size), dtype=np.float64)
        mean_value = np.zeros((self.domain_size, self.domain_size), dtype=np.float64)

        # nine points where histogram of solution values will be plotted
        point_coord = np.array([0.2,0.5,0.7],dtype=np.float64)
        npoint_1d = point_coord.shape[0]
        point_values = np.zeros((num_query_sample, npoint_1d**2), dtype=np.float64)
        point_idx = (point_coord * self.domain_size).astype(int)
        print("Point_idx", point_idx)


        ## LOOP ON STRUCTURED SAMPLE QUERIES
        for ibatch in range(num_batch):
            print("Batches done = ", ibatch , " / ", num_batch)
            idx_0 = ibatch * query_batch_size
            idx_1 = idx_0 + query_batch_size
            if ibatch == num_batch-1:
                idx_1 = num_query_sample

            inputs, forcing = self.dataset[idx_0:idx_1]
            forcing = forcing.repeat(query_batch_size,1,1,1)

            # print("\nInference for: ", self.dataset.coeffs[idx_0:idx_1])

            u, inputs_tensor, forcing_tensor = self.forward((inputs.type_as(next(self.network.parameters())), forcing.type_as(next(self.network.parameters()))))

            # extract diffusivity and boundary conditions here
            nu = inputs_tensor[:,0:1,:,:]
            bc1 = inputs_tensor[:,1:2,:,:]
            bc2 = inputs_tensor[:,2:3,:,:]

            # apply boundary conditions
            u = torch.where(bc1>0.5,1.0+u*0.0,u)
            u = torch.where(bc2>0.5,u*0.0,u)

            print("u.shape = ", (u).shape)

            u = u.squeeze(1).detach().cpu().numpy()
            all_inf[idx_0:idx_1,:,:] = u
            mean_value += u.sum(axis=0)

            for j in range(npoint_1d):
                for i in range(npoint_1d):
                    I = j * npoint_1d + i
                    point_values[idx_0:idx_1, I] =  u[:,point_idx[j], point_idx[i]]

        np.save(os.path.join(output_path,'q_all.npy'), all_inf)
        np.save(os.path.join(output_path,'q_point_values.npy'), point_values)

        q_mean, q_sdev = self.calc_mean_stddev(all_inf)
        np.save(os.path.join(output_path, 'q_mean.npy'), q_mean)
        np.save(os.path.join(output_path, 'q_sdev.npy'), q_sdev)

    def calc_mean_stddev(self, X):
        mean = np.mean(X, axis=0)
        var = (X-mean)**2
        var = np.mean(var, axis=0)
        std = np.sqrt(var)
        return mean, std

    def calc_mean_stddev_from_file(self, args):
        q_all = np.load('./q_all.npy')
        q_mean, q_sdev = self.calc_mean_stddev(q_all)
        np.save(os.path.join(output_path, 'q_mean.npy'), q_mean)
        np.save(os.path.join(output_path, 'q_sdev.npy'), q_sdev)
        

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-m', help='Directory to store model files', required=True)
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    # args.model_dir = get_next_run(args.model_dir)

    
    case_dir = args.model_dir
    print("Case_dir, model_dir, output_path = ", case_dir)
    epoch = 5
    step = 1001
    # query_output_path = get_next_run(case_dir)

    kl_terms = 6
    domain_size = 32
    batch_size = 10
    # sobol_file = 'query-coeffs.npy'
    sobol_file = 'qcoeff-stat-epsilon-sobol.npy'
    
    dataset = KLSumStochastic(sobol_file, domain_size=domain_size, kl_terms=kl_terms)
    # dataset = Dataset('../single_instance/example-coefficients.txt', domain_size=64)
    # network = AE(in_channels=1, out_channels=1, dims=64, n_downsample=3)
    network = torch.load(os.path.join(case_dir, 'network.pt'))
    basecase = Poisson(network, dataset, batch_size=batch_size, domain_size=domain_size, learning_rate=0.01)
    # print(next(basecase.network.parameters()))
    # trainer = Trainer()
    # trainer.predict(basecase, dataset)

    # basecase.query_and_plot(args)
    basecase.query_statistical(args)

if __name__=="__main__":
    main()


