import os
import json
import numpy as np
import hydra
import torch
from torch.utils import data

class Dataset(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, args):
        """
        Initialization
        """
        self.args = args        
        # kl_dimension = args.dataset.nu_calc_info.n_sum_nu
        self.coeff_full_batch = []

        if args.dataset.parametric.name == 'single':
            if os.path.exists(hydra.utils.to_absolute_path(args.dataset.parametric.coeff_file)):
                self.coeff = np.loadtxt(hydra.utils.to_absolute_path(args.dataset.parametric.coeff_file), dtype=np.float32)
            else:
                raise FileNotFoundError("Single instance: Wrong path to coefficient file.")
            if not os.path.exists(hydra.utils.to_absolute_path(args.log_dir)):
                os.makedirs(args.log_dir, exist_ok=True)
            if not os.path.exists(os.path.join(hydra.utils.to_absolute_path(args.log_dir),  'coefficients.txt')):
                np.savetxt(os.path.join(args.log_dir,  'coefficients.txt'), self.coeff)
            self.n_samples = args.dataset.parametric.n_samples
        else: # stochastic/parametric
            self.sampling_method = args.dataset.parametric.sampling_method
            if self.sampling_method == 'structured':
                n_samples_1d = args.n_samples_1d
                n_samples = (n_samples_1d ** args.n_sum_nu)  
                self.coeff_full_batch = create_coefficient_samples(args)
                np.save(os.path.join(hydra.utils.to_absolute_path(args.model_dir), 'structured_coeff.npy'), self.coeff_full_batch)
            elif self.sampling_method == 'sobol':
                self.coeff_full_batch = np.load(hydra.utils.to_absolute_path(args.dataset.parametric.sequence_file))
                n_samples = self.coeff_full_batch.shape[0]
                kl_dimension = self.coeff_full_batch.shape[1]
                if kl_dimension != args.dataset.nu_calc_info.n_sum_nu:
                    raise ValueError("Dimension of Sobol sequence should match n_sum_nu")
            elif self.sampling_method == 'random':
                n_samples = args.n_samples
            self.n_samples = n_samples
        print("Number of samples during training = ", self.n_samples)
        # print("Number of minibatches = ", n_batch) 
        # return n_samples_1d, n_samples, kl_dimension, n_batch 
        

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.args.dataset.parametric.name == 'single':
            coeff = self.coeff
        else:
            if self.sampling_method == 'structured' or self.sampling_method == 'sobol':
                coeff = self.coeff_full_batch[index]
                # coeff = coeff[np.newaxis, :]
            elif self.sampling_method == 'random':
                coeff = np.zeros((1,self.args.dataset.nu_calc_info.n_sum_nu))
                for i in range(self.args.dataset.nu_calc_info.n_sum_nu):
                    a_min = -3
                    a_max = 3
                    coeff_rand = a_min + (a_max - a_min) * np.random.rand()
                    coeff[0,i] = coeff_rand
        # print("size of coeff (in dataset) = ", coeff.shape)
        return coeff
