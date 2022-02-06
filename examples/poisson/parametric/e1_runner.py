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
from DiffNet.networks.wgan import GoodNetwork
from DiffNet.DiffNetFEM import DiffNet2DFEM
from DiffNet.datasets.parametric.images import ImageIMBack

from e1_complex_immersed_background import Poisson

def query_plot_contours_and_save(dirpath,nu,f,u, saveid=None):
    outfilename = os.path.join(dirpath, 'query.png')
    if saveid != None:
        outfilename = os.path.join(dirpath, 'query_'+str(saveid)+'.png')
    # plotting
    num_query = nu.shape[0]
    plt_num_row = num_query
    plt_num_col = 2
    fig, axs = plt.subplots(plt_num_row, plt_num_col, figsize=(4*plt_num_col,2.4*plt_num_row),
                        subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
    for ax_row in axs:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    for idx in range(num_query):
        # extract diffusivity and boundary conditions here
        kp = nu[idx,:,:]
        up = u[idx,:,:]

        im0 = axs[idx][0].imshow(kp,cmap='jet')
        fig.colorbar(im0, ax=axs[idx,0])
        im1 = axs[idx][1].imshow(up,cmap='jet')
        fig.colorbar(im1, ax=axs[idx,1]) 
    plt.savefig(outfilename)
    plt.close('all')

def e1_train():
    # dirname = '../ImageDataset'
    dirname = '../AirfoilImageSet'
    dataset = ImageIMBack(dirname, domain_size=256)
    network = GoodNetwork(in_channels=2, out_channels=1, in_dim=64, out_dim=64)
    basecase = Poisson(network, dataset, batch_size=16, domain_size=256)

    logger = pl.loggers.TensorBoardLogger('.', name="complex_immersed_background")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping],
        checkpoint_callback=checkpoint, logger=[logger,csv_logger],
        max_epochs=150, deterministic=True, profiler='simple')

    # Training
    trainer.fit(basecase)
    torch.save(basecase.network, os.path.join(logger.log_dir, 'network.pt'))

def e1_query():
    print("Running query.")
    # dirname = './test_img_data'
    dirname = './af-test'
    case_dir = './complex_immersed_background/version_29'
    query_out_path = os.path.join(case_dir, 'query-test_img_data')
    if not os.path.exists(query_out_path):
        os.makedirs(query_out_path)
    dataset = ImageIMBack(dirname, domain_size=256)
    #network = GoodNetwork(in_channels=2, out_channels=1, in_dim=64, out_dim=64)
    network = torch.load(os.path.join(case_dir, 'network.pt'))
    basecase = Poisson(network, dataset, batch_size=16, domain_size=256)

    nsample = len(basecase.dataset)
    print("nsample = ", nsample)

    nCasesPerImage = 5
    for i in range(nsample//nCasesPerImage+1):
        id0 = nCasesPerImage*i
        id1 = id0+nCasesPerImage
        if id1 > nsample:
            id1 = nsample            
        # core query code
        inputs, forcing = basecase.dataset[id0:id1]
        nu, f, u = basecase.do_query(inputs, forcing)
        query_plot_contours_and_save(query_out_path, nu, f, u, i)

        print("Completed: ", id1 , '/', nsample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtype', '-r', help='Runtype: Train (t) or Query (q)', type=str, choices=['t', 'q'], required=True)
    args = parser.parse_args()

    if args.runtype == 't':
        e1_train()
    elif args.runtype == 'q':
        e1_query()
