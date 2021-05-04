import os
import sys
import time
import json
import hydra
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from torch import nn
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from diffnet import pde_dict
from matplotlib.colors import LogNorm
seed_everything(42)

torch.cuda.set_device(0)    
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

def gen_ellipse_sink():
	temp =  torch.zeros(64, 64)
	cy = torch.randint(low=32, high=63, size=(1,))
	cx = torch.randint(low=0, high=62, size=(1,))
	r = torch.randint(low=1, high=min(int(64-cx),int(64-cy)), size=(1,))

	# Create index arrays to temp
	X,Y=torch.meshgrid(torch.arange(temp.shape[0]),torch.arange(temp.shape[1]))
	# calculate distance of all points to centre
	dist=torch.sqrt((X-cx)**2+(Y-cy)**2)
	# Assign value of 1 to those points where dist<cr:
	temp[torch.where(dist<r)]=1
	temp = torch.unsqueeze(temp,0)
	return temp

def gen_ellipse_src(i):
	temp =  torch.zeros(64, 64)
	cy = torch.randint(low=0, high=30, size=(1,))
	cx = torch.randint(low=0, high=62, size=(1,))
	r = torch.randint(low=1, high=min(int(64-cx),int(32-cy)), size=(1,))

	# Create index arrays to temp
	X,Y=torch.meshgrid(torch.arange(temp.shape[0]),torch.arange(temp.shape[1]))
	# calculate distance of all points to centre
	dist=torch.sqrt((X-cx)**2+(Y-cy)**2)
	# Assign value of 1 to those points where dist<cr:
	temp[torch.where(dist<r)]=1
	temp = torch.unsqueeze(temp,0)

	return temp

def gen_linear_sink():
	temp =  torch.zeros(1, 64, 64)
	# starting pixel of source (idx 0 to 61) with minimum length 2
	start = torch.randint(low=0, high=62, size=(1,))
	# length of linear source
	length = torch.randint(low=2, high=int(64-start), size=(1,))
	# column position of source
	pos = torch.randint(low=32, high=62, size=(1,))
	width = torch.randint(low=2, high=int(64-pos), size=(1,))
	temp[0, start:start+length, pos:pos+width] = 1
	return temp

def gen_linear_src():
	temp =  torch.zeros(1, 64, 64)
	# starting pixel of source (idx 0 to 61) with minimum length 2
	start = torch.randint(low=0, high=62, size=(1,))
	# length of linear source
	length = torch.randint(low=2, high=int(64-start), size=(1,))
	# column position of source
	pos = torch.randint(low=0, high=30, size=(1,))
	width = torch.randint(low=2, high=int(32-pos), size=(1,))
	temp[0, start:start+length, pos:pos+width] = 1
	return temp

# source 1 at source locations and 0 everywhere else
def gen_source(i): 
	source = torch.zeros(16, 1, 64, 64)
	for i in range(source.shape[0]):
		#source[i] = gen_linear_src()
		source[i] = gen_ellipse_src(i)
	#source = source.to(device)
	return source

# sink 1 at sink locations and 0 everywhere else
def gen_sink(): 
	sink = torch.zeros(16, 1, 64, 64)
	for i in range(sink.shape[0]):
		sink[i] = gen_linear_sink()
		#sink[i] = gen_ellipse_sink()
	#sink = sink.to(device)
	return sink

def gen_mass():
	mass = torch.ones([16, 1, 64, 64], dtype=torch.float32)
	#mass = mass.to(device)
	return mass

@hydra.main(config_path="../conf", config_name="config")
def train(args: DictConfig):
	pde = pde_dict['linear_poisson'](args) #object of Linear Poisson Class
	opt = torch.optim.Adam(pde.network.parameters(), lr=3e-4)
	losses = []
	epochs = 4000
	for i in range(epochs):
		source = gen_source(i)
		sink = gen_sink()
		mass = gen_mass()

		in_data = torch.cat((source, sink, mass), axis=1)

		in_data = in_data.type_as(next(pde.network.parameters()))
		opt.zero_grad()
		u = pde.network(in_data) 

		# Implement the BC conditions from source and sink
		# notation: source will have u = 1 at source and u=0 at sink
		nu_4d_tensor_bc = mass

		# applying bc to solution
		u_bc = (u - u*source + source)*(1-sink)
		# zero forcing function
		forcing_4d_tensor_bc = torch.zeros_like(nu_4d_tensor_bc)

		# optimize pde_network
		loss = pde.loss(u_bc, nu_4d_tensor_bc, forcing_4d_tensor_bc).mean()
		losses.append(loss)
		print(loss)
		loss.backward()
		opt.step()

		if i%100==0 or i==(epochs-1):
			plt.figure()
			plt.imshow(source[0][0].detach().cpu().numpy())
			plt.title('Source')
			plt.savefig(str(i) + '_src.png')
			plt.close()

			plt.figure()
			plt.imshow(sink[0][0].detach().cpu().numpy())
			plt.title('Sink')
			plt.savefig(str(i) + '_snk.png')
			plt.close()

			# plt.figure()
			# plt.imshow(mass[0][0].detach().cpu().numpy())
			# plt.title('Mass')
			# plt.savefig(str(i) + '_mass.png')
			# plt.close()

			plt.figure()
			plt.imshow(u_bc[0][0].detach().cpu().numpy(), cmap='jet')#, norm=LogNorm())
			plt.title('Solution')
			plt.colorbar()
			plt.savefig(str(i) + '_u.png')
			plt.close()

	plt.figure()
	plt.plot(losses)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.savefig('Loss.png')

if __name__ == '__main__':
	train()