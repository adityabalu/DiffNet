import os
from functools import wraps
from time import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch

def timing(f):
	@wraps(f)
	def wrap(*args, **kwargs):
		ts = time()
		result = f(*args, **kwargs)
		te = time()
		dt = te-ts
		print(f"[TIMING] {f.__name__} took {dt}")
		return result
	return wrap

def plot_losses(casepath, delimiter=','):
	data = np.loadtxt(os.path.join(casepath, 'metrics.csv'), delimiter=delimiter, skiprows=1)
	epoch = data[:,1]
	loss = data[:,0]
	plt.plot(epoch, loss)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.savefig(os.path.join(casepath, 'lossplot.png'))
	plt.close()

	plt.plot(epoch, np.log10(loss))
	plt.xlabel('epoch')
	plt.ylabel('log10(loss)')
	plt.savefig(os.path.join(casepath, 'lossplot_log.png'))
	plt.close()

def load_ilu_data(filepath, extract_transpose=True):
	def extract(fulldata, row_key, col_key, val_key):
		rows = fulldata[row_key].squeeze().astype(np.short)
		cols = fulldata[col_key].squeeze().astype(np.short)
		data = fulldata[val_key].squeeze()
		idx = np.vstack((rows.T, cols.T))
		return torch.sparse_coo_tensor(idx, data, (rows[-1], cols[-1]))

	invLdata = scipy.io.loadmat(filepath)

	# rows = invLdata['rows'].astype(np.short)
	# cols = invLdata['cols'].astype(np.short)
	# data = invLdata['data'].squeeze()
	# idx = np.vstack((rows.T, cols.T))
	# invL = torch.sparse_coo_tensor(idx, data, (rows[-1], cols[-1]))

	# if extract_transpose:
	# 	rows = invLdata['rowst'].astype(np.short)
	# 	cols = invLdata['colst'].astype(np.short)
	# 	data = invLdata['datat'].squeeze()
	# 	idx = np.vstack((rows.T, cols.T))
	# 	invLt = torch.sparse_coo_tensor(idx, data, (rows[-1], cols[-1]))
	
	invL = extract(invLdata,
					'rows',
					'cols',
					'data')
	if extract_transpose:
		invLt = extract(invLdata,
						'rowst',
						'colst',
						'datat')

	return invL, invLt

