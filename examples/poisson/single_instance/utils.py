import os
from functools import wraps
from time import time
import numpy as np
import matplotlib.pyplot as plt

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
	plt.plot(epoch, np.log10(loss))
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.savefig(os.path.join(casepath, 'lossplot.png'))
