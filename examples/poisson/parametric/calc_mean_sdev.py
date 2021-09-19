import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import argparse

def calc_mean_stddev(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # var = (X-mean)**2
    # var = np.mean(var, axis=0)
    # std = np.sqrt(var)
    return mean, std

def calc_mean_stddev_from_file():
    print("cwd = ", os.getcwd())
    q_all = np.load('./q_all.npy')
    q_mean, q_sdev = calc_mean_stddev(q_all)
    np.save(os.path.join('q_mean.npy'), q_mean)
    np.save(os.path.join('q_sdev.npy'), q_sdev)

if __name__=="__main__":
    calc_mean_stddev_from_file()