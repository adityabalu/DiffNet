import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import argparse
from DiffNet.vti_writer import *

def vti_from_npy(args):
    ifile = args.ifilepath
    filename_with_ext = os.path.basename(ifile)
    filename = os.path.splitext(filename_with_ext)[0]
    ofile = os.path.join('.', (filename + '.vti'))
    print("Inp file = ", ifile)
    print("Out file = ", ofile)

    mat = np.load(ifile)
    u = mat.flatten()

    # geometry info
    nsd = 2

    nx = mat.shape[0]
    ny = mat.shape[1]
    nz = 100

    ## =========================================================================
    # THIS BLOCK IS NON-INTERVENTION
    # adjustments for 2d
    dx = 1. / (nx - 1)
    dy = 1. / (ny - 1)
    dz = 1. / (nz - 1)

    if nsd == 2:
        nz = 1
        dz = 0.

    p0 = [0, 0, 0]
    p1 = [nx - 1,ny - 1, nz - 1]
    originV = [0, 0, 0]
    v_dx = [dx, dy, dz]

    # this assertion should happen HERE, not before
    # assert(u.shape[0] == nx*ny*nz), "Mismatch: (Nx,Ny,Nz)=(%r,%r,%r) AND shape(array) = %r" %(nx,ny,nz,u.shape[0])
    ## =========================================================================

    vtiwriter = vtiWriter(p0, p1, originV, v_dx)
    vtiwriter.vti_from_vector(ofile, u, False, 'u')

# or write them combined, in a single file
# vtiwriter.vti_from_multiple_vector('combined.vti', [k,u], [False,False], ['k','u'])

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ifilepath', '-ip', help='Input filename', required=True)
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    vti_from_npy(args)

if __name__ == "__main__":
    # filename = str(sys.argv[1])
    # eta_x = float(sys.argv[2])
    # eta_y = float(sys.argv[3])
    # print(filename)
    # chidx = int(sys.argv[1])
    main()
