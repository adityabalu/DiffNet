#!/usr/bin/env bash

#######################################################################
source /work/baskarg/bkhara/python_virtual_envs/lightning/bin/activate

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"  # specify which GPU(s) to be used

echo "Launch time = `date +"%T"`"
echo "Working directory = ${CASE_DIR}"
#######################################################################
########## NO CHANGE ABOVE ############################################
#######################################################################
cd /work/baskarg/bkhara/diffnet/examples/poisson/parametric/klsum_32/version_15; python3 ../../calc_mean_sdev.py
cd /work/baskarg/bkhara/diffnet/examples/poisson/parametric/klsum_32/version_16; python3 ../../calc_mean_sdev.py
cd /work/baskarg/bkhara/diffnet/examples/poisson/parametric/klsum_32/version_17; python3 ../../calc_mean_sdev.py
cd /work/baskarg/bkhara/diffnet/examples/poisson/parametric/klsum_32/version_18; python3 ../../calc_mean_sdev.py
cd /work/baskarg/bkhara/diffnet/examples/poisson/parametric/klsum_32/version_19; python3 ../../calc_mean_sdev.py
