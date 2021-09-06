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
QUERY_SCRIPT="query.py"

# example
time $(python ${QUERY_SCRIPT} >out_query.txt 2>&1 -m /work/baskarg/bkhara/diffnet/examples/poisson/parametric/klsum_32/version_15)
