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
TRAIN_SCRIPT="2_klsum_fem.py"

samplesize=4096
sed -i "138s/.*/    sample_size = ${samplesize}/" ${TRAIN_SCRIPT}
echo "Launch time = `date +"%T"`"
time python ${TRAIN_SCRIPT} >out_train_${samplesize}.txt 2>&1

samplesize=8192
sed -i "138s/.*/    sample_size = ${samplesize}/" ${TRAIN_SCRIPT}
echo "Launch time = `date +"%T"`"
time python ${TRAIN_SCRIPT} >out_train_${samplesize}.txt 2>&1

samplesize=16384
sed -i "138s/.*/    sample_size = ${samplesize}/" ${TRAIN_SCRIPT}
echo "Launch time = `date +"%T"`"
time python ${TRAIN_SCRIPT} >out_train_${samplesize}.txt 2>&1

samplesize=32768
sed -i "138s/.*/    sample_size = ${samplesize}/" ${TRAIN_SCRIPT}
echo "Launch time = `date +"%T"`"
time python ${TRAIN_SCRIPT} >out_train_${samplesize}.txt 2>&1

samplesize=65536
sed -i "138s/.*/    sample_size = ${samplesize}/" ${TRAIN_SCRIPT}
echo "Launch time = `date +"%T"`"
time python ${TRAIN_SCRIPT} >out_train_${samplesize}.txt 2>&1

