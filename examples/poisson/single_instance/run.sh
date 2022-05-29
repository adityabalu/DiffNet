#!/usr/bin/env bash

#######################################################################
source /work/baskarg/bkhara/python_virtual_envs/lightning/bin/activate

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1"  # specify which GPU(s) to be used

echo "Launch time = `date +"%T"`"
echo "Working directory = ${CASE_DIR}"
#######################################################################
########## NO CHANGE ABOVE ############################################
#######################################################################

# TRAIN_SCRIPT="12_klsum.py"
# TRAIN_SCRIPT="12_klsum_fdm.py"
# TRAIN_SCRIPT="10_manufactured_strong_form_higher_order.py"
# TRAIN_SCRIPT="12_fdm_mms.py"
# TRAIN_SCRIPT="13_klsum_network.py"
# TRAIN_SCRIPT="13_klsum_network_fdm.py"
# TRAIN_SCRIPT="12_klsum_fdm_nbc.py"
# TRAIN_SCRIPT="14_helmholtz_mms.py"
# TRAIN_SCRIPT="14_helmholtz_ddelta.py"
# TRAIN_SCRIPT="2_manufactured.py"
# TRAIN_SCRIPT="2_manufactured_network.py"
# TRAIN_SCRIPT="2_manufactured_nonzeroBC_network.py"
# TRAIN_SCRIPT="2_nz_temp.py"
# TRAIN_SCRIPT="e2_manufactured_resmin.py"
# TRAIN_SCRIPT="e3_st_mms_resmin.py"
# TRAIN_SCRIPT="e12_klsum_resmin.py"
# TRAIN_SCRIPT="e17_adv_diff_1d_resmin.py"
# TRAIN_SCRIPT="e17_adv_diff_2d_resmin.py"
# TRAIN_SCRIPT="e17_adv_diff_2d_resmin_network.py"
# TRAIN_SCRIPT="e18_allen_cahn_ice_melt.py"
TRAIN_SCRIPT="pc_complex_immersed_background.py"
# TRAIN_SCRIPT="pc_poisson_disk.py"
time python ${TRAIN_SCRIPT} #>out_train.txt 2>&1
