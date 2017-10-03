#!/bin/bash

# define array
folders_of_interest=("../dmp_coupling/learn_obs_avoid/feature_trajectory/static_obs/single_baseline/single_demo/" "../dmp_coupling/learn_obs_avoid/feature_trajectory/static_obs/single_baseline/multi_demo/" "../dmp_coupling/learn_obs_avoid/tau_invariance_evaluation/")
shortcut_names=("ftraj_loa_static_1bl_1demo" "ftraj_loa_static_1bl_mdemo" "tau_inv_loa_static_1bl_1demo")
 
# get length of an array
tLen=${#folders_of_interest[@]}
 
# use for loop over the arrays
for (( i=0; i<${tLen}; i++ ));
do
	ln -f -s "${folders_of_interest[$i]}" "${shortcut_names[$i]}"
done
