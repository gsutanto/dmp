#!/bin/bash

matlab_funcs_src_folder=$'../../../MATLAB_codes/'

matlab_funcs=$'compute_feature\ncompute_rotation_axes\ncompute_theta\nplot_learn_obs_avoid_coupling_term\nplot_learn_obs_avoid_feature_trajectory\ncompute_X_properties\nlearn_obs_avoid'

for matlab_func in $matlab_funcs
do
	ln -f -s "$matlab_funcs_src_folder$matlab_func.m" "$matlab_func.m"
done
