#!/bin/bash

matlab_funcs_src_folder=$'../../../MATLAB_codes/'

matlab_funcs=$'compute_feature\ncompute_rotation_axes\ncompute_theta\nplot_learn_obs_avoid_coupling_term\nplot_learn_obs_avoid_feature_trajectory\ncompute_X_properties\nlearn_obs_avoid\nplot_loa_Ct_target_vs_Ct_fit_per_setting\nplot_loa_static_obs_single_baseline_multi_demo_trajectories\nplot_loa_static_obs_single_baseline_multi_demo_unrolling\nplot_sphere\nvarycolor'

for matlab_func in $matlab_funcs
do
	ln -f -s "$matlab_funcs_src_folder$matlab_func.m" "$matlab_func.m"
done
