#!/bin/bash

matlab_funcs_src_folder=$'../learn_obs_avoid_vicon_data/'

matlab_funcs=$'visualizeSetting'

for matlab_func in $matlab_funcs
do
	ln -f -s "$matlab_funcs_src_folder$matlab_func.m" "$matlab_func.m"
done
