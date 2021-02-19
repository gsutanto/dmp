#!/bin/bash

dirs_src_folder=$'../../../../../dmp_data/dmp_coupling/learn_obs_avoid/static_obs/'

dirs=$'data_multi_demo_static\ndata_reinforcement_learning\ndata_sph_new'

for dir in $dirs
do
	ln -f -s "$dirs_src_folder$dir" "$dir"
done
