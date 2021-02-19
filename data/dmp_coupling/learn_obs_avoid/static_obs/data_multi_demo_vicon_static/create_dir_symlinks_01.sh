#!/bin/bash

dirs_src_folder=$'../../../../../../dmp_data/dmp_coupling/learn_obs_avoid/static_obs/data_multi_demo_vicon_static/'

dirs=$'baseline'
max_numbered_dir=222

for dir in $dirs
do
	ln -f -s "$dirs_src_folder$dir" "$dir"
done

for dir in $(seq 1 $max_numbered_dir)
do
        ln -f -s "$dirs_src_folder$dir" "$dir"
done

