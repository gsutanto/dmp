#!/bin/bash

target_dir="../plot/dmp_coupling/learn_obs_avoid/feature_trajectory/static_obs/single_baseline/multi_demo/regularization_comparison"

if [ -d "$target_dir" ]; then
	rm -rf "$target_dir"
fi
mkdir "$target_dir"

i=1
for regularization_const in 1e-4 1e-3 1e-2 1e-1 1.0 5.0
do
	echo "regularization_const = $regularization_const"
	mkdir "$target_dir/$i/"
	echo "$regularization_const" > "$target_dir/$i/regularization_const.txt"
	`../demos/amd_clmc_dmp_dc_loa_so_sb_multi_demo_demo -f 0 -c 2 -r $regularization_const -o $target_dir/$i/ -e ../rt_errors/rt_err.txt`
	let "i=i+1"
done
