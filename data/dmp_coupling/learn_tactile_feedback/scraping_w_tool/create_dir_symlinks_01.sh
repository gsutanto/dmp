#!/bin/bash

dirs_src_folder=$'../../../../../amd_clmc_dmp_data/dmp_coupling/learn_tactile_feedback/scraping_w_tool/'

dirs=$'baseline\nhuman_baseline'
max_numbered_dir=8

for dir in $dirs
do
	ln -f -s "$dirs_src_folder$dir" "$dir"
done

for dir in $(seq 1 $max_numbered_dir)
do
        ln -f -s "$dirs_src_folder$dir" "$dir"
done

