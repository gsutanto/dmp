#!/bin/bash

dirs_src_folder=$'../../../../dmp_data/dmp_coupling/learn_tactile_feedback/'

dirs=$'peg_in_hole_big_cone\nscrewing\nunscrewing'

for dir in $dirs
do
	ln -f -s "$dirs_src_folder$dir" "$dir"
done
