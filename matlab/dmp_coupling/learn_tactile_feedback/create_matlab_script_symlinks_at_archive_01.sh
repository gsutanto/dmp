#!/bin/bash

matlab_archive_folder="$HOME/Desktop/archives_learn_tactile_fb/"

matlab_scripts=$'analyze_robot_unroll_data'

for matlab_script in $matlab_scripts
do
	ln -f -s "$PWD/$matlab_script.m" "$matlab_archive_folder$matlab_script.m"
done
