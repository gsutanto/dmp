#!/bin/bash

matlab_funcs_src_folder=$'../utilities/'

matlab_funcs=$'prepareData\nstackDataset'

for matlab_func in $matlab_funcs
do
	ln -f -s "$matlab_funcs_src_folder$matlab_func.m" "$matlab_func.m"
done
