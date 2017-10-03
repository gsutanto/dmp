#!/bin/bash

matlab_funcs_src_folder=$'../../dmp_general/'

matlab_funcs=$'dcp_franzi\ncomputeDMPCtTarget'

for matlab_func in $matlab_funcs
do
	ln -f -s "$matlab_funcs_src_folder$matlab_func.m" "$matlab_func.m"
done
