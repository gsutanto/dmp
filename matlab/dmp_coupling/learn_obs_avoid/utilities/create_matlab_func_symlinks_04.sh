#!/bin/bash

matlab_funcs_src_folder=$'../../../cart_dmp/cart_coord_dmp/'

matlab_funcs=$'learnCartPrimitiveMultiOnLocalCoord\nunrollCartPrimitiveOnLocalCoord\ncompleteCartCoordDMPParams\nextractSetCartCoordTrajectories\nextractCartCoordTrajectory'

for matlab_func in $matlab_funcs
do
	ln -f -s "$matlab_funcs_src_folder$matlab_func.m" "$matlab_func.m"
done
