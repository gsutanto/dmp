#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import numpy as np
import os
import sys
import copy
with open ("../software_test/amd_clmc_dmp_software_test_dir_abs_path.txt", "r") as myfile:
    amd_clmc_dmp_software_test_dir_abs_path = myfile.read().splitlines()[0]
amd_clmc_dmp_home_dir_abs_path = amd_clmc_dmp_software_test_dir_abs_path + "/../"
devel_software_test_dir_abs_path = os.getcwd() + '/../software_test/'
assert (os.path.isdir(amd_clmc_dmp_software_test_dir_abs_path + '/../python/dmp_test/dmp_1D/'))
assert (os.path.isdir(amd_clmc_dmp_software_test_dir_abs_path + '/../python/dmp_test/cart_dmp/cart_coord_dmp/'))
assert (os.path.isdir(amd_clmc_dmp_software_test_dir_abs_path + '/../python/dmp_test/dmp_coupling/learn_obs_avoid/static_obs/single_baseline/'))
sys.path.append(os.path.join(os.path.dirname(__file__), amd_clmc_dmp_software_test_dir_abs_path + '/../python/dmp_test/dmp_1D/'))
sys.path.append(os.path.join(os.path.dirname(__file__), amd_clmc_dmp_software_test_dir_abs_path + '/../python/dmp_test/cart_dmp/cart_coord_dmp/'))
sys.path.append(os.path.join(os.path.dirname(__file__), amd_clmc_dmp_software_test_dir_abs_path + '/../python/dmp_test/dmp_coupling/learn_obs_avoid/static_obs/single_baseline/'))
sys.path.append(os.path.join(os.path.dirname(__file__), amd_clmc_dmp_software_test_dir_abs_path + '/../python/utilities/'))
from dmp_1D_test import *
from cart_coord_dmp_single_traj_training_test import *
from cart_coord_dmp_multi_traj_training_test import *
from ct_loa_so_sb_multi_demo_vicon_PMNN_unrolling_test import *
from utilities import *

dmp_1D_test(amd_clmc_dmp_home_dir_abs_path, 2, 0.0, 0.0, 0.0, 0.0, devel_software_test_dir_abs_path, "test_python_dmp_1D_test_0_2_0.txt")
dmp_1D_test(amd_clmc_dmp_home_dir_abs_path, 1, 0.0, 0.0, 0.0, 0.0, devel_software_test_dir_abs_path, "test_python_dmp_1D_test_0_1_0.txt")
dmp_1D_test(amd_clmc_dmp_home_dir_abs_path, 2, 6.0, 1.0, 10.0, 4.0, devel_software_test_dir_abs_path, "test_python_dmp_1D_test_0_2_0_6.0_1.0_10.0_4.0.txt")
compareTwoNumericFiles(devel_software_test_dir_abs_path+'/test_cpp_dmp_1D_test_0_2_0.txt', 
                       devel_software_test_dir_abs_path+'/test_python_dmp_1D_test_0_2_0.txt')
compareTwoNumericFiles(devel_software_test_dir_abs_path+'/test_cpp_dmp_1D_test_0_1_0.txt', 
                       devel_software_test_dir_abs_path+'/test_python_dmp_1D_test_0_1_0.txt')
compareTwoNumericFiles(devel_software_test_dir_abs_path+'/result_dmp_1D_test_0_2_0_6.0_1.0_10.0_4.0.txt', 
                       devel_software_test_dir_abs_path+'/test_python_dmp_1D_test_0_2_0_6.0_1.0_10.0_4.0.txt')

cart_coord_dmp_single_traj_training_test(amd_clmc_dmp_home_dir_abs_path, 2, 0.0, 0.0, 0.0, devel_software_test_dir_abs_path, "test_python_cart_coord_dmp_single_traj_training_test_0_2.txt")
cart_coord_dmp_single_traj_training_test(amd_clmc_dmp_home_dir_abs_path, 1, 0.0, 0.0, 0.0, devel_software_test_dir_abs_path, "test_python_cart_coord_dmp_single_traj_training_test_0_1.txt")
cart_coord_dmp_single_traj_training_test(amd_clmc_dmp_home_dir_abs_path, 2, 6.0, 1.0, 6.0, devel_software_test_dir_abs_path, "test_python_cart_coord_dmp_single_traj_training_test_0_2_6.0_1.0_6.0.txt")
cart_coord_dmp_single_traj_training_test(amd_clmc_dmp_home_dir_abs_path, 2, 6.0, 2.0, 6.0, devel_software_test_dir_abs_path, "test_python_cart_coord_dmp_single_traj_training_test_0_2_6.0_2.0_6.0.txt")
compareTwoNumericFiles(devel_software_test_dir_abs_path+'/test_cpp_cart_coord_dmp_single_traj_training_test_0_2.txt', 
                       devel_software_test_dir_abs_path+'/test_python_cart_coord_dmp_single_traj_training_test_0_2.txt')
compareTwoNumericFiles(devel_software_test_dir_abs_path+'/test_cpp_cart_coord_dmp_single_traj_training_test_0_1.txt', 
                       devel_software_test_dir_abs_path+'/test_python_cart_coord_dmp_single_traj_training_test_0_1.txt')
compareTwoNumericFiles(devel_software_test_dir_abs_path+'/result_cart_coord_dmp_single_traj_training_test_0_2_6.0_1.0_6.0.txt', 
                       devel_software_test_dir_abs_path+'/test_python_cart_coord_dmp_single_traj_training_test_0_2_6.0_1.0_6.0.txt')
compareTwoNumericFiles(devel_software_test_dir_abs_path+'/result_cart_coord_dmp_single_traj_training_test_0_2_6.0_2.0_6.0.txt', 
                       devel_software_test_dir_abs_path+'/test_python_cart_coord_dmp_single_traj_training_test_0_2_6.0_2.0_6.0.txt')

cart_coord_dmp_multi_traj_training_test(amd_clmc_dmp_home_dir_abs_path, 2, 0.5, 0.5, devel_software_test_dir_abs_path, "test_python_cart_coord_dmp_multi_traj_training_test_0_2.txt")
compareTwoNumericFiles(devel_software_test_dir_abs_path+'/test_cpp_cart_coord_dmp_multi_traj_training_test_0_2.txt', 
                       devel_software_test_dir_abs_path+'/test_python_cart_coord_dmp_multi_traj_training_test_0_2.txt')

ct_loa_so_sb_multi_demo_vicon_PMNN_unrolling_test(amd_clmc_dmp_home_dir_abs_path, 2, devel_software_test_dir_abs_path, "test_python_ct_loa_so_sb_multi_demo_vicon_PMNN_unrolling_test.txt")
compareTwoNumericFiles(devel_software_test_dir_abs_path+'/test_cpp_ct_loa_so_sb_multi_demo_vicon_PMNN_unrolling_test.txt', 
                       devel_software_test_dir_abs_path+'/test_python_ct_loa_so_sb_multi_demo_vicon_PMNN_unrolling_test.txt',
                       1.56e-5)

print "execute_python_tests_and_compare_w_cpp_tests.py script execution done!"