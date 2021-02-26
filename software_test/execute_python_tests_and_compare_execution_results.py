#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import copy
dmp_home_dir_abs_path = os.path.dirname(__file__) + "/../"
dmp_software_test_dir_abs_path = dmp_home_dir_abs_path + '/software_test/'
assert (os.path.isdir(dmp_home_dir_abs_path + '/python/dmp_test/dmp_1D/'))
assert (os.path.isdir(dmp_home_dir_abs_path + '/python/dmp_test/cart_dmp/cart_coord_dmp/'))
assert (os.path.isdir(dmp_home_dir_abs_path + '/python/dmp_test/dmp_coupling/learn_obs_avoid/static_obs/single_baseline/'))
sys.path.append(os.path.join(dmp_home_dir_abs_path + '/python/dmp_test/dmp_1D/'))
sys.path.append(os.path.join(dmp_home_dir_abs_path + '/python/dmp_test/cart_dmp/cart_coord_dmp/'))
sys.path.append(os.path.join(dmp_home_dir_abs_path + '/python/dmp_test/cart_dmp/quat_dmp/'))
sys.path.append(os.path.join(dmp_home_dir_abs_path + '/python/dmp_test/dmp_coupling/learn_obs_avoid/static_obs/single_baseline/'))
sys.path.append(os.path.join(dmp_home_dir_abs_path + '/python/utilities/'))
from dmp_1D_test import *
from cart_coord_dmp_single_traj_training_test import *
from cart_coord_dmp_multi_traj_training_test import *
from quat_dmp_single_traj_training_test import *
from quat_dmp_multi_traj_training_test import *
# from ct_loa_so_sb_multi_demo_vicon_PMNN_unrolling_test import *
from utilities import *

dmp_1D_test(dmp_home_dir_abs_path, 2, 0.0, 0.0, 0.0, 0.0, dmp_software_test_dir_abs_path, "test_python_dmp_1D_test_0_2_0.txt")
dmp_1D_test(dmp_home_dir_abs_path, 1, 0.0, 0.0, 0.0, 0.0, dmp_software_test_dir_abs_path, "test_python_dmp_1D_test_0_1_0.txt")
dmp_1D_test(dmp_home_dir_abs_path, 2, 6.0, 1.0, 10.0, 4.0, dmp_software_test_dir_abs_path, "test_python_dmp_1D_test_0_2_0_6.0_1.0_10.0_4.0.txt")
compareTwoNumericFiles(dmp_software_test_dir_abs_path+'/result_dmp_1D_test_0_2_0.txt', 
                        dmp_software_test_dir_abs_path+'/test_python_dmp_1D_test_0_2_0.txt')
compareTwoNumericFiles(dmp_software_test_dir_abs_path+'/result_dmp_1D_test_0_1_0.txt', 
                        dmp_software_test_dir_abs_path+'/test_python_dmp_1D_test_0_1_0.txt')
compareTwoNumericFiles(dmp_software_test_dir_abs_path+'/result_dmp_1D_test_0_2_0_6.0_1.0_10.0_4.0.txt', 
                        dmp_software_test_dir_abs_path+'/test_python_dmp_1D_test_0_2_0_6.0_1.0_10.0_4.0.txt')

cart_coord_dmp_single_traj_training_test(dmp_home_dir_abs_path, 2, 0.0, 0.0, 0.0, dmp_software_test_dir_abs_path, "test_python_cart_coord_dmp_single_traj_training_test_0_2.txt")
cart_coord_dmp_single_traj_training_test(dmp_home_dir_abs_path, 1, 0.0, 0.0, 0.0, dmp_software_test_dir_abs_path, "test_python_cart_coord_dmp_single_traj_training_test_0_1.txt")
cart_coord_dmp_single_traj_training_test(dmp_home_dir_abs_path, 2, 6.0, 1.0, 6.0, dmp_software_test_dir_abs_path, "test_python_cart_coord_dmp_single_traj_training_test_0_2_6.0_1.0_6.0.txt")
cart_coord_dmp_single_traj_training_test(dmp_home_dir_abs_path, 2, 6.0, 2.0, 6.0, dmp_software_test_dir_abs_path, "test_python_cart_coord_dmp_single_traj_training_test_0_2_6.0_2.0_6.0.txt")
compareTwoNumericFiles(dmp_software_test_dir_abs_path+'/result_cart_coord_dmp_single_traj_training_test_0_2.txt', 
                        dmp_software_test_dir_abs_path+'/test_python_cart_coord_dmp_single_traj_training_test_0_2.txt')
compareTwoNumericFiles(dmp_software_test_dir_abs_path+'/result_cart_coord_dmp_single_traj_training_test_0_1.txt', 
                        dmp_software_test_dir_abs_path+'/test_python_cart_coord_dmp_single_traj_training_test_0_1.txt')
compareTwoNumericFiles(dmp_software_test_dir_abs_path+'/result_cart_coord_dmp_single_traj_training_test_0_2_6.0_1.0_6.0.txt', 
                        dmp_software_test_dir_abs_path+'/test_python_cart_coord_dmp_single_traj_training_test_0_2_6.0_1.0_6.0.txt')
compareTwoNumericFiles(dmp_software_test_dir_abs_path+'/result_cart_coord_dmp_single_traj_training_test_0_2_6.0_2.0_6.0.txt', 
                        dmp_software_test_dir_abs_path+'/test_python_cart_coord_dmp_single_traj_training_test_0_2_6.0_2.0_6.0.txt')

cart_coord_dmp_multi_traj_training_test(dmp_home_dir_abs_path, 2, 0.5, 0.5, dmp_software_test_dir_abs_path, "test_python_cart_coord_dmp_multi_traj_training_test_0_2.txt")
compareTwoNumericFiles(dmp_software_test_dir_abs_path+'/result_cart_coord_dmp_multi_traj_training_test_0_2.txt', 
                        dmp_software_test_dir_abs_path+'/test_python_cart_coord_dmp_multi_traj_training_test_0_2.txt')

quat_dmp_single_traj_training_test(dmp_home_dir_abs_path, 1, 1.9976, 1.9976, dmp_software_test_dir_abs_path, 'test_python_quat_dmp_single_traj_training_test_0_1.txt')
quat_dmp_single_traj_training_test(dmp_home_dir_abs_path, 2, 1.9976, 1.9976, dmp_software_test_dir_abs_path, 'test_python_quat_dmp_single_traj_training_test_0_2.txt')
compareTwoNumericFiles(dmp_software_test_dir_abs_path+'/result_quat_dmp_single_traj_training_test_0_1.txt', 
                        dmp_software_test_dir_abs_path+'/test_python_quat_dmp_single_traj_training_test_0_1.txt')
compareTwoNumericFiles(dmp_software_test_dir_abs_path+'/result_quat_dmp_single_traj_training_test_0_2.txt', 
                        dmp_software_test_dir_abs_path+'/test_python_quat_dmp_single_traj_training_test_0_2.txt', 2.301e-5)

quat_dmp_multi_traj_training_test(dmp_home_dir_abs_path, 2, 1.9976, 1.9976, dmp_software_test_dir_abs_path, 'test_python_quat_dmp_multi_traj_training_test_0_2.txt', 
                                  False, None, None, None, None, dmp_software_test_dir_abs_path, 'test_python_quat_dmp_multi_traj_training_test_0_2_learned_params.txt')
compareTwoNumericFiles(dmp_software_test_dir_abs_path+'/result_quat_dmp_multi_traj_training_test_0_2.txt', 
                        dmp_software_test_dir_abs_path+'/test_python_quat_dmp_multi_traj_training_test_0_2.txt', 1.251e-5)
compareTwoNumericFiles(dmp_software_test_dir_abs_path+'/result_quat_dmp_multi_traj_training_test_0_2_learned_params.txt', 
                        dmp_software_test_dir_abs_path+'/test_python_quat_dmp_multi_traj_training_test_0_2_learned_params.txt', 7.501e-5, 1.5e-3, True)

print("Testing Learning QuaternionDMP from Smoothed Quaternion Trajectory...")
quat_dmp_multi_traj_training_test(dmp_home_dir_abs_path, 2, 1.9976, 1.9976, dmp_software_test_dir_abs_path, 'test_python_quat_dmp_multi_smoothed_traj_training_test_0_2.txt', 
                                  True, 1.5, 3.0, 3, 5.0, dmp_software_test_dir_abs_path, 'test_python_quat_dmp_multi_smoothed_traj_training_test_0_2_learned_params.txt')
compareTwoNumericFiles(dmp_software_test_dir_abs_path+'/result_quat_dmp_multi_smoothed_traj_training_test_0_2.txt', 
                        dmp_software_test_dir_abs_path+'/test_python_quat_dmp_multi_smoothed_traj_training_test_0_2.txt')
compareTwoNumericFiles(dmp_software_test_dir_abs_path+'/result_quat_dmp_multi_smoothed_traj_training_test_0_2_learned_params.txt', 
                        dmp_software_test_dir_abs_path+'/test_python_quat_dmp_multi_smoothed_traj_training_test_0_2_learned_params.txt', 1.001e-5, 7.5e-4, True)

# ct_loa_so_sb_multi_demo_vicon_PMNN_unrolling_test(dmp_home_dir_abs_path, 2, dmp_software_test_dir_abs_path, "test_python_ct_loa_so_sb_multi_demo_vicon_PMNN_unrolling_test.txt")
# compareTwoNumericFiles(dmp_software_test_dir_abs_path+'/result_ct_loa_so_sb_multi_demo_vicon_PMNN_unrolling_test.txt', 
#                         dmp_software_test_dir_abs_path+'/test_python_ct_loa_so_sb_multi_demo_vicon_PMNN_unrolling_test.txt',
#                         scalar_max_abs_diff_threshold=1.6e-5)

print("execute_python_tests_and_compare_execution_results.py script execution done!")
