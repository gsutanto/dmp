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
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_coupling/utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from convertDemoToSupervisedObsAvoidFbDataset import *
from DataStacking import *
from utilities import *


data_global_coord = loadObj('data_multi_demo_vicon_static_global_coord.pkl')
dataset_Ct_obs_avoid = loadObj('dataset_Ct_obs_avoid.pkl')
unroll_dataset_Ct_obs_avoid = loadObj('unroll_dataset_Ct_obs_avoid.pkl')
#unroll_dataset_Ct_obs_avoid = loadObj('unroll_dataset_Ct_obs_avoid_recur_Ct_dataset.pkl')

model_parent_dir_path = '../tf/models/'

selected_settings_indices_file_path = model_parent_dir_path + 'selected_settings_indices.txt'
if not os.path.isfile(selected_settings_indices_file_path):
    N_settings = len(data_global_coord["obs_avoid"][0])
    selected_settings_indices = range(N_settings)
else:
    selected_settings_indices = [(i-1) for i in list(np.loadtxt(selected_settings_indices_file_path, dtype=np.int, ndmin=1))] # file is saved following MATLAB's convention (1~222)
    N_settings = len(selected_settings_indices)

print('N_settings = ' + str(N_settings))

subset_settings_indices = selected_settings_indices
subset_demos_indices = range(1)
mode_stack_dataset = 2
feature_type = 'raw'
N_primitive = 1

for prim_no in range(N_primitive):
    [_,
     Ct_target,
     _,
     _] = stackDataset(dataset_Ct_obs_avoid, 
                       subset_settings_indices, 
                       mode_stack_dataset, 
                       subset_demos_indices, 
                       feature_type, 
                       prim_no)
    [_,
     Ct_unroll,
     _,
     _] = stackDataset(unroll_dataset_Ct_obs_avoid, 
                       subset_settings_indices, 
                       mode_stack_dataset, 
                       subset_demos_indices, 
                       feature_type, 
                       prim_no)
    nmse_unroll = computeNMSE(Ct_unroll, Ct_target)
    print ('nmse_unroll        = ' + str(nmse_unroll))