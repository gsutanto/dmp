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
import keyboard
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_coupling/utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from convertDemoToSupervisedObsAvoidFbDataset import *
from DataStacking import *
from utilities import *

is_converting_demo_to_supervised_obs_avoid_fb_dataset = False
is_comparing_w_MATLAB_implementation = False
task_type = 'obs_avoid'

if (is_converting_demo_to_supervised_obs_avoid_fb_dataset):
    [data_global_coord, 
     dmp_baseline_params, 
     ccdmp_baseline_unroll_global_traj, 
     dataset_Ct_obs_avoid, 
     min_num_considered_demo] = convertDemoToSupervisedObsAvoidFbDataset()
else:
    data_global_coord = loadObj('data_multi_demo_vicon_static_global_coord.pkl')
    dmp_baseline_params = loadObj('dmp_baseline_params_' + task_type + '.pkl')
    ccdmp_baseline_unroll_global_traj = loadObj('ccdmp_baseline_unroll_global_traj.pkl')
    dataset_Ct_obs_avoid = loadObj('dataset_Ct_' + task_type + '.pkl')

if (is_comparing_w_MATLAB_implementation):
    N_settings = len(data_global_coord["obs_avoid"][0])
    selected_settings_indices = range(N_settings)
else:
    selected_settings_indices_file_path = '../tf/models/selected_settings_indices.txt'
    assert (os.path.isfile(selected_settings_indices_file_path)), selected_settings_indices_file_path + ' file does NOT exist!'
    selected_settings_indices = [(i-1) for i in list(np.loadtxt(selected_settings_indices_file_path, dtype=np.int, ndmin=1))] # file is saved following MATLAB's convention (1~222)
    N_settings = len(selected_settings_indices)

print('N_settings = ' + str(N_settings))

considered_subset_outlier_ranked_demo_indices = range(3)
generalization_subset_outlier_ranked_demo_indices = [3]
post_filename_stacked_data = ''
out_data_dir = '../tf/input_data/'

[X, Ct_target, 
 normalized_phase_PSI_mult_phase_V,
 data_point_priority] = prepareData(task_type, dataset_Ct_obs_avoid, 
                                    selected_settings_indices,
                                    considered_subset_outlier_ranked_demo_indices,
                                    generalization_subset_outlier_ranked_demo_indices,
                                    post_filename_stacked_data,
                                    out_data_dir)

if is_comparing_w_MATLAB_implementation:
    # Some comparison with the result of MATLAB's implementation of prepareData() function (for implementation verification):
    generalization_test_id_string = ''
    generalization_test_sub_path = ''
    prim_no = 1
    
    mX = sio.loadmat('../tf/input_data/'+generalization_test_sub_path+'prim_'+str(prim_no)+'_X_raw_obs_avoid'+generalization_test_id_string+'.mat', struct_as_record=True)['X'].astype(np.float32)
    mCt_target = sio.loadmat('../tf/input_data/'+generalization_test_sub_path+'prim_'+str(prim_no)+'_Ct_target_obs_avoid'+generalization_test_id_string+'.mat', struct_as_record=True)['Ct_target'].astype(np.float32)
    mnormalized_phase_kernels = sio.loadmat('../tf/input_data/'+generalization_test_sub_path+'prim_'+str(prim_no)+'_normalized_phase_PSI_mult_phase_V_obs_avoid'+generalization_test_id_string+'.mat', struct_as_record=True)['normalized_phase_PSI_mult_phase_V'].astype(np.float32)
    mdata_point_priority = sio.loadmat('../tf/input_data/'+generalization_test_sub_path+'prim_'+str(prim_no)+'_data_point_priority_obs_avoid'+generalization_test_id_string+'.mat', struct_as_record=True)['data_point_priority'].astype(np.float32)
    
    mX_generalization_test = sio.loadmat('../tf/input_data/'+generalization_test_sub_path+'test_unroll_prim_'+str(prim_no)+'_X_raw_obs_avoid'+generalization_test_id_string+'.mat', struct_as_record=True)['X'].astype(np.float32)
    mCtt_generalization_test = sio.loadmat('../tf/input_data/'+generalization_test_sub_path+'test_unroll_prim_'+str(prim_no)+'_Ct_target_obs_avoid'+generalization_test_id_string+'.mat', struct_as_record=True)['Ct_target'].astype(np.float32)
    mnPSI_generalization_test = sio.loadmat('../tf/input_data/'+generalization_test_sub_path+'test_unroll_prim_'+str(prim_no)+'_normalized_phase_PSI_mult_phase_V_obs_avoid'+generalization_test_id_string+'.mat', struct_as_record=True)['normalized_phase_PSI_mult_phase_V'].astype(np.float32)
    mW_generalization_test = sio.loadmat('../tf/input_data/'+generalization_test_sub_path+'test_unroll_prim_'+str(prim_no)+'_data_point_priority_obs_avoid'+generalization_test_id_string+'.mat', struct_as_record=True)['data_point_priority'].astype(np.float32)
    
    print ('Comparing between Python and MATLAB Implementation Results: (no message will be printed out if similar...)')
    compareTwoMatrices(X[0][0], mX, 1.02e-4)
    compareTwoMatrices(Ct_target[0][0], mCt_target, 4.5e-3)
    compareTwoMatrices(normalized_phase_PSI_mult_phase_V[0][0], mnormalized_phase_kernels)
    compareTwoMatrices(data_point_priority[0][0], mdata_point_priority)
    
    compareTwoMatrices(X[1][0], mX_generalization_test, 1.01e-4)
    compareTwoMatrices(Ct_target[1][0], mCtt_generalization_test, 4.2e-3)
    compareTwoMatrices(normalized_phase_PSI_mult_phase_V[1][0], mnPSI_generalization_test)
    compareTwoMatrices(data_point_priority[1][0], mW_generalization_test)