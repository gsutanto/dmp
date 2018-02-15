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

is_converting_demo_to_supervised_obs_avoid_fb_dataset = False
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

N_settings = len(data_global_coord["obs_avoid"][0])

subset_settings_indices = range(N_settings)

considered_subset_outlier_ranked_demo_indices = range(3)
generalization_subset_outlier_ranked_demo_indices = [3]
post_filename_stacked_data = '_recur_Ct_dataset'
out_data_dir = '../tf/input_data/'

[X, diff_Ct_target, 
 normalized_phase_PSI_mult_phase_V,
 data_point_priority,
 Ct_t_minus_1_times_dt_per_tau,
 Ct_t_minus_1] = prepareRecurCtData(task_type, dataset_Ct_obs_avoid, 
                                    subset_settings_indices,
                                    considered_subset_outlier_ranked_demo_indices,
                                    generalization_subset_outlier_ranked_demo_indices,
                                    post_filename_stacked_data,
                                    out_data_dir)