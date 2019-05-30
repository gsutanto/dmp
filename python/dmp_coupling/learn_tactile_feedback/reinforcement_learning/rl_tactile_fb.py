#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:00:00 2019

@author: gsutanto
"""

import os
import sys
import re
import numpy as np
import numpy.linalg as npla
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/clmcplot/'))
import utilities as py_util
import clmcplot_utils as clmcplot_util

catkin_ws_path = py_util.getCatkinWSPath()
sl_data_path = catkin_ws_path + "/install/bin/arm"

is_deleting_all_dfiles_at_init = False#True
N_total_sense_dimensionality = 45
N_primitive = 3

if (is_deleting_all_dfiles_at_init):
    # initialization by removing all SL data files inside sl_data_path
    initial_dfilepaths = [sl_data_path+'/'+f for f in os.listdir(sl_data_path) if re.match(r'd+\d{5,5}$', f)] # match the regex pattern of SL data files
    for initial_dfilepath in initial_dfilepaths:
        os.remove(initial_dfilepath)

all_trial_prim_Rewards_list = list()
init_new_env_dfilepaths = [sl_data_path+'/'+f for f in os.listdir(sl_data_path) if re.match(r'd+\d{5,5}$', f)] # match the regex pattern of SL data files
for init_new_env_dfilepath in init_new_env_dfilepaths:
    print("Processing datafile %s..." % init_new_env_dfilepath)
    clmcfile = clmcplot_util.ClmcFile(init_new_env_dfilepath)
    trial_prim_id = clmcfile.get_variables(["ul_curr_prim_no"])[0].T
    
    trial_X_vector = np.vstack([clmcfile.get_variables(["X_vector_%02d" % i for i in range(N_total_sense_dimensionality)])]).T
    assert(trial_X_vector.shape[0] == trial_prim_id.shape[0])
    assert(trial_X_vector.shape[1] == N_total_sense_dimensionality)
    
    trial_prim_indices = list()
    trial_prim_X_vectors = list()
    trial_prim_Rewards = list()
    for ip in range(N_primitive):
        trial_prim_indices.append(np.where(trial_prim_id == ip)[0])
        trial_prim_X_vectors.append(trial_X_vector[trial_prim_indices[ip],:])
        trial_prim_Rewards.append(-npla.norm(trial_prim_X_vectors[ip], ord=2))
    all_trial_prim_Rewards_list.append(trial_prim_Rewards)
all_trial_prim_Rewards = np.vstack(all_trial_prim_Rewards_list)
mean_prim_Rewards = np.mean(all_trial_prim_Rewards, axis=0)