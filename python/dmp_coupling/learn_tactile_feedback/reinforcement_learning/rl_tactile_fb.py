#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:00:00 2019

@author: gsutanto
"""

import os
import sys
import numpy as np
import numpy.linalg as npla
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/clmcplot/'))
import utilities as py_util
import clmcplot_utils as clmcplot_util
import rl_tactile_fb_utils as rl_util

catkin_ws_path = py_util.getCatkinWSPath()
sl_data_path = catkin_ws_path + "/install/bin/arm"

is_deleting_all_dfiles_at_init = False#True
N_total_sense_dimensionality = 45
N_primitive = 3

if (is_deleting_all_dfiles_at_init):
    # initialization by removing all SL data files inside sl_data_path
    initial_dfilepaths = py_util.getAllCLMCDataFilePathsInDirectory(sl_data_path)
    for initial_dfilepath in initial_dfilepaths:
        os.remove(initial_dfilepath)

# compute initial reward
all_trial_prim_Rewards_list = list()
init_new_env_dfilepaths = py_util.getAllCLMCDataFilePathsInDirectory(sl_data_path)
for init_new_env_dfilepath in init_new_env_dfilepaths:
    print("Computing rewards from datafile %s..." % init_new_env_dfilepath)
    trial_prim_Rewards = rl_util.computeRewardFromCLMCDataFile(init_new_env_dfilepath, 
                                                               N_Reward_components=N_total_sense_dimensionality)
    assert (len(trial_prim_Rewards) == N_primitive)
    all_trial_prim_Rewards_list.append(trial_prim_Rewards)
all_trial_prim_Rewards = np.vstack(all_trial_prim_Rewards_list)
mean_prim_Rewards = np.mean(all_trial_prim_Rewards, axis=0)

count_pmnn_param_reuse = 0
