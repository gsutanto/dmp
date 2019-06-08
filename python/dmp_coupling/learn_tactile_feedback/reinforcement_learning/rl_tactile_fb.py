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
import scipy.io as sio
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/clmcplot/'))
import utilities as py_util
import clmcplot_utils as clmcplot_util
import rl_tactile_fb_utils as rl_util

plt.close('all')

catkin_ws_path = py_util.getCatkinWSPath()
sl_data_dirpath = catkin_ws_path + "/install/bin/arm"
outdata_dirpath = './'

is_deleting_dfiles = False#True
is_smoothing_training_traj_before_learning = True
is_plotting = True#False

N_total_sense_dimensionality = 45
N_primitives = 3
prim_to_be_improved = [1,2] # 2nd and 3rd primitives

if (is_deleting_dfiles):
    # initialization by removing all SL data files inside sl_data_dirpath
    py_util.deleteAllCLMCDataFilesInDirectory(sl_data_dirpath)

rl_data = {}

# extract initial unrolling results: trajectories, sensor trace deviations, reward
rl_data[0] = rl_util.extractUnrollResultsFromCLMCDataFilesInDirectory(sl_data_dirpath, 
                                                                      N_primitives=N_primitives, 
                                                                      N_reward_components=N_total_sense_dimensionality)

py_util.saveObj(rl_data, outdata_dirpath+'rl_data.pkl')

count_pmnn_param_reuse = 0
cdmp_trajs = rl_util.extractCartDMPTrajectoriesFromUnrollResults(rl_data[0])
[
 cdmp_params, 
 cdmp_unroll
] = rl_util.learnCartDMPUnrollParams(cdmp_trajs, 
                                     prim_to_be_learned="All", 
                                     is_smoothing_training_traj_before_learning=is_smoothing_training_traj_before_learning, 
                                     is_plotting=is_plotting)

if (is_deleting_dfiles):
    py_util.deleteAllCLMCDataFilesInDirectory(sl_data_dirpath)

