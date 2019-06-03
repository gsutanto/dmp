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

is_deleting_dfiles = False#True
N_total_sense_dimensionality = 45
N_primitive = 3

if (is_deleting_dfiles):
    # initialization by removing all SL data files inside sl_data_path
    py_util.deleteAllCLMCDataFilesInDirectory(sl_data_path)

# extract initial unrolling results: trajectories, sensor trace deviations, reward
[
 prim_unroll_results, 
 mean_prim_Rewards
] = rl_util.extractUnrollResultsFromCLMCDataFilesInDirectory(sl_data_path, 
                                                             N_primitive=N_primitive, 
                                                             N_Reward_components=N_total_sense_dimensionality)

count_pmnn_param_reuse = 0
