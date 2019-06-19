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
orig_prims_params_dirpath = "../../../../data/dmp_coupling/learn_tactile_feedback/scraping/learned_prims_params/"
outdata_dirpath = './'

is_deleting_dfiles = False#True
is_smoothing_training_traj_before_learning = True
is_plotting = True#False

N_total_sense_dimensionality = 45
N_primitives = 3
prim_to_be_improved = [1,2] # 2nd and 3rd primitives

# not sure if the original (nominal) primitives below is needed or not...:
orig_cdmp_params = rl_util.loadPrimsParamsAsDictFromDirPath(orig_prims_params_dirpath, N_primitives)

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
 rl_data[0]["cdmp_params_all_dim_learned"], 
 rl_data[0]["cdmp_unroll_all_dim_learned"]
] = rl_util.learnCartDMPUnrollParams(cdmp_trajs, 
                                     prim_to_be_learned="All", 
                                     is_smoothing_training_traj_before_learning=is_smoothing_training_traj_before_learning, 
                                     is_plotting=is_plotting)

if (is_deleting_dfiles):
    py_util.deleteAllCLMCDataFilesInDirectory(sl_data_dirpath)

# set to-be-perturbed DMP params (2nd dimension of orientation) as mean, and define the initial covariance matrix
# sample K perturbed DMP params from the multivariate normal distribution with mean and cov parameters (and log these K samples of perturbed DMP params)
# For Debugging via visualizations: unroll each of these K perturbed DMP params, log the unrolled trajectories, and visualize them as plots (as needed)
# save these K perturbed DMP params (one at a time) as text files, to be loaded by C++ program and executed by the robot, to evaluate each of their costs.
# summarize these K perturbed DMP params into mean_new and cov_new using PI2 update, based on each of their cost
# save mean_new (which is a DMP params by itself) as text files, to be loaded by C++ program and executed by the robot, to evaluate its cost Nu times, to ensure the average cost is really lower than the original one