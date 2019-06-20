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

is_deleting_dfiles = False#True # TODO (remove this)
is_smoothing_training_traj_before_learning = True
is_plotting = True#False

N_total_sense_dimensionality = 45
N_primitives = 3
#prims_tbi = [1,2] # TODO (un-comment): 2nd and 3rd primitives are to-be-improved (tbi)
prims_tbi = [1] # TODO (comment): for testing purpose we work on 2nd primitive only as the one to-be-improved (tbi)
cost_threshold = [0.0, 18928850.8053, 11066375.797]
cart_dim_tbi = {}
cart_dim_tbi["Quaternion"] = np.array([1]) # to-be-improved (tbi): Quaternion DMP, 2nd dimension

# not sure if the original (nominal) primitives below is needed or not...:
#orig_cdmp_params = rl_util.loadPrimsParamsAsDictFromDirPath(orig_prims_params_dirpath, N_primitives)

if (is_deleting_dfiles): # TODO (remove this)
    # initialization by removing all SL data files inside sl_data_dirpath
    py_util.deleteAllCLMCDataFilesInDirectory(sl_data_dirpath)

rl_data = {}

count_pmnn_param_reuse = 0
for prim_tbi in prims_tbi:
    rl_data[prim_tbi] = {}
    it = 0
    
    if (is_deleting_dfiles): # TODO (remove this)
        py_util.deleteAllCLMCDataFilesInDirectory(sl_data_dirpath)
    
    # TODO: Robot Execution: to evaluate initial cost on current closed-loop behavior 
    #                        (involving feedback model) by unrolling 3 times and averaging the cost
    
    # extract initial unrolling results: trajectories, sensor trace deviations, cost
    rl_data[prim_tbi][it] = rl_util.extractUnrollResultsFromCLMCDataFilesInDirectory(sl_data_dirpath, 
                                                                                     N_primitives=N_primitives, 
                                                                                     N_cost_components=N_total_sense_dimensionality)
    
    py_util.saveObj(rl_data, outdata_dirpath+'rl_data.pkl')
    
    while (rl_data[prim_tbi][it]["mean_accum_cost"][prim_tbi] > cost_threshold[prim_tbi]): # while (J > threshold):
        # convert current closed-loop behavior into an equivalent open-loop behavior on
        # the current (assumed-static) environment setting
        cdmp_trajs = rl_util.extractCartDMPTrajectoriesFromUnrollResults(rl_data[prim_tbi][it])
        [
         rl_data[prim_tbi][it]["cdmp_params_all_dim_learned"], 
         rl_data[prim_tbi][it]["cdmp_unroll_all_dim_learned"]
        ] = rl_util.learnCartDMPUnrollParams(cdmp_trajs, 
                                             prim_to_be_learned="All", 
                                             is_smoothing_training_traj_before_learning=is_smoothing_training_traj_before_learning, 
                                             is_plotting=is_plotting)
        
        py_util.saveObj(rl_data, outdata_dirpath+'rl_data.pkl')
        
        if (is_deleting_dfiles): # TODO (remove this)
            py_util.deleteAllCLMCDataFilesInDirectory(sl_data_dirpath)
        
        # TODO: Robot Execution: unroll DMP params of the (assumed) equivalent open-loop behavior, 
        #                        and measure cost J'
        
        # TODO: check (assert?) if J' is closely similar to J?
        
        for cart_type_tbi in cart_dim_tbi.keys():
            # set to-be-perturbed DMP params (2nd dimension of orientation) as mean, and define the initial covariance matrix
            param_mean = rl_data[prim_tbi][it]["cdmp_params_all_dim_learned"][cart_type_tbi][prim_tbi]["W"][cart_dim_tbi[cart_type_tbi],:]
            # param_cov = 
            
            assert False
            
            # TODO: sample K perturbed DMP params from the multivariate normal distribution with mean and cov parameters (and log these K samples of perturbed DMP params)
            # TODO: For Debugging via visualizations: unroll each of these K perturbed DMP params, log the unrolled trajectories, and visualize them as plots (as needed)
            # TODO: save these K perturbed DMP params (one at a time) as text files, to be loaded by C++ program and executed by the robot, to evaluate each of their costs.
            # TODO: summarize these K perturbed DMP params into mean_new and cov_new using PI2 update, based on each of their cost
            # TODO: save mean_new (which is a DMP params by itself) as text files, to be loaded by C++ program and executed by the robot, to evaluate its cost Nu times, to ensure the average cost is really lower than the original one