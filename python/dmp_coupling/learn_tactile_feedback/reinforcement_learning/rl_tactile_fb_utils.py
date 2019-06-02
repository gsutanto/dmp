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

def computeNPrimitives(prim_id_list):
    prim_ids = list(set(prim_id_list))
    prim_ids.sort()
    valid_prim_ids = [int(prim_id) for prim_id in prim_ids if (prim_id >= 0)]
    N_prim = len(valid_prim_ids)
    assert(max(valid_prim_ids) == N_prim - 1)
    return N_prim

def computePrimRewardsFromCLMCDataFile(dfilepath, N_Reward_components):
    clmcfile = clmcplot_util.ClmcFile(dfilepath)
    trial_prim_id = clmcfile.get_variables(["ul_curr_prim_no"])[0].T
    N_primitive = computeNPrimitives(trial_prim_id)
    
    trial_X_vector = np.vstack([clmcfile.get_variables(["X_vector_%02d" % i for i in range(N_Reward_components)])]).T
    assert(trial_X_vector.shape[0] == trial_prim_id.shape[0])
    assert(trial_X_vector.shape[1] == N_Reward_components)
    
    trial_prim_indices = list()
    trial_prim_X_vectors = list()
    trial_prim_Rewards = list()
    for ip in range(N_primitive):
        trial_prim_indices.append(np.where(trial_prim_id == ip)[0])
        trial_prim_X_vectors.append(trial_X_vector[trial_prim_indices[ip],:])
        trial_prim_Rewards.append(-npla.norm(trial_prim_X_vectors[ip], ord=2))
    return trial_prim_Rewards

def computeMeanPrimRewardsFromCLMCDataFilesInDirectory(directory_path, 
                                                       N_primitive, 
                                                       N_Reward_components):
    all_trial_prim_Rewards_list = list()
    init_new_env_dfilepaths = py_util.getAllCLMCDataFilePathsInDirectory(directory_path)
    for init_new_env_dfilepath in init_new_env_dfilepaths:
        print("Computing rewards from datafile %s..." % init_new_env_dfilepath)
        trial_prim_Rewards = computePrimRewardsFromCLMCDataFile(init_new_env_dfilepath, 
                                                                N_Reward_components=N_Reward_components)
        assert (len(trial_prim_Rewards) == N_primitive)
        all_trial_prim_Rewards_list.append(trial_prim_Rewards)
    all_trial_prim_Rewards = np.vstack(all_trial_prim_Rewards_list)
    mean_prim_Rewards = np.mean(all_trial_prim_Rewards, axis=0)
    return mean_prim_Rewards