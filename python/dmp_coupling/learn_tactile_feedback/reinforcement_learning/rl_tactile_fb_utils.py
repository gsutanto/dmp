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
import clmcplot_utils as clmcplot_util

def computeNPrimitives(prim_id_list):
    prim_ids = list(set(prim_id_list))
    prim_ids.sort()
    valid_prim_ids = [int(prim_id) for prim_id in prim_ids if (prim_id >= 0)]
    N_prim = len(valid_prim_ids)
    assert(max(valid_prim_ids) == N_prim - 1)
    return N_prim

def computeRewardFromCLMCDataFile(dfilepath, N_Reward_components):
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