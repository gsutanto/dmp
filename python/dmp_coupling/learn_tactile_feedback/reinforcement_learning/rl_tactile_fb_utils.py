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

dim_cart = 3
dim_Q = 4
dim_omega = 3

def computeNPrimitives(prim_id_list):
    prim_ids = list(set(prim_id_list))
    prim_ids.sort()
    valid_prim_ids = [int(prim_id) for prim_id in prim_ids if (prim_id >= 0)]
    N_prim = len(valid_prim_ids)
    assert(max(valid_prim_ids) == N_prim - 1)
    return N_prim

def extractUnrollResultFromCLMCDataFile(dfilepath, N_Reward_components):
    clmcfile = clmcplot_util.ClmcFile(dfilepath)
    prim_id = clmcfile.get_variables(["ul_curr_prim_no"])[0].T
    N_primitive = computeNPrimitives(prim_id)
    
    # time:
    timeT = np.vstack([clmcfile.get_variables(['time'])]).T
    assert(timeT.shape[0] == prim_id.shape[0])
    assert(timeT.shape[1] == 1)
    
    # trajectory of X, Xd, and Xdd:
    XT = np.vstack([clmcfile.get_variables(['R_HAND_x','R_HAND_y','R_HAND_z'])]).T
    assert(XT.shape[0] == prim_id.shape[0])
    assert(XT.shape[1] == dim_cart)
    XdT = np.vstack([clmcfile.get_variables(['R_HAND_xd','R_HAND_yd','R_HAND_zd'])]).T
    assert(XdT.shape[0] == prim_id.shape[0])
    assert(XdT.shape[1] == dim_cart)
    XddT = np.vstack([clmcfile.get_variables(['R_HAND_xdd','R_HAND_ydd','R_HAND_zdd'])]).T
    assert(XddT.shape[0] == prim_id.shape[0])
    assert(XddT.shape[1] == dim_cart)
    
    # trajectory of Q, omega, and omegad:
    QT = np.vstack([clmcfile.get_variables(['R_HAND_q0','R_HAND_q1','R_HAND_q2','R_HAND_q3'])]).T
    assert(QT.shape[0] == prim_id.shape[0])
    assert(QT.shape[1] == dim_Q)
    omegaT = np.vstack([clmcfile.get_variables(['R_HAND_ad','R_HAND_bd','R_HAND_gd'])]).T
    assert(omegaT.shape[0] == prim_id.shape[0])
    assert(omegaT.shape[1] == dim_omega)
    omegadT = np.vstack([clmcfile.get_variables(['R_HAND_add','R_HAND_bdd','R_HAND_gdd'])]).T
    assert(omegadT.shape[0] == prim_id.shape[0])
    assert(omegadT.shape[1] == dim_omega)
    
    # trajectory of Delta S:
    DeltaST = np.vstack([clmcfile.get_variables(["X_vector_%02d" % i for i in range(N_Reward_components)])]).T
    assert(DeltaST.shape[0] == prim_id.shape[0])
    assert(DeltaST.shape[1] == N_Reward_components)
    
    unroll_result = {}
    unroll_result["filepath"] = dfilepath
    unroll_result["id"] = [None] * N_primitive
    unroll_result["timeT"] = [None] * N_primitive
    unroll_result["XT"] = [None] * N_primitive
    unroll_result["XdT"] = [None] * N_primitive
    unroll_result["XddT"] = [None] * N_primitive
    unroll_result["QT"] = [None] * N_primitive
    unroll_result["omegaT"] = [None] * N_primitive
    unroll_result["omegadT"] = [None] * N_primitive
    unroll_result["DeltaST"] = [None] * N_primitive
    unroll_result["Reward"] = [None] * N_primitive
    for ip in range(N_primitive):
        unroll_result["id"][ip] = np.where(prim_id == ip)[0]
        unroll_result["timeT"][ip] = timeT[unroll_result["id"][ip],:]
        unroll_result["XT"][ip] = XT[unroll_result["id"][ip],:]
        unroll_result["XdT"][ip] = XdT[unroll_result["id"][ip],:]
        unroll_result["XddT"][ip] = XddT[unroll_result["id"][ip],:]
        unroll_result["QT"][ip] = QT[unroll_result["id"][ip],:]
        unroll_result["omegaT"][ip] = omegaT[unroll_result["id"][ip],:]
        unroll_result["omegadT"][ip] = omegadT[unroll_result["id"][ip],:]
        unroll_result["DeltaST"][ip] = DeltaST[unroll_result["id"][ip],:]
        unroll_result["Reward"][ip] = -npla.norm(unroll_result["DeltaST"][ip], ord=2)
    return unroll_result

def extractUnrollResultsFromCLMCDataFilesInDirectory(directory_path, 
                                                     N_primitive, 
                                                     N_Reward_components):
    all_trial_unroll_results_list = list()
    all_trial_prim_Rewards_list = list()
    init_new_env_dfilepaths = py_util.getAllCLMCDataFilePathsInDirectory(directory_path)
    for init_new_env_dfilepath in init_new_env_dfilepaths:
        print("Computing rewards from datafile %s..." % init_new_env_dfilepath)
        trial_unroll_result = extractUnrollResultFromCLMCDataFile(init_new_env_dfilepath, 
                                                                  N_Reward_components=N_Reward_components)
        assert (len(trial_unroll_result["Reward"]) == N_primitive)
        all_trial_unroll_results_list.append(trial_unroll_result)
        all_trial_prim_Rewards_list.append(trial_unroll_result["Reward"])
    all_trial_prim_Rewards = np.vstack(all_trial_prim_Rewards_list)
    mean_prim_Rewards = np.mean(all_trial_prim_Rewards, axis=0)
    return all_trial_unroll_results_list, mean_prim_Rewards