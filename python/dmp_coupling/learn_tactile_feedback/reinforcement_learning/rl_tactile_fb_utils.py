#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:00:00 2019

@author: gsutanto
"""

import os
import sys
import copy
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../cart_dmp/cart_coord_dmp'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../cart_dmp/quat_dmp'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/clmcplot/'))
from TauSystem import *
from QuaternionDMPUnrollInitParams import *
from CanonicalSystemDiscrete import *
from CartesianCoordDMP import *
from QuaternionDMP import *
import DMPTrajectory as dmp_traj
import QuaternionDMPTrajectory as qdmp_traj
import utilities as py_util
import pyplot_util as pypl_util
import clmcplot_utils as clmcplot_util

dim_cart = 3
dim_Q = 4
dim_omega = 3

task_servo_rate = 300.0
dt = 1.0/task_servo_rate
tau = MIN_TAU
canonical_order = 2
model_size = 25

tau_sys = TauSystem(dt, tau)
canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
ccdmp = CartesianCoordDMP(model_size, canonical_sys_discr, SCHAAL_LOCAL_COORD_FRAME)
qdmp = QuaternionDMP(model_size, canonical_sys_discr)

MAX_NUM_CONSECUTIVE_INVALID_VICON_DATA = 6

IS_USING_X_VECTOR_SQUARED_NORM_AS_COST = 0
IS_USING_ROT_DIFF_ERR_SQUARED_NORM_AS_COST = 1

cost_mode = IS_USING_X_VECTOR_SQUARED_NORM_AS_COST
#cost_mode = IS_USING_ROT_DIFF_ERR_SQUARED_NORM_AS_COST

def computeNPrimitives(prim_id_list):
    prim_ids = list(set(prim_id_list))
    prim_ids.sort()
    valid_prim_ids = [int(prim_id) for prim_id in prim_ids if (prim_id >= 0)]
    N_prim = len(valid_prim_ids)
    assert(max(valid_prim_ids) == N_prim - 1)
    return N_prim

def computeValidPrimitiveIDAddresses(prim_id, timeT, N_primitives):
    valid_prim_id_addresses = list()
    for ip in range(N_primitives):
        id_candidate0 = np.where(prim_id == ip)[0]
        id_candidate1 = np.union1d(np.where(timeT[:,0] > 0)[0], np.array([0]))
        id_candidate = np.sort(np.intersect1d(id_candidate0, id_candidate1))
        assert ((id_candidate[1:] - id_candidate[:-1]) == 1).all(), "id_candidate should all be consecutive and has NO jumps!!!"
        valid_prim_id_addresses.append(id_candidate)
        if (ip > 0):
            assert (np.intersect1d(valid_prim_id_addresses[ip], valid_prim_id_addresses[ip-1]).size == 0), "Two Different Primitive ID Address Groups cannot intersect!!!"
    return valid_prim_id_addresses

def computeNumConsecutiveInvalidsAndLastValidAddress(is_vicon_det_valid):
    prev_count_consecutive_invalids = 0
    last_valid_address = -1
    num_consecutive_invalids_array = np.zeros(is_vicon_det_valid.shape[0])
    last_valid_address_array = np.zeros(is_vicon_det_valid.shape[0])
    for aidx in range(is_vicon_det_valid.shape[0]):
        if (is_vicon_det_valid[aidx] == 0):
            prev_count_consecutive_invalids += 1
        elif (is_vicon_det_valid[aidx] == 1):
            prev_count_consecutive_invalids = 0
            last_valid_address = aidx
        num_consecutive_invalids_array[aidx] = prev_count_consecutive_invalids
        last_valid_address_array[aidx] = last_valid_address
    return num_consecutive_invalids_array, last_valid_address_array

def checkUnrollResultCLMCDataFileValidity(dfilepath):
    clmcfile = clmcplot_util.ClmcFile(dfilepath)
    prim_id = clmcfile.get_variables(["ul_curr_prim_no"])[0].T
    N_primitives = computeNPrimitives(prim_id)
    
    # time:
    timeT = np.vstack([clmcfile.get_variables(['time'])]).T
    assert(timeT.shape[0] == prim_id.shape[0])
    assert(timeT.shape[1] == 1)
    
    valid_prim_id_addresses = computeValidPrimitiveIDAddresses(prim_id, timeT, N_primitives)
    
    for ip in range(N_primitives):
        prim_indices = copy.deepcopy(valid_prim_id_addresses[ip])
        prim_timeT_unsubtracted = timeT[prim_indices,:]
        prim_timeT = prim_timeT_unsubtracted - prim_timeT_unsubtracted[0,0]
        if (np.amin(prim_timeT) < 0.0):
            [row_min_prim_timeT, col_min_prim_timeT] = np.unravel_index(prim_timeT.argmin(), prim_timeT.shape)
            if ((row_min_prim_timeT > 0) and (row_min_prim_timeT < prim_timeT.shape[0]-1)):
                for row_idx_offset_plus_1 in range(3):
                    print ("prim # %d/%d: prim_timeT_unsubtracted[%d,%d] = %f" % (ip+1, N_primitives, row_min_prim_timeT+row_idx_offset_plus_1-1, col_min_prim_timeT, prim_timeT_unsubtracted[row_min_prim_timeT+row_idx_offset_plus_1-1, col_min_prim_timeT]))
            print ("Datafile %s is invalid because of condition (np.amin(prim_timeT) < 0.0) at primitive %d!" % (dfilepath, ip+1))
            return False # invalid!
    
    if (cost_mode == IS_USING_ROT_DIFF_ERR_SQUARED_NORM_AS_COST):
        # (binary) trajectory of is_vicon_detection_valid:
        is_vicon_det_valid = clmcfile.get_variables(["is_vicon_det_valid"])[0].T
        assert (is_vicon_det_valid.shape[0] == prim_id.shape[0])
        assert ((is_vicon_det_valid >= 0) and (is_vicon_det_valid <= 1)).all()
        
        all_valid_addresses = np.concatenate(valid_prim_id_addresses)
        assert ((all_valid_addresses[1:] - all_valid_addresses[:-1]) == 1).all(), "all_valid_addresses should all be consecutive and has NO jumps!!!"
        
        [num_consecutive_invalids_array, last_valid_address_array
         ] = computeNumConsecutiveInvalidsAndLastValidAddress(is_vicon_det_valid)
        
        if (num_consecutive_invalids_array[all_valid_addresses] > MAX_NUM_CONSECUTIVE_INVALID_VICON_DATA).any():
            actual_max_num_consecutive_invalids = np.amax(num_consecutive_invalids_array[all_valid_addresses])
            arg_actual_max_num_consecutive_invalids = np.argmax(num_consecutive_invalids_array[all_valid_addresses])
            print ("Datafile %s is invalid because of condition (num_consecutive_invalids_array[%d] = %d > %d = MAX_NUM_CONSECUTIVE_INVALID_VICON_DATA)!" % (dfilepath, arg_actual_max_num_consecutive_invalids, actual_max_num_consecutive_invalids, MAX_NUM_CONSECUTIVE_INVALID_VICON_DATA))
            return False # invalid!
        
        if (last_valid_address_array[all_valid_addresses] < 0).any():
            invalid_last_valid_address_address = np.where(last_valid_address_array[all_valid_addresses] < 0)[0][0]
            print ("Datafile %s is invalid because of condition (last_valid_address_array[%d] = %d < 0)!" % (dfilepath, all_valid_addresses[invalid_last_valid_address_address], last_valid_address_array[all_valid_addresses[invalid_last_valid_address_address]]))
            return False # invalid!
    
    return True # valid!

def extractUnrollResultFromCLMCDataFile(dfilepath, N_cost_components, N_supposed_primitives=None):
    clmcfile = clmcplot_util.ClmcFile(dfilepath)
    prim_id = clmcfile.get_variables(["ul_curr_prim_no"])[0].T
    N_primitives = computeNPrimitives(prim_id)
    if (N_supposed_primitives is not None):
        assert (N_primitives == N_supposed_primitives), "N_primitives = %d ; N_supposed_primitives = %d" % (N_primitives, N_supposed_primitives)
    
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
    DeltaST = np.vstack([clmcfile.get_variables(["X_vector_%02d" % i for i in range(N_cost_components)])]).T
    assert(DeltaST.shape[0] == prim_id.shape[0])
    assert(DeltaST.shape[1] == N_cost_components)
    
    # (binary) trajectory of is_vicon_detection_valid:
    is_vicon_det_valid = clmcfile.get_variables(["is_vicon_det_valid"])[0].T
    assert(is_vicon_det_valid.shape[0] == prim_id.shape[0])
    
    # trajectory of Rotation Difference Error:
    RotDiffErrT = np.vstack([clmcfile.get_variables(['rot_diff_err_a','rot_diff_err_b','rot_diff_err_g'])]).T
    assert(RotDiffErrT.shape[0] == prim_id.shape[0])
    assert(RotDiffErrT.shape[1] == dim_cart)
    
    valid_prim_id_addresses = computeValidPrimitiveIDAddresses(prim_id, timeT, N_primitives)
    
    if (cost_mode == IS_USING_ROT_DIFF_ERR_SQUARED_NORM_AS_COST):
        all_valid_addresses = np.concatenate(valid_prim_id_addresses)
        assert ((all_valid_addresses[1:] - all_valid_addresses[:-1]) == 1).all(), "all_valid_addresses should all be consecutive and has NO jumps!!!"
        
        [_, last_valid_address_array
         ] = computeNumConsecutiveInvalidsAndLastValidAddress(is_vicon_det_valid)
        assert(last_valid_address_array.shape[0] == prim_id.shape[0])
        
        copyRotDiffErrT = copy.deepcopy(RotDiffErrT)
        RotDiffErrT[all_valid_addresses,:] = copyRotDiffErrT[last_valid_address_array[all_valid_addresses],:]
    
    unroll_trajectory = {}
    unroll_trajectory["filepath"] = dfilepath
    unroll_trajectory["id"] = [None] * N_primitives
    unroll_trajectory["timeT"] = [None] * N_primitives
    unroll_trajectory["XT"] = [None] * N_primitives
    unroll_trajectory["XdT"] = [None] * N_primitives
    unroll_trajectory["XddT"] = [None] * N_primitives
    unroll_trajectory["QT"] = [None] * N_primitives
    unroll_trajectory["omegaT"] = [None] * N_primitives
    unroll_trajectory["omegadT"] = [None] * N_primitives
    unroll_trajectory["DeltaST"] = [None] * N_primitives
    unroll_trajectory["RotDiffErrT"] = [None] * N_primitives
    unroll_trajectory["cost_per_timestep"] = [None] * N_primitives
    unroll_cost = [None] * N_primitives
    for ip in range(N_primitives):
        unroll_trajectory["id"][ip] = copy.deepcopy(valid_prim_id_addresses[ip])
        unroll_trajectory["timeT"][ip] = timeT[unroll_trajectory["id"][ip],:]
        unroll_trajectory["XT"][ip] = XT[unroll_trajectory["id"][ip],:]
        unroll_trajectory["XdT"][ip] = XdT[unroll_trajectory["id"][ip],:]
        unroll_trajectory["XddT"][ip] = XddT[unroll_trajectory["id"][ip],:]
        unroll_trajectory["QT"][ip] = QT[unroll_trajectory["id"][ip],:]
        unroll_trajectory["omegaT"][ip] = omegaT[unroll_trajectory["id"][ip],:]
        unroll_trajectory["omegadT"][ip] = omegadT[unroll_trajectory["id"][ip],:]
        unroll_trajectory["DeltaST"][ip] = DeltaST[unroll_trajectory["id"][ip],:]
        unroll_trajectory["RotDiffErrT"][ip] = RotDiffErrT[unroll_trajectory["id"][ip],:]
        if (cost_mode == IS_USING_X_VECTOR_SQUARED_NORM_AS_COST):
            unroll_trajectory["cost_per_timestep"][ip] = py_util.computeSumSquaredL2Norm(unroll_trajectory["DeltaST"][ip],
                                                                                         axis=1).reshape((1,len(unroll_trajectory["id"][ip])))
        elif (cost_mode == IS_USING_ROT_DIFF_ERR_SQUARED_NORM_AS_COST):
            unroll_trajectory["cost_per_timestep"][ip] = py_util.computeSumSquaredL2Norm(unroll_trajectory["RotDiffErrT"][ip],
                                                                                         axis=1).reshape((1,len(unroll_trajectory["id"][ip])))
        unroll_cost[ip] = np.sum(unroll_trajectory["cost_per_timestep"][ip])
    return unroll_trajectory, unroll_cost

def extractUnrollResultsFromCLMCDataFilesInDirectory(directory_path, 
                                                     N_primitives, 
                                                     N_cost_components):
    unroll_results = {}
    unroll_results["trajectory"] = list()
    unroll_results["accum_cost_per_trial"] = list()
    init_new_env_dfilepaths = py_util.getAllCLMCDataFilePathsInDirectory(directory_path)
    for init_new_env_dfilepath in init_new_env_dfilepaths:
        print("Computing costs from datafile %s..." % init_new_env_dfilepath)
        [
         trial_unroll_traj, 
         trial_unroll_cost
        ] = extractUnrollResultFromCLMCDataFile(init_new_env_dfilepath, 
                                                N_cost_components=N_cost_components, 
                                                N_supposed_primitives=N_primitives)
        assert (len(trial_unroll_cost) == N_primitives), "len(trial_unroll_cost) = %d ; N_primitives = %d" % (len(trial_unroll_cost), N_primitives)
        unroll_results["trajectory"].append(trial_unroll_traj)
        unroll_results["accum_cost_per_trial"].append(np.array(trial_unroll_cost).reshape((1,N_primitives)))
    all_trial_prim_costs = np.vstack(unroll_results["accum_cost_per_trial"])
    unroll_results["mean_accum_cost"] = np.mean(all_trial_prim_costs, axis=0)
    assert (len(unroll_results["mean_accum_cost"]) == N_primitives)
    return unroll_results

def extractCartDMPTrajectoriesFromUnrollResults(unroll_results, 
                                                is_time_start_from_zero=True):
    N_trials = len(unroll_results["trajectory"])
    N_primitives = len(unroll_results["trajectory"][0]["id"])
    ccdmp_trajs = [None] * N_primitives
    qdmp_trajs = [None] * N_primitives
    for n_trial in range(N_trials):
        for n_prim in range(N_primitives):
            timeT = unroll_results["trajectory"][n_trial]["timeT"][n_prim].T
            if (is_time_start_from_zero):
                if ((timeT < timeT[0,0]).any()): # debugging for abnormality:
                    [row_min_timeT, col_min_timeT] = np.unravel_index(timeT.argmin(), timeT.shape)
                    if ((col_min_timeT > 0) and (col_min_timeT < timeT.shape[1]-1)):
                        for col_idx_offset_plus_1 in range(3):
                            print ("timeT[%d,%d] = %f" % (row_min_timeT, col_min_timeT+col_idx_offset_plus_1-1, timeT[row_min_timeT, col_min_timeT+col_idx_offset_plus_1-1]))
                timeT = timeT - timeT[0,0]
            assert (np.amin(timeT) >= 0.0), "min(timeT)=" + str(np.amin(timeT)) + " < 0.0 (invalid!), at index " + str(np.unravel_index(timeT.argmin(), timeT.shape)) + ", timeT.shape = " + str(timeT.shape)
            XT = unroll_results["trajectory"][n_trial]["XT"][n_prim].T
            XdT = unroll_results["trajectory"][n_trial]["XdT"][n_prim].T
            XddT = unroll_results["trajectory"][n_trial]["XddT"][n_prim].T
            QT = unroll_results["trajectory"][n_trial]["QT"][n_prim].T
            omegaT = unroll_results["trajectory"][n_trial]["omegaT"][n_prim].T
            omegadT = unroll_results["trajectory"][n_trial]["omegadT"][n_prim].T
            if (n_trial == 0):
                ccdmp_trajs[n_prim] = [None] * N_trials
                qdmp_trajs[n_prim] = [None] * N_trials
            ccdmp_trajs[n_prim][n_trial] = dmp_traj.DMPTrajectory(XT, 
                                                                  XdT, 
                                                                  XddT, 
                                                                  timeT)
            qdmp_trajs[n_prim][n_trial] = qdmp_traj.QuaternionDMPTrajectory(Q_init=QT, 
                                                                            omega_init=omegaT, 
                                                                            omegad_init=omegadT, 
                                                                            time_init=timeT)
            qdmp_trajs[n_prim][n_trial].computeQdAndQdd()
    cdmp_trajs = {}
    cdmp_trajs["CartCoord"] = ccdmp_trajs
    cdmp_trajs["Quaternion"] = qdmp_trajs
    return cdmp_trajs

def learnCartDMPUnrollParams(cdmp_trajs, prim_to_be_learned="All", 
                             is_smoothing_training_traj_before_learning=True, 
                             is_plotting=False, 
                             threshold_var_ground_truth_Q= 5.0e-4):
    N_primitives = len(cdmp_trajs["Quaternion"])
    if (prim_to_be_learned is "All"):
        prim_to_be_learned = range(N_primitives)
    
    if (is_smoothing_training_traj_before_learning):
        percentage_padding = 1.5
        percentage_smoothing_points = 3.0
        smoothing_cutoff_frequency = 5.0
    else:
        percentage_padding = None
        percentage_smoothing_points = None
        smoothing_cutoff_frequency = None
    
    cdmp_params = {}
    cdmp_params["CartCoord"] = [None] * N_primitives
    cdmp_params["Quaternion"] = [None] * N_primitives
    
    cdmp_unroll = {}
    cdmp_unroll["CartCoord"] = [None] * N_primitives
    cdmp_unroll["Quaternion"] = [None] * N_primitives
    
    cdmp_smoothened_trajs = {}
    cdmp_smoothened_trajs["CartCoord"] = [None] * N_primitives
    cdmp_smoothened_trajs["Quaternion"] = [None] * N_primitives
    for n_prim in prim_to_be_learned:
        print("Learning (Modified) Open-Loop Primitive #%d" % (n_prim+1))
        if (is_smoothing_training_traj_before_learning):
            if (np == 0):
                smoothing_mode = 1 # smooth start only
            elif (np == N_primitives - 1):
                smoothing_mode = 2 # smooth end only
            else:
                smoothing_mode = 0 # do not smooth
        else:
            smoothing_mode = None
        
        [
         [ccdmp_critical_states_learn, 
          _, _, _, 
          _, _, _, _, _, 
          _], 
         cdmp_smoothened_trajs["CartCoord"][n_prim]
         ] = ccdmp.learnFromSetTrajectories(cdmp_trajs["CartCoord"][n_prim], task_servo_rate, 
                                            is_smoothing_training_traj_before_learning=is_smoothing_training_traj_before_learning, 
                                            percentage_padding=percentage_padding, 
                                            percentage_smoothing_points=percentage_smoothing_points, 
                                            smoothing_mode=smoothing_mode, 
                                            smoothing_cutoff_frequency=smoothing_cutoff_frequency, 
                                            is_returning_smoothened_training_traj=True
                                            )
        cdmp_params["CartCoord"][n_prim] = ccdmp.getParamsAsDict()
        cdmp_params["CartCoord"][n_prim]["critical_states_learn"] = ccdmp_critical_states_learn
        
        [
         [qdmp_critical_states_learn, 
          _, _, _, 
          _, _, _, _, _, 
          _], 
         cdmp_smoothened_trajs["Quaternion"][n_prim]
         ] = qdmp.learnFromSetTrajectories(cdmp_trajs["Quaternion"][n_prim], task_servo_rate, 
                                           is_smoothing_training_traj_before_learning=is_smoothing_training_traj_before_learning, 
                                           percentage_padding=percentage_padding, 
                                           percentage_smoothing_points=percentage_smoothing_points, 
                                           smoothing_mode=smoothing_mode, 
                                           smoothing_cutoff_frequency=smoothing_cutoff_frequency, 
                                           is_returning_smoothened_training_traj=True
                                           )
        cdmp_params["Quaternion"][n_prim] = qdmp.getParamsAsDict()
        cdmp_params["Quaternion"][n_prim]["critical_states_learn"] = qdmp_critical_states_learn
        
        cdmp_unroll["CartCoord"][n_prim] = ccdmp.unroll(cdmp_params["CartCoord"][n_prim]["critical_states_learn"], 
                                                        cdmp_params["CartCoord"][n_prim]["mean_tau"], 
                                                        cdmp_params["CartCoord"][n_prim]["mean_tau"], 
                                                        dt)
        cdmp_unroll["Quaternion"][n_prim] = qdmp.unroll(cdmp_params["Quaternion"][n_prim]["critical_states_learn"], 
                                                        cdmp_params["Quaternion"][n_prim]["mean_tau"], 
                                                        cdmp_params["Quaternion"][n_prim]["mean_tau"], 
                                                        dt)
        py_util.computeAndDisplayTrajectoryMSEVarGTNMSE(cdmp_trajs["CartCoord"][n_prim], cdmp_unroll["CartCoord"][n_prim], 
                                                        print_prefix="CartCoord  Prim. #%d w.r.t. Original   Demo Trajs " % (n_prim+1), is_orientation_trajectory=False)
        py_util.computeAndDisplayTrajectoryMSEVarGTNMSE(cdmp_trajs["Quaternion"][n_prim], cdmp_unroll["Quaternion"][n_prim], 
                                                        print_prefix="Quaternion Prim. #%d w.r.t. Original   Demo Trajs " % (n_prim+1), is_orientation_trajectory=True)
        print("")
        [mse_smoothened_X, _, _, vargt_smoothened_X, _, _, nmse_smoothened_X, _, _
         ]= py_util.computeAndDisplayTrajectoryMSEVarGTNMSE(cdmp_smoothened_trajs["CartCoord"][n_prim], cdmp_unroll["CartCoord"][n_prim], 
                                                            print_prefix="CartCoord  Prim. #%d w.r.t. Smoothened Demo Trajs " % (n_prim+1), is_orientation_trajectory=False)
        [mse_smoothened_Q, _, _, vargt_smoothened_Q, _, _, nmse_smoothened_Q, _, _
         ]= py_util.computeAndDisplayTrajectoryMSEVarGTNMSE(cdmp_smoothened_trajs["Quaternion"][n_prim], cdmp_unroll["Quaternion"][n_prim], 
                                                            print_prefix="Quaternion Prim. #%d w.r.t. Smoothened Demo Trajs " % (n_prim+1), is_orientation_trajectory=True)
        print("")
        print("")
        
        if (is_plotting or ((n_prim != 1) and ((nmse_smoothened_X >= 1.0).any())) or ((nmse_smoothened_Q >= 1.0).any())):
            ccdmp.plotDemosVsUnroll(cdmp_trajs["CartCoord"][n_prim], cdmp_unroll["CartCoord"][n_prim], 
                                    title_suffix=" Prim. #%d" % (n_prim+1), fig_num_offset=6*n_prim)
            qdmp.plotDemosVsUnroll(cdmp_trajs["Quaternion"][n_prim], cdmp_unroll["Quaternion"][n_prim], 
                                   title_suffix=" Prim. #%d" % (n_prim+1), fig_num_offset=(6*n_prim)+3)
        
        if (n_prim != 1): # 2nd primitive's position DMP fitting maybe bad because there's no change in position (no position movement)
            assert ((nmse_smoothened_X < 1.0).all())
        assert (np.bitwise_or(nmse_smoothened_Q < 1.0, vargt_smoothened_Q < threshold_var_ground_truth_Q).all())
        
    return cdmp_params, cdmp_unroll

def unrollPI2ParamsSamples(pi2_params_samples, prim_to_be_improved, cart_types_to_be_improved, pi2_unroll_mean=None, is_plotting=False):
    K_PI2_samples = len(pi2_params_samples.keys())
    pi2_unroll_samples = {}
    for cart_type_tbi in cart_types_to_be_improved:
        print ("Unrolling PI2 Params Samples of Type %s, Primitive # %d" % (cart_type_tbi, prim_to_be_improved+1))
        pi2_unroll_samples[cart_type_tbi] = [None] * K_PI2_samples
        if (cart_type_tbi == "CartCoord"):
            cdmp_instance = ccdmp
            components_to_be_plotted = ["X", "Xd", "Xdd"]
        elif (cart_type_tbi == "Quaternion"):
            cdmp_instance = qdmp
            components_to_be_plotted = ["Q", "omega", "omegad"]
        else:
            assert False, "cart_type_tbi == %s is un-defined!"
        for k in range(K_PI2_samples):
            print ("   Unrolling PI2 sample # %d/%d ..." % (k+1, K_PI2_samples))
            cdmp_instance.setParamsFromDict(pi2_params_samples[k]["ole_cdmp_params_all_dim_learned"][cart_type_tbi][prim_to_be_improved])
            pi2_unroll_samples[cart_type_tbi][k] = cdmp_instance.unroll(pi2_params_samples[k]["ole_cdmp_params_all_dim_learned"][cart_type_tbi][prim_to_be_improved]["critical_states_learn"], 
                                                                        pi2_params_samples[k]["ole_cdmp_params_all_dim_learned"][cart_type_tbi][prim_to_be_improved]["mean_tau"], 
                                                                        pi2_params_samples[k]["ole_cdmp_params_all_dim_learned"][cart_type_tbi][prim_to_be_improved]["mean_tau"], 
                                                                        dt)
        
        if (is_plotting):
            assert (pi2_unroll_mean is not None)
            plt.close('all')
            py_util.plotManyTrajsVsOneTraj(set_many_trajs=pi2_unroll_samples[cart_type_tbi], 
                                           one_traj=pi2_unroll_mean[cart_type_tbi][prim_to_be_improved], 
                                           title_suffix=" Prim. #%d" % (prim_to_be_improved+1), 
                                           fig_num_offset=0, 
                                           components_to_be_plotted=components_to_be_plotted, 
                                           many_traj_label="pi2_samples", one_traj_label="pi2_mean")
    return pi2_unroll_samples

def plotUnrollPI2ParamSampleVsParamMean(k, prim_to_be_improved, cart_types_to_be_improved, pi2_unroll_samples, pi2_unroll_mean):
    for cart_type_tbi in cart_types_to_be_improved:
        if (cart_type_tbi == "CartCoord"):
            components_to_be_plotted = ["X", "Xd", "Xdd"]
        elif (cart_type_tbi == "Quaternion"):
            components_to_be_plotted = ["Q", "omega", "omegad"]
        else:
            assert False, "cart_type_tbi == %s is un-defined!"
        plt.close('all')
        py_util.plotManyTrajsVsOneTraj(set_many_trajs=[pi2_unroll_samples[cart_type_tbi][k]], 
                                       one_traj=pi2_unroll_mean[cart_type_tbi][prim_to_be_improved], 
                                       title_suffix=" Prim. #%d" % (prim_to_be_improved+1), 
                                       fig_num_offset=0, 
                                       components_to_be_plotted=components_to_be_plotted, 
                                       many_traj_label="pi2_sample # %d/%d" % (k+1, len(pi2_unroll_samples[cart_type_tbi])), one_traj_label="pi2_mean")
    return None

def checkUnrollPI2ParamSampleSupervisionRequirement(k, cart_types_to_be_improved, cart_dim_tbi_supervision_threshold_dict, pi2_unroll_samples):
    for cart_type_tbi in cart_types_to_be_improved:
        for comp in cart_dim_tbi_supervision_threshold_dict[cart_type_tbi].keys():
            if (comp == "X"):
                if ((np.fabs(pi2_unroll_samples[cart_type_tbi][k].X) > cart_dim_tbi_supervision_threshold_dict[cart_type_tbi][comp]).any()):
                    return True
            elif (comp == "Xd"):
                if ((np.fabs(pi2_unroll_samples[cart_type_tbi][k].Xd) > cart_dim_tbi_supervision_threshold_dict[cart_type_tbi][comp]).any()):
                    return True
            elif (comp == "Xdd"):
                if ((np.fabs(pi2_unroll_samples[cart_type_tbi][k].Xdd) > cart_dim_tbi_supervision_threshold_dict[cart_type_tbi][comp]).any()):
                    return True
            elif (comp == "Q"):
                if ((np.fabs(pi2_unroll_samples[cart_type_tbi][k].X) > cart_dim_tbi_supervision_threshold_dict[cart_type_tbi][comp]).any()):
                    return True
            elif (comp == "omega"):
                if ((np.fabs(pi2_unroll_samples[cart_type_tbi][k].omega) > cart_dim_tbi_supervision_threshold_dict[cart_type_tbi][comp]).any()):
                    return True
            elif (comp == "omegad"):
                if ((np.fabs(pi2_unroll_samples[cart_type_tbi][k].omegad) > cart_dim_tbi_supervision_threshold_dict[cart_type_tbi][comp]).any()):
                    return True
            else:
                assert False, "Trajectory component named %s is un-defined!"%comp
    return False

def plotLearningCurve(rl_data, prim_to_be_improved, end_plot_iter, save_filepath=None):
    it = 0
    J_list = list()
    J_prime_list = list()
    J_prime_new_list = list()
    it_list = list()
    while ((it in rl_data[prim_to_be_improved].keys()) and (it <= end_plot_iter)):
        J_list.append(rl_data[prim_to_be_improved][it]["unroll_results"]["mean_accum_cost"][prim_to_be_improved])
        J_prime_list.append(rl_data[prim_to_be_improved][it]["ole_cdmp_evals_all_dim_learned"]["mean_accum_cost"][prim_to_be_improved])
        J_prime_new_list.append(rl_data[prim_to_be_improved][it]["ole_cdmp_new_evals"]["mean_accum_cost"][prim_to_be_improved])
        it_list.append(it)
        it += 1
    Y_list = list()
    Y_list.append(np.array(J_list))
    Y_list.append(np.array(J_prime_list))
    Y_list.append(np.array(J_prime_new_list))
    X_list = [np.array(it_list).astype(int)] * len(Y_list)
    plt.close('all')
    pypl_util.plot_2D(X_list=X_list, 
                      Y_list=Y_list, 
                      title='Total Cost per Iteration', 
                      X_label='Iteration', 
                      Y_label='Total Cost', 
                      fig_num=0, 
                      label_list=['J',"J_prime", "J_prime_new"], 
                      color_style_list=[['r','-'],['g','-.'],['b',':']], 
                      save_filepath=save_filepath)
    plt.close('all')
    pypl_util.plot_2D(X_list=X_list, 
                      Y_list=Y_list, 
                      title='Total Cost per Iteration', 
                      X_label='Iteration', 
                      Y_label='Total Cost', 
                      fig_num=0, 
                      label_list=['J',"J_prime", "J_prime_new"], 
                      color_style_list=[['r','-'],['g','-.'],['b',':']], 
                      save_filepath=None)
    return None

def loadPrimsParamsAsDictFromDirPath(prims_params_dirpath, N_primitives):
    cdmp_params = {}
    cdmp_params["CartCoord"] = [None] * N_primitives
    cdmp_params["Quaternion"] = [None] * N_primitives
    for n_prim in range(N_primitives):
        cdmp_params["CartCoord"][n_prim] = ccdmp.loadParamsAsDict(prims_params_dirpath+"/position/prim%d/"%(n_prim+1), 
                                                                  file_name_weights="w", 
                                                                  file_name_A_learn="A_learn", 
                                                                  file_name_mean_start_position="start_global", 
                                                                  file_name_mean_goal_position="goal_global", 
                                                                  file_name_mean_tau="tau", 
                                                                  file_name_canonical_system_order="canonical_sys_order", 
                                                                  file_name_mean_start_position_global="start_global", 
                                                                  file_name_mean_goal_position_global="goal_global", 
                                                                  file_name_mean_start_position_local="start_local", 
                                                                  file_name_mean_goal_position_local="goal_local", 
                                                                  file_name_ctraj_local_coordinate_frame_selection="ctraj_local_coordinate_frame_selection", 
                                                                  file_name_ctraj_hmg_transform_local_to_global_matrix="T_local_to_global_H", 
                                                                  file_name_ctraj_hmg_transform_global_to_local_matrix="T_global_to_local_H")
        cdmp_params["Quaternion"][n_prim] = qdmp.loadParamsAsDict(prims_params_dirpath+"/orientation/prim%d/"%(n_prim+1), 
                                                                  file_name_weights="w", 
                                                                  file_name_A_learn="A_learn", 
                                                                  file_name_mean_start_position="start", 
                                                                  file_name_mean_goal_position="goal", 
                                                                  file_name_mean_tau="tau", 
                                                                  file_name_canonical_system_order="canonical_sys_order")
    return cdmp_params

def savePrimsParamsFromDictAtDirPath(prims_params_dirpath, cdmp_params):
    N_primitives = len(cdmp_params["CartCoord"])
    py_util.createDirIfNotExist(prims_params_dirpath)
    for n_prim in range(N_primitives):
        ccdmp_prim_param_dirpath = prims_params_dirpath+"/position/prim%d/"%(n_prim+1)
        py_util.recreateDir(ccdmp_prim_param_dirpath)
        ccdmp.saveParamsFromDict(dir_path=ccdmp_prim_param_dirpath, cart_coord_dmp_params=cdmp_params["CartCoord"][n_prim], 
                                 file_name_weights="w", 
                                 file_name_A_learn="A_learn", 
                                 file_name_mean_start_position="start_global", 
                                 file_name_mean_goal_position="goal_global", 
                                 file_name_mean_tau="tau", 
                                 file_name_canonical_system_order="canonical_sys_order", 
                                 file_name_mean_start_position_global="start_global", 
                                 file_name_mean_goal_position_global="goal_global", 
                                 file_name_mean_start_position_local="start_local", 
                                 file_name_mean_goal_position_local="goal_local", 
                                 file_name_ctraj_local_coordinate_frame_selection="ctraj_local_coordinate_frame_selection", 
                                 file_name_ctraj_hmg_transform_local_to_global_matrix="T_local_to_global_H", 
                                 file_name_ctraj_hmg_transform_global_to_local_matrix="T_global_to_local_H")
        qdmp_prim_param_dirpath = prims_params_dirpath+"/orientation/prim%d/"%(n_prim+1)
        py_util.recreateDir(qdmp_prim_param_dirpath)
        qdmp.saveParamsFromDict(dir_path=qdmp_prim_param_dirpath, dmp_params=cdmp_params["Quaternion"][n_prim], 
                                file_name_weights="w", 
                                file_name_A_learn="A_learn", 
                                file_name_mean_start_position="start", 
                                file_name_mean_goal_position="goal", 
                                file_name_mean_tau="tau", 
                                file_name_canonical_system_order="canonical_sys_order")
    return None

def extractParamsToBeImproved(params_dict, type_dim_tbi_dict, types_tbi_list, prim_tbi):
    params_tbi_list = list()
    params_tbi_dims_list = list()
    for type_tbi in types_tbi_list:
        params_tbi_dims = [len(type_dim_tbi_dict[type_tbi]), params_dict[type_tbi][prim_tbi]["W"].shape[1]]
        params_tbi_length = params_tbi_dims[0] * params_tbi_dims[1]
        params_tbi_list.append(params_dict[type_tbi][prim_tbi]["W"][type_dim_tbi_dict[type_tbi],:].reshape(params_tbi_length,1))
        params_tbi_dims_list.append(params_tbi_dims)
    params_tbi_column_vector = np.vstack(params_tbi_list)
    params_tbi = params_tbi_column_vector.reshape((params_tbi_column_vector.shape[0],))
    return params_tbi, params_tbi_dims_list

def updateParamsToBeImproved(params_dict, type_dim_tbi_dict, types_tbi_list, prim_tbi, params_tbi, params_tbi_dims_list):
    # the opposite operation of extractParamsToBeImproved():
    params_tbi_addr_offset = 0
    params_tbi_dims_list_idx = 0
    for type_tbi in types_tbi_list:
        params_tbi_dims = params_tbi_dims_list[params_tbi_dims_list_idx]
        params_tbi_length = params_tbi_dims[0] * params_tbi_dims[1]
        params_dict[type_tbi][prim_tbi]["W"][type_dim_tbi_dict[type_tbi],:] = params_tbi[params_tbi_addr_offset:(params_tbi_addr_offset+params_tbi_length)].reshape(params_tbi_dims)
        params_tbi_addr_offset += params_tbi_length
        params_tbi_dims_list_idx += 1
    return params_dict

def computeParamInitStdHeuristic(param_mean, params_mean_extrema_to_init_std_factor=0.5):
    param_init_std = params_mean_extrema_to_init_std_factor * (0.5 * (np.max(np.fabs(param_mean)) + 
                                                                      np.min(np.fabs(param_mean))))
    return param_init_std