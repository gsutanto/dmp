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

def extractUnrollResultFromCLMCDataFile(dfilepath, N_reward_components):
    clmcfile = clmcplot_util.ClmcFile(dfilepath)
    prim_id = clmcfile.get_variables(["ul_curr_prim_no"])[0].T
    N_primitives = computeNPrimitives(prim_id)
    
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
    DeltaST = np.vstack([clmcfile.get_variables(["X_vector_%02d" % i for i in range(N_reward_components)])]).T
    assert(DeltaST.shape[0] == prim_id.shape[0])
    assert(DeltaST.shape[1] == N_reward_components)
    
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
    unroll_reward = [None] * N_primitives
    for ip in range(N_primitives):
        unroll_trajectory["id"][ip] = np.where(prim_id == ip)[0]
        unroll_trajectory["timeT"][ip] = timeT[unroll_trajectory["id"][ip],:]
        unroll_trajectory["XT"][ip] = XT[unroll_trajectory["id"][ip],:]
        unroll_trajectory["XdT"][ip] = XdT[unroll_trajectory["id"][ip],:]
        unroll_trajectory["XddT"][ip] = XddT[unroll_trajectory["id"][ip],:]
        unroll_trajectory["QT"][ip] = QT[unroll_trajectory["id"][ip],:]
        unroll_trajectory["omegaT"][ip] = omegaT[unroll_trajectory["id"][ip],:]
        unroll_trajectory["omegadT"][ip] = omegadT[unroll_trajectory["id"][ip],:]
        unroll_trajectory["DeltaST"][ip] = DeltaST[unroll_trajectory["id"][ip],:]
        unroll_reward[ip] = -npla.norm(unroll_trajectory["DeltaST"][ip], ord=2)
    return unroll_trajectory, unroll_reward

def extractUnrollResultsFromCLMCDataFilesInDirectory(directory_path, 
                                                     N_primitives, 
                                                     N_reward_components):
    unroll_results = {}
    unroll_results["trajectory"] = list()
    unroll_results["reward_per_trial"] = list()
    init_new_env_dfilepaths = py_util.getAllCLMCDataFilePathsInDirectory(directory_path)
    for init_new_env_dfilepath in init_new_env_dfilepaths:
        print("Computing rewards from datafile %s..." % init_new_env_dfilepath)
        [
         trial_unroll_traj, 
         trial_unroll_reward
        ] = extractUnrollResultFromCLMCDataFile(init_new_env_dfilepath, 
                                                N_reward_components=N_reward_components)
        assert (len(trial_unroll_reward) == N_primitives)
        unroll_results["trajectory"].append(trial_unroll_traj)
        unroll_results["reward_per_trial"].append(trial_unroll_reward)
    all_trial_prim_rewards = np.vstack(unroll_results["reward_per_trial"])
    unroll_results["mean_reward"] = np.mean(all_trial_prim_rewards, axis=0)
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
                timeT = timeT - timeT[0,0]
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
                             is_plotting=False):
    N_primitives = len(cdmp_trajs["Quaternion"])
    if (prim_to_be_learned is "All"):
        prim_to_be_learned = range(N_primitives)
    
    task_servo_rate = 300.0
    dt = 1.0/task_servo_rate
    tau = MIN_TAU
    canonical_order = 2
    model_size = 25
    
    if (is_smoothing_training_traj_before_learning):
        percentage_padding = 1.5
        percentage_smoothing_points = 3.0
        smoothing_cutoff_frequency = 5.0
    else:
        percentage_padding = None
        percentage_smoothing_points = None
        smoothing_cutoff_frequency = None
    
    tau_sys = TauSystem(dt, tau)
    canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
    ccdmp = CartesianCoordDMP(model_size, canonical_sys_discr, SCHAAL_LOCAL_COORD_FRAME)
    qdmp = QuaternionDMP(model_size, canonical_sys_discr)
    
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
        
        cdmp_unroll["CartCoord"][n_prim] = ccdmp.unroll(ccdmp_critical_states_learn, 
                                                        cdmp_params["CartCoord"][n_prim]["mean_tau"], 
                                                        cdmp_params["CartCoord"][n_prim]["mean_tau"], 
                                                        dt)
        cdmp_unroll["Quaternion"][n_prim] = qdmp.unroll(qdmp_critical_states_learn, 
                                                        cdmp_params["Quaternion"][n_prim]["mean_tau"], 
                                                        cdmp_params["Quaternion"][n_prim]["mean_tau"], 
                                                        dt)
        py_util.computeAndDisplayTrajectoryNMSE(cdmp_trajs["CartCoord"][n_prim], cdmp_unroll["CartCoord"][n_prim], 
                                                print_prefix="CartCoord  Prim. #%d w.r.t. Original   Demo Trajs " % (n_prim+1), is_orientation_trajectory=False)
        py_util.computeAndDisplayTrajectoryNMSE(cdmp_trajs["Quaternion"][n_prim], cdmp_unroll["Quaternion"][n_prim], 
                                                print_prefix="Quaternion Prim. #%d w.r.t. Original   Demo Trajs " % (n_prim+1), is_orientation_trajectory=True)
        print("")
        [nmse_smoothened_X, _, _
         ]= py_util.computeAndDisplayTrajectoryNMSE(cdmp_smoothened_trajs["CartCoord"][n_prim], cdmp_unroll["CartCoord"][n_prim], 
                                                    print_prefix="CartCoord  Prim. #%d w.r.t. Smoothened Demo Trajs " % (n_prim+1), is_orientation_trajectory=False)
        [nmse_smoothened_Q, _, _
         ]= py_util.computeAndDisplayTrajectoryNMSE(cdmp_smoothened_trajs["Quaternion"][n_prim], cdmp_unroll["Quaternion"][n_prim], 
                                                    print_prefix="Quaternion Prim. #%d w.r.t. Smoothened Demo Trajs " % (n_prim+1), is_orientation_trajectory=True)
        print("")
        print("")
        
        if (is_plotting):
            ccdmp.plotDemosVsUnroll(cdmp_trajs["CartCoord"][n_prim], cdmp_unroll["CartCoord"][n_prim], 
                                    title_suffix=" Prim. #%d" % (n_prim+1), fig_num_offset=6*n_prim)
            qdmp.plotDemosVsUnroll(cdmp_trajs["Quaternion"][n_prim], cdmp_unroll["Quaternion"][n_prim], 
                                   title_suffix=" Prim. #%d" % (n_prim+1), fig_num_offset=(6*n_prim)+3)
        
        if (n_prim != 1): # 2nd primitive's position DMP fitting maybe bad because there's no change in position (no position movement)
            assert ((nmse_smoothened_X < 1.0).all())
        assert ((nmse_smoothened_Q < 1.0).all())
        
    return cdmp_params, cdmp_unroll

def loadPrimsParamsAsDictFromDirPath(prims_params_dirpath, N_primitives):
    task_servo_rate = 300.0
    dt = 1.0/task_servo_rate
    tau = MIN_TAU
    canonical_order = 2
    model_size = 25
    
    tau_sys = TauSystem(dt, tau)
    canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
    ccdmp = CartesianCoordDMP(model_size, canonical_sys_discr, SCHAAL_LOCAL_COORD_FRAME)
    qdmp = QuaternionDMP(model_size, canonical_sys_discr)
    
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