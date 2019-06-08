#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 01 16:00:00 2019

@author: gsutanto
"""

import numpy as np
import os
import sys
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../cart_dmp/quat_dmp'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from TauSystem import *
from QuaternionDMPUnrollInitParams import *
from CanonicalSystemDiscrete import *
from QuaternionDMP import *
from utilities import *

def quat_dmp_multi_traj_training_test(amd_clmc_dmp_home_dir_path="../../../../", 
                                      canonical_order=2, time_reproduce_max=2.0, 
                                      tau_reproduce=2.0, unroll_qtraj_save_dir_path="", 
                                      unroll_qtraj_save_filename="", 
                                      is_smoothing_training_traj_before_learning=False, 
                                      percentage_padding=None, percentage_smoothing_points=None, 
                                      smoothing_mode=None, dt=None, smoothing_cutoff_frequency=None, 
                                      unroll_learned_params_save_dir_path="", 
                                      unroll_learned_params_save_filename=""):
    task_servo_rate = 300.0
    dt = 1.0/task_servo_rate
    model_size = 25
    tau = MIN_TAU
    
    assert ((canonical_order >= 1) and (canonical_order <= 2))
    assert (time_reproduce_max >= 0.0)
    assert (tau_reproduce >= 0.0)
    
    tau_sys = TauSystem(dt, tau)
    canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
    quat_dmp = QuaternionDMP(model_size, canonical_sys_discr)
    tau = tau_reproduce
    
    unroll_qtraj_length = int(np.round(time_reproduce_max*task_servo_rate) + 1)
    unroll_qtraj_list = list()
    unroll_learned_params_list = list()
    
    for k in range(2):
        unroll_qtraj = np.zeros((unroll_qtraj_length, 5))
        if (k == 0):
            sub_quat_dmp_training_path = "/data/dmp_coupling/learn_tactile_feedback/scraping_w_tool/human_baseline/prim03/"
        elif (k == 1):
            sub_quat_dmp_training_path = "/data/dmp_coupling/learn_tactile_feedback/scraping_wo_tool/human_baseline/prim02/"
        
        [critical_states_learn, 
         W, mean_A_learn, mean_tau, 
         Ft, Fp, QgT, cX, cV, 
         PSI] = quat_dmp.learnFromPath(amd_clmc_dmp_home_dir_path + sub_quat_dmp_training_path, 
                                       task_servo_rate, 
                                       start_column_idx=10, time_column_idx=0, 
                                       is_smoothing_training_traj_before_learning=is_smoothing_training_traj_before_learning, 
                                       percentage_padding=percentage_padding, 
                                       percentage_smoothing_points=percentage_smoothing_points, 
                                       smoothing_mode=smoothing_mode, 
                                       dt=dt, 
                                       smoothing_cutoff_frequency=smoothing_cutoff_frequency)
        
        ## Reproduce
        quatdmp_unroll_init_params = QuaternionDMPUnrollInitParams(critical_states_learn, tau)
        
        quat_dmp.startWithUnrollParams(quatdmp_unroll_init_params)
        
        for i in range(int(np.round(time_reproduce_max*task_servo_rate) + 1)):
            time = 1.0 * (i*dt)
            
            [current_state, 
             transform_sys_forcing_term, 
             transform_sys_coupling_term_acc, 
             transform_sys_coupling_term_vel, 
             func_approx_basis_function_vector] = quat_dmp.getNextState(dt, True)
            
            current_Q = current_state.getQ()
            unroll_qtraj[i,0] = time
            unroll_qtraj[i,1] = current_Q[0,0]
            unroll_qtraj[i,2] = current_Q[1,0]
            unroll_qtraj[i,3] = current_Q[2,0]
            unroll_qtraj[i,4] = current_Q[3,0]
        unroll_qtraj_list.append(copy.deepcopy(unroll_qtraj))
        
        unroll_learned_params = np.zeros((model_size+1+2,5))
        unroll_learned_params[0:model_size,0:3] = W.T
        unroll_learned_params[model_size:model_size+1,0:3] = mean_A_learn.T
        unroll_learned_params[model_size,3] = mean_tau
        unroll_learned_params[model_size+1:model_size+2,0:4] = quat_dmp.mean_start_position.T
        unroll_learned_params[model_size+2:model_size+3,0:4] = quat_dmp.mean_goal_position.T
        unroll_learned_params_list.append(copy.deepcopy(unroll_learned_params))
    
    unroll_qtraj_concatenated = np.vstack(unroll_qtraj_list)
    unroll_learned_params_concatenated = np.vstack(unroll_learned_params_list)
    
    if (os.path.isdir(unroll_qtraj_save_dir_path)):
        np.savetxt(unroll_qtraj_save_dir_path + "/" + unroll_qtraj_save_filename, unroll_qtraj_concatenated)
    
    if (os.path.isdir(unroll_learned_params_save_dir_path)):
        np.savetxt(unroll_learned_params_save_dir_path + "/" + unroll_learned_params_save_filename, unroll_learned_params_concatenated)
    
    return unroll_qtraj_concatenated, W, mean_A_learn, mean_tau, Ft, Fp, QgT, cX, cV, PSI

if __name__ == "__main__":
    unroll_qtraj_concatenated, W, mean_A_learn, mean_tau, Ft, Fp, QgT, cX, cV, PSI = quat_dmp_multi_traj_training_test()