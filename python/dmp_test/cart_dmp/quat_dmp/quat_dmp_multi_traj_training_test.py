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
                                      tau_reproduce=2.0, unroll_ctraj_save_dir_path="", 
                                      unroll_ctraj_save_filename="", 
                                      is_smoothing_training_traj_before_learning=False, 
                                      percentage_padding=None, percentage_smoothing_points=None, 
                                      smoothing_mode=None, dt=None, smoothing_cutoff_frequency=None):
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
    
    unroll_ctraj_length = int(np.round(time_reproduce_max*task_servo_rate) + 1)
    unroll_ctraj = np.zeros((2 * unroll_ctraj_length, 5))
    
    for k in range(2):
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
        
        start_logging_idx = (k*unroll_ctraj_length)
        
        for i in range(int(np.round(time_reproduce_max*task_servo_rate) + 1)):
            time = 1.0 * (i*dt)
            
            [current_state, 
             transform_sys_forcing_term, 
             transform_sys_coupling_term_acc, 
             transform_sys_coupling_term_vel, 
             func_approx_basis_function_vector] = quat_dmp.getNextState(dt, True)
            
            current_Q = current_state.getQ()
            unroll_ctraj[start_logging_idx+i,0] = time
            unroll_ctraj[start_logging_idx+i,1] = current_Q[0,0]
            unroll_ctraj[start_logging_idx+i,2] = current_Q[1,0]
            unroll_ctraj[start_logging_idx+i,3] = current_Q[2,0]
            unroll_ctraj[start_logging_idx+i,4] = current_Q[3,0]
    
    if (os.path.isdir(unroll_ctraj_save_dir_path)):
        np.savetxt(unroll_ctraj_save_dir_path + "/" + unroll_ctraj_save_filename, unroll_ctraj
#                   , fmt='%.5f'
                   )
    
    return unroll_ctraj, W, mean_A_learn, mean_tau, Ft, Fp, QgT, cX, cV, PSI

if __name__ == "__main__":
    unroll_ctraj, W, mean_A_learn, mean_tau, Ft, Fp, QgT, cX, cV, PSI = quat_dmp_multi_traj_training_test()