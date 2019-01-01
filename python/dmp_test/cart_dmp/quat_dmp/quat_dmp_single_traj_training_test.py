#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 21:00:00 2018

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

def quat_dmp_single_traj_training_test(amd_clmc_dmp_home_dir_path="../../../../", 
                                       canonical_order=2, time_reproduce_max=0.0, 
                                       tau_reproduce=0.0, unroll_ctraj_save_dir_path="", 
                                       unroll_ctraj_save_filename=""):
    task_servo_rate = 300.0
    dt = 1.0/task_servo_rate
    model_size = 50
    tau = MIN_TAU
    
    assert ((canonical_order >= 1) and (canonical_order <= 2))
    assert (time_reproduce_max >= 0.0)
    assert (tau_reproduce >= 0.0)
    
    tau_sys = TauSystem(dt, MIN_TAU)
    canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
    quat_dmp = QuaternionDMP(model_size, canonical_sys_discr)
    [critical_states_learn, 
     W, mean_A_learn, mean_tau, 
     Ft, Fp, QgT, cX, cV, 
     PSI] = quat_dmp.learnFromPath(amd_clmc_dmp_home_dir_path + "/data/dmp_coupling/learn_tactile_feedback/scraping_w_tool/human_baseline/prim03/07.txt", 
                                   task_servo_rate, 
                                   start_column_idx=10, time_column_idx=0)
    
    ## Reproduce
    if (time_reproduce_max <= 0.0):
        time_reproduce_max = mean_tau
    if (tau_reproduce <= 0.0):
        tau_reproduce = mean_tau
    tau = tau_reproduce
    quatdmp_unroll_init_parameters = QuaternionDMPUnrollInitParams(critical_states_learn, tau)
    
    quat_dmp.startWithUnrollParams(quatdmp_unroll_init_parameters)
    
    unroll_ctraj = np.zeros((int(np.round(time_reproduce_max*task_servo_rate) + 1), 5))
    for i in range(int(np.round(time_reproduce_max*task_servo_rate) + 1)):
        time = 1.0 * (i*dt)
        
        [current_state, 
         transform_sys_forcing_term, 
         transform_sys_coupling_term_acc, 
         transform_sys_coupling_term_vel, 
         func_approx_basis_function_vector] = quat_dmp.getNextState(dt, True)
        
        current_Q = current_state.getQ()
        unroll_ctraj[i,0] = time
        unroll_ctraj[i,1] = current_Q[0,0]
        unroll_ctraj[i,2] = current_Q[1,0]
        unroll_ctraj[i,3] = current_Q[2,0]
        unroll_ctraj[i,4] = current_Q[3,0]
    
    if (os.path.isdir(unroll_ctraj_save_dir_path)):
        np.savetxt(unroll_ctraj_save_dir_path + "/" + unroll_ctraj_save_filename, unroll_ctraj, fmt='%.5f')
    
    return unroll_ctraj, W, mean_A_learn, mean_tau, Ft, Fp, QgT, cX, cV, PSI

if __name__ == "__main__":
#    unroll_ctraj, W, mean_A_learn, mean_tau, Ft, Fp, QgT, cX, cV, PSI = quat_dmp_single_traj_training_test()
    unroll_ctraj, W, mean_A_learn, mean_tau, Ft, Fp, QgT, cX, cV, PSI = quat_dmp_single_traj_training_test(canonical_order=1, 
                                                                                                           unroll_ctraj_save_dir_path='/home/amdgsutanto/Desktop/', 
                                                                                                           unroll_ctraj_save_filename='test_python_quat_dmp_single_traj_training_test_0_1.txt')
    unroll_ctraj, W, mean_A_learn, mean_tau, Ft, Fp, QgT, cX, cV, PSI = quat_dmp_single_traj_training_test(canonical_order=2, 
                                                                                                           unroll_ctraj_save_dir_path='/home/amdgsutanto/Desktop/', 
                                                                                                           unroll_ctraj_save_filename='test_python_quat_dmp_single_traj_training_test_0_2.txt')