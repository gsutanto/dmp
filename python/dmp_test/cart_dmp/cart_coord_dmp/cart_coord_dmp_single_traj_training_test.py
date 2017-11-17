#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import numpy as np
import os
import sys
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../cart_dmp/cart_coord_dmp'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from TauSystem import *
from DMPUnrollInitParams import *
from CanonicalSystemDiscrete import *
from CartesianCoordTransformer import *
from CartesianCoordDMP import *
from utilities import *

def cart_coord_dmp_single_traj_training_test(amd_clmc_dmp_home_dir_path="../../../../", 
                                             canonical_order=2, time_reproduce_max=0.0, time_goal_change=0.0, 
                                             tau_reproduce=0.0, unroll_ctraj_save_dir_path="", 
                                             unroll_ctraj_save_filename=""):
    task_servo_rate = 420.0
    dt = 1.0/task_servo_rate
    model_size = 50
    tau = MIN_TAU
    
    assert ((canonical_order >= 1) and (canonical_order <= 2))
    assert (time_reproduce_max >= 0.0)
    assert (time_goal_change >= 0.0)
    assert (tau_reproduce >= 0.0)
    
    tau_sys = TauSystem(MIN_TAU)
    canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
    cart_dmp = CartesianCoordDMP(model_size, canonical_sys_discr, GSUTANTO_LOCAL_COORD_FRAME)
    [tau_learn, critical_states_learn, 
     W, mean_A_learn, mean_tau, 
     Ft, Fp, G, cX, cV, PSI] = cart_dmp.learnFromPath(amd_clmc_dmp_home_dir_path + "/data/cart_dmp/cart_coord_dmp/single_traj_training/sample_traj_3D_1.txt", task_servo_rate)
    
    ## Reproduce
    if (time_reproduce_max <= 0.0):
        time_reproduce_max = tau_learn
    if (time_goal_change <= 0.0):
        time_goal_change = time_reproduce_max
    if (tau_reproduce <= 0.0):
        tau_reproduce = tau_learn
    tau = tau_reproduce
    dmp_unroll_init_parameters = DMPUnrollInitParams(critical_states_learn, tau)
    
    steady_state_goal_position = np.array([[0.5, 1.0, 0.0]]).T
    
    cart_dmp.startWithUnrollParams(dmp_unroll_init_parameters)
    
    unroll_ctraj = np.zeros((int(np.round(time_reproduce_max*task_servo_rate) + 1), 4))
    for i in range(int(np.round(time_reproduce_max*task_servo_rate) + 1)):
        time = 1.0 * (i*dt)
        
        [current_state, 
         current_state_local, 
         transform_sys_forcing_term, 
         transform_sys_coupling_term_acc, 
         transform_sys_coupling_term_vel, 
         func_approx_basis_function_vector] = cart_dmp.getNextState(dt, True)
        
        epsilon = sys.float_info.epsilon
        if (np.abs(time - time_goal_change) < (5 * epsilon)):
            steady_state_goal_position = np.array([[0.5, 0.5, 0.0]]).T
            cart_dmp.setNewSteadyStateGoalPosition(steady_state_goal_position)
        
        current_position = current_state.getX()
        unroll_ctraj[i,0] = time
        unroll_ctraj[i,1] = current_position[0,0]
        unroll_ctraj[i,2] = current_position[1,0]
        unroll_ctraj[i,3] = current_position[2,0]
    
    if (os.path.isdir(unroll_ctraj_save_dir_path)):
        np.savetxt(unroll_ctraj_save_dir_path + "/" + unroll_ctraj_save_filename, unroll_ctraj)
    
    return unroll_ctraj, W, mean_A_learn, mean_tau, Ft, Fp, G, cX, cV, PSI

if __name__ == "__main__":
    unroll_ctraj, W, mean_A_learn, mean_tau, Ft, Fp, G, cX, cV, PSI = cart_coord_dmp_single_traj_training_test()