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

def cart_coord_dmp_multi_traj_training_test(amd_clmc_dmp_home_dir_path="../../../../", 
                                             canonical_order=2, time_reproduce_max=0.5, 
                                             tau_reproduce=0.5, unroll_ctraj_save_dir_path="", 
                                             unroll_ctraj_save_filename=""):
    task_servo_rate = 420.0
    dt = 1.0/task_servo_rate
    model_size = 25
    tau = 0.5
    
    assert ((canonical_order >= 1) and (canonical_order <= 2))
    assert (time_reproduce_max >= 0.0)
    assert (tau_reproduce >= 0.0)
    
    if (time_reproduce_max <= 0.5):
        time_reproduce_max = 0.5
    
    tau_sys = TauSystem(dt, tau)
    canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
    cart_dmp = CartesianCoordDMP(model_size, canonical_sys_discr, GSUTANTO_LOCAL_COORD_FRAME)
    tau = tau_reproduce
    
    unroll_ctraj_length = int(np.round(time_reproduce_max*task_servo_rate) + 1)
    unroll_ctraj = np.zeros((2 * unroll_ctraj_length, 4))
    
    for k in range(2):
        if (k == 0): # Without Obstacle
            sub_cart_coord_dmp_training_path = "/data/dmp_coupling/learn_obs_avoid/static_obs/data_sph_new/SubjectNo1/baseline/endeff_trajs/"
        elif (k == 1): # With Obstacle
            sub_cart_coord_dmp_training_path = "/data/dmp_coupling/learn_obs_avoid/static_obs/data_sph_new/SubjectNo1/1/endeff_trajs/"
        
        [critical_states_learn, 
         W, mean_A_learn, mean_tau, 
         Ft, Fp, G, cX, cV, 
         PSI] = cart_dmp.learnFromPath(amd_clmc_dmp_home_dir_path + sub_cart_coord_dmp_training_path, task_servo_rate)
            
        ## Reproduce
        dmp_unroll_init_params = DMPUnrollInitParams(critical_states_learn, tau)
        
        cart_dmp.startWithUnrollParams(dmp_unroll_init_params)
        
        start_logging_idx = (k*unroll_ctraj_length)
        
        for i in range(int(np.round(time_reproduce_max*task_servo_rate) + 1)):
            time = 1.0 * (i*dt)
            
            [current_state, 
             current_state_local, 
             transform_sys_forcing_term, 
             transform_sys_coupling_term_acc, 
             transform_sys_coupling_term_vel, 
             func_approx_basis_function_vector] = cart_dmp.getNextState(dt, True)
            
            current_position = current_state.getX()
            unroll_ctraj[start_logging_idx+i,0] = time
            unroll_ctraj[start_logging_idx+i,1] = current_position[0,0]
            unroll_ctraj[start_logging_idx+i,2] = current_position[1,0]
            unroll_ctraj[start_logging_idx+i,3] = current_position[2,0]
    
    if (os.path.isdir(unroll_ctraj_save_dir_path)):
        np.savetxt(unroll_ctraj_save_dir_path + "/" + unroll_ctraj_save_filename, unroll_ctraj)
    
    return unroll_ctraj, W, mean_A_learn, mean_tau, Ft, Fp, G, cX, cV, PSI

if __name__ == "__main__":
    unroll_ctraj, W, mean_A_learn, mean_tau, Ft, Fp, G, cX, cV, PSI = cart_coord_dmp_multi_traj_training_test("../../../../", 2, 0.5, 0.5)