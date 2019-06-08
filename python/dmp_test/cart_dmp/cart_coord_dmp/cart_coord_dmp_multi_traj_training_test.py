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
    unroll_ctraj_list = list()
    unroll_log = list()
    
    for k in range(2):
        unroll_ctraj = np.zeros((unroll_ctraj_length, 4))
        if (k == 0): # Without Obstacle
            sub_cart_coord_dmp_training_path = "/data/dmp_coupling/learn_obs_avoid/static_obs/data_sph_new/SubjectNo1/baseline/endeff_trajs/"
        elif (k == 1): # With Obstacle
            sub_cart_coord_dmp_training_path = "/data/dmp_coupling/learn_obs_avoid/static_obs/data_sph_new/SubjectNo1/1/endeff_trajs/"
        
        [critical_states_learn, 
         W, mean_A_learn, mean_tau, 
         Ft, Fp, G, cX, cV, 
         PSI] = cart_dmp.learnFromPath(amd_clmc_dmp_home_dir_path + sub_cart_coord_dmp_training_path, 
                                       task_servo_rate, 
                                       start_column_idx=1, time_column_idx=0)
        
        ## Reproduce
        dmp_unroll_init_params = DMPUnrollInitParams(critical_states_learn, tau)
        
        cart_dmp.startWithUnrollParams(dmp_unroll_init_params)
        
        for i in range(int(np.round(time_reproduce_max*task_servo_rate) + 1)):
            time = 1.0 * (i*dt)
            
            [current_state, 
             transform_sys_forcing_term, 
             transform_sys_coupling_term_acc, 
             transform_sys_coupling_term_vel, 
             func_approx_basis_function_vector] = cart_dmp.getNextState(dt, True)
            
            current_position = current_state.getX()
            unroll_ctraj[i,0] = time
            unroll_ctraj[i,1] = current_position[0,0]
            unroll_ctraj[i,2] = current_position[1,0]
            unroll_ctraj[i,3] = current_position[2,0]
        unroll_ctraj_list.append(copy.deepcopy(unroll_ctraj))
        unroll_log.append(copy.deepcopy(unroll_ctraj))
        
        unroll_learned_params = np.zeros((model_size+1+4+4+4,4))
        unroll_learned_params[0:model_size,0:3] = W.T
        unroll_learned_params[model_size:model_size+1,0:3] = mean_A_learn.T
        unroll_learned_params[model_size,3] = mean_tau
        unroll_learned_params[model_size+1:model_size+2,0:3] = cart_dmp.mean_start_global_position.T
        unroll_learned_params[model_size+2:model_size+3,0:3] = cart_dmp.mean_goal_global_position.T
        unroll_learned_params[model_size+3:model_size+4,0:3] = cart_dmp.mean_start_local_position.T
        unroll_learned_params[model_size+4:model_size+5,0:3] = cart_dmp.mean_goal_local_position.T
        unroll_learned_params[model_size+5:model_size+9,:] = cart_dmp.ctraj_hmg_transform_local_to_global_matrix
        unroll_learned_params[model_size+9:model_size+13,:] = cart_dmp.ctraj_hmg_transform_global_to_local_matrix
        unroll_log.append(copy.deepcopy(unroll_learned_params))
    
    unroll_ctraj_concatenated = np.vstack(unroll_ctraj_list)
    
    if (os.path.isdir(unroll_ctraj_save_dir_path)):
        np.savetxt(unroll_ctraj_save_dir_path + "/" + unroll_ctraj_save_filename, np.vstack(unroll_log))
    
    return unroll_ctraj_concatenated, W, mean_A_learn, mean_tau, Ft, Fp, G, cX, cV, PSI

if __name__ == "__main__":
    unroll_ctraj_concatenated, W, mean_A_learn, mean_tau, Ft, Fp, G, cX, cV, PSI = cart_coord_dmp_multi_traj_training_test("../../../../", 2, 0.5, 0.5)