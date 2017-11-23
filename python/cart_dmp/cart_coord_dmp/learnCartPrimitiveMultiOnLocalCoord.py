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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../cart_dmp/cart_coord_dmp'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
from DMPTrajectory import *
from DMPState import *
from TauSystem import *
from DMPUnrollInitParams import *
from CanonicalSystemDiscrete import *
from CartesianCoordTransformer import *
from CartesianCoordDMP import *
from utilities import *

def learnCartPrimitiveMultiOnLocalCoord(cart_global_traj, train_data_dt,
                                        dmp_basis_funcs_size=25, canonical_order=2,
                                        ctraj_local_coordinate_frame_selection=GSUTANTO_LOCAL_COORD_FRAME,
                                        unroll_traj_length=-1,
                                        unroll_dt=None, 
                                        is_using_scaling=[False] * 3):
    # is_using_scaling=[False] * 3 # default is NOT using scaling on DMPs
    if (unroll_dt == None):
        unroll_dt = train_data_dt
    
    D = 3
    N_demo = len(cart_global_traj)
    
    tau_sys = TauSystem(MIN_TAU)
    canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
    cart_dmp = CartesianCoordDMP(dmp_basis_funcs_size, canonical_sys_discr, ctraj_local_coordinate_frame_selection)
    cart_dmp.setScalingUsage(is_using_scaling)
    
    [critical_states_learn, 
     W, mean_A_learn, mean_tau, 
     Ft, Fp, 
     G, cX, cV, 
     PSI] = cart_dmp.learnGetDefaultUnrollParams(cart_global_traj, 1.0/train_data_dt)
    
    if (unroll_traj_length == -1):
        unroll_tau = mean_tau
        unroll_traj_length = int(np.round(unroll_tau / dt)) + 1
    else:
        unroll_tau = 1.0 * unroll_dt * (unroll_traj_length - 1)
    
    ## Reproduce
    dmp_unroll_init_params = DMPUnrollInitParams(critical_states_learn, unroll_tau)
    
    cart_dmp.startWithUnrollParams(dmp_unroll_init_params)
    
    cart_coord_dmp_params = {}
    cart_coord_dmp_params['train_data_dt'] = train_data_dt
    cart_coord_dmp_params['model_size'] = dmp_basis_funcs_size
    cart_coord_dmp_params['canonical_order'] = canonical_order
    cart_coord_dmp_params['W'] = W
    cart_coord_dmp_params['A_learn'] = mean_A_learn
    cart_coord_dmp_params['mean_tau'] = mean_tau
    cart_coord_dmp_params['mean_start_global_position'] = copy.copy(cart_dmp.mean_start_global_position)
    cart_coord_dmp_params['mean_goal_global_position'] = copy.copy(cart_dmp.mean_goal_global_position)
    cart_coord_dmp_params['mean_start_local_position'] = copy.copy(cart_dmp.mean_start_local_position)
    cart_coord_dmp_params['mean_goal_local_position'] = copy.copy(cart_dmp.mean_goal_local_position)
    cart_coord_dmp_params['ctraj_local_coordinate_frame_selection'] = ctraj_local_coordinate_frame_selection
    cart_coord_dmp_params['T_local_to_global_H'] = copy.copy(cart_dmp.ctraj_hmg_transform_local_to_global_matrix)
    cart_coord_dmp_params['T_global_to_local_H'] = copy.copy(cart_dmp.ctraj_hmg_transform_global_to_local_matrix)
    cart_coord_dmp_params['unroll_dt'] = unroll_dt
    cart_coord_dmp_params['unroll_tau'] = unroll_tau
    
    list_cart_coord_dmpstate_unroll_fit_global = [None] * unroll_traj_length
    list_cart_coord_dmpstate_unroll_fit_local = [None] * unroll_traj_length
    list_forcing_term = [None] * unroll_traj_length
        
    for i in range(unroll_traj_length):
        time = 1.0 * (i*unroll_dt)
        
        [current_state_global, 
         current_state_local, 
         transform_sys_forcing_term, 
         transform_sys_coupling_term_acc, 
         transform_sys_coupling_term_vel, 
         func_approx_basis_function_vector] = cart_dmp.getNextState(unroll_dt, True)
        
        list_cart_coord_dmpstate_unroll_fit_global[i] = current_state_global
        list_cart_coord_dmpstate_unroll_fit_local[i] = current_state_local
        list_forcing_term[i] = transform_sys_forcing_term
    
    cart_coord_dmp_unroll_fit_global_traj = convertDMPStatesListIntoDMPTrajectory(list_cart_coord_dmpstate_unroll_fit_global)
    cart_coord_dmp_unroll_fit_local_traj = convertDMPStatesListIntoDMPTrajectory(list_cart_coord_dmpstate_unroll_fit_local)
    Ffit = np.hstack(list_forcing_term)
    
    return cart_coord_dmp_params, cart_coord_dmp_unroll_fit_global_traj, cart_coord_dmp_unroll_fit_local_traj, Ffit