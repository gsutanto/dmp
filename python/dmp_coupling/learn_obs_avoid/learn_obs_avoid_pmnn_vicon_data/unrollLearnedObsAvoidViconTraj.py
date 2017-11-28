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
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../cart_dmp/cart_coord_dmp/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_coupling/learn_obs_avoid/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_coupling/learn_obs_avoid/vicon/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from DMPState import *
from DMPTrajectory import *
from TauSystem import *
from DMPUnrollInitParams import *
from CartesianCoordDMP import *
from CartesianCoordTransformer import *
from DataIO import *
from utilities import *
from ObstacleStates import *
from TransformCouplingLearnObsAvoid import *
from TCLearnObsAvoidFeatureParameter import *

def unrollLearnedObsAvoidViconTraj(demo_obs_avoid_traj_global,
                                   point_obstacles_cart_position_global, 
                                   dt,
                                   cart_coord_dmp_baseline_params,
                                   cart_coord_dmp):
    tcloa = cart_coord_dmp.transform_sys_discrete_cart_coord.transform_couplers_list[0]
    
    unroll_traj_length = demo_obs_avoid_traj_global.getLength()
    start_time_global_obs_avoid_demo = demo_obs_avoid_traj_global.getDMPStateAtIndex(0).time[0,0]
    end_time_global_obs_avoid_demo = demo_obs_avoid_traj_global.getDMPStateAtIndex(unroll_traj_length-1).time[0,0]
    unroll_dt = (end_time_global_obs_avoid_demo - start_time_global_obs_avoid_demo)/(unroll_traj_length - 1.0)
    if (unroll_dt <= 0.0):
        unroll_dt = dt
    unroll_tau = 1.0 * unroll_dt * (unroll_traj_length - 1)
    unroll_critical_states_list = [None] * 2
    unroll_critical_states_list[0] = DMPState(cart_coord_dmp_baseline_params['mean_start_global_position'])
    unroll_critical_states_list[-1] = DMPState(cart_coord_dmp_baseline_params['mean_goal_global_position'])
    unroll_critical_states = convertDMPStatesListIntoDMPTrajectory(unroll_critical_states_list)
    
    dmp_unroll_init_params = DMPUnrollInitParams(unroll_critical_states, unroll_tau)
    
    cart_coord_dmp.startWithUnrollParams(dmp_unroll_init_params)
    
    list_ccdmpstate_loa_unroll_global = [None] * unroll_traj_length
    list_ctacc = [None] * unroll_traj_length
    list_pmnn_input_vector = [None] * unroll_traj_length
    
    tcloa.setCartCoordDMP(cart_coord_dmp)
    tcloa.setPointObstaclesCCStateGlobal(ObstacleStates(point_obstacles_cart_position_global.T))
    tcloa.func_approx_basis_functions = cart_coord_dmp.func_approx_discrete.getBasisFunctionTensor(tcloa.canonical_sys_discrete.getCanonicalPosition()).T
    
    for i in range(unroll_traj_length):
        [current_state_global, 
         _, 
         _, 
         transform_sys_coupling_term_acc, 
         _, 
         tcloa.func_approx_basis_functions] = cart_coord_dmp.getNextState(unroll_dt, True)
        tcloa.func_approx_basis_functions = tcloa.func_approx_basis_functions.T
        
        list_ccdmpstate_loa_unroll_global[i] = current_state_global
        list_ctacc[i] = transform_sys_coupling_term_acc
        list_pmnn_input_vector[i] = tcloa.pmnn_input_vector.T
    
    sub_X_unroll = np.hstack(list_pmnn_input_vector)
    sub_Ct_unroll = np.hstack(list_ctacc)
    ccdmp_loa_unroll_global_traj = convertDMPStatesListIntoDMPTrajectory(list_ccdmpstate_loa_unroll_global)
    
    return sub_X_unroll, sub_Ct_unroll, ccdmp_loa_unroll_global_traj