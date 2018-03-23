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
                                   cart_coord_dmp,
                                   is_using_coupling_term=True,
                                   fraction_data_points_included_per_demo_traj=1.0):
    tcloa = cart_coord_dmp.transform_sys_discrete_cart_coord.transform_couplers_list[0]
    if (is_using_coupling_term == False):
        cart_coord_dmp.transform_sys_discrete_cart_coord.transform_couplers_list[0] = None # NOT using the learned obstacle avoidance coupling term, i.e. nominal/baseline unrolling
    
    traj_length = demo_obs_avoid_traj_global.getLength()
    assert (fraction_data_points_included_per_demo_traj <= 1.0)
    unroll_traj_length = int(round(fraction_data_points_included_per_demo_traj * traj_length))
    start_time_global_obs_avoid_demo = demo_obs_avoid_traj_global.getDMPStateAtIndex(0).time[0,0]
    end_time_global_obs_avoid_demo = demo_obs_avoid_traj_global.getDMPStateAtIndex(traj_length-1).time[0,0]
    unroll_dt = (end_time_global_obs_avoid_demo - start_time_global_obs_avoid_demo)/(traj_length - 1.0)
    if (unroll_dt <= 0.0):
        unroll_dt = dt
    unroll_tau = 1.0 * unroll_dt * (traj_length - 1)
    unroll_critical_states_list = [None] * 2
    unroll_critical_states_list[0] = DMPState(cart_coord_dmp_baseline_params['mean_start_global_position'])
    unroll_critical_states_list[-1] = DMPState(cart_coord_dmp_baseline_params['mean_goal_global_position'])
    unroll_critical_states = convertDMPStatesListIntoDMPTrajectory(unroll_critical_states_list)
    
    dmp_unroll_init_params = DMPUnrollInitParams(unroll_critical_states, unroll_tau)
    
    cart_coord_dmp.startWithUnrollParams(dmp_unroll_init_params)
    
    list_ccdmpstate_loa_unroll_global = [None] * unroll_traj_length
    list_ctacc = [None] * unroll_traj_length
    list_pmnn_input_vector = [None] * unroll_traj_length
    
    if (is_using_coupling_term):
        tcloa.setCartCoordDMP(cart_coord_dmp)
        tcloa.setPointObstaclesCCStateGlobal(ObstacleStates(point_obstacles_cart_position_global.T))
    
    for i in range(unroll_traj_length):
# =============================================================================
#         # if unrolling coupling term without dynamics:
#         if (i == 0):
#             tcloa.endeff_ccstate_global = demo_obs_avoid_traj_global.getDMPStateAtIndex(i)
#         else:
#             tcloa.endeff_ccstate_global = demo_obs_avoid_traj_global.getDMPStateAtIndex(i-1)
# =============================================================================
        [current_state_global, 
         _, 
         _, 
         transform_sys_coupling_term_acc, 
         _, 
         _] = cart_coord_dmp.getNextState(unroll_dt, True)
        
        list_ccdmpstate_loa_unroll_global[i] = current_state_global
        list_ctacc[i] = transform_sys_coupling_term_acc
        if (is_using_coupling_term):
            list_pmnn_input_vector[i] = tcloa.pmnn_input_vector.T
    
    if (is_using_coupling_term):
        sub_X_unroll = np.hstack(list_pmnn_input_vector)
    else:
        sub_X_unroll = []
        cart_coord_dmp.transform_sys_discrete_cart_coord.transform_couplers_list[0] = tcloa # restore the learned obstacle avoidance coupling term attachment to the transformation system
    sub_Ct_unroll = np.hstack(list_ctacc)
    ccdmp_loa_unroll_global_traj = convertDMPStatesListIntoDMPTrajectory(list_ccdmpstate_loa_unroll_global)
    
    return sub_X_unroll, sub_Ct_unroll, ccdmp_loa_unroll_global_traj