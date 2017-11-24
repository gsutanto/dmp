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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../cart_dmp/cart_coord_dmp/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from DMPState import *
from DMPTrajectory import *
from TransformSystemDiscrete import *
from CartesianCoordDMP import *
from CartesianCoordTransformer import *
from constructObsAvoidViconFeatMat import *
from DataIO import *
from utilities import *

def computeSubFeatMatAndSubTargetCt(demo_obs_avoid_traj_global, point_obstacles_cart_position_global, 
                                    dt, cart_coord_dmp_baseline_params, cart_coord_dmp):
    max_critical_point_distance_baseline_vs_oa_demo = 0.1 # in meter
    
    start_position_global_obs_avoid_demo = demo_obs_avoid_traj_global.getDMPStateAtIndex(0).getX()
    goal_position_global_obs_avoid_demo = demo_obs_avoid_traj_global.getDMPStateAtIndex(demo_obs_avoid_traj_global.getLength()-1).getX()
    
    # some error checking on the demonstration:
    # (distance between start position of baseline DMP and obstacle avoidance demonstration,
    #  as well as distance between goal position of baseline DMP and obstacle avoidance demonstration,
    #  both should be lower than max_tolerable_distance, otherwise the demonstrated obstacle avoidance
    #  trajectory is flawed):
    if ((np.linalg.norm(start_position_global_obs_avoid_demo - cart_coord_dmp_baseline_params['mean_start_global_position']) > max_critical_point_distance_baseline_vs_oa_demo) or
        (np.linalg.norm(goal_position_global_obs_avoid_demo - cart_coord_dmp_baseline_params['mean_goal_global_position']) > max_critical_point_distance_baseline_vs_oa_demo)):
        print ('ERROR: Critical position distance between baseline and obstacle avoidance demonstration is beyond tolerable threshold!!!')
        is_good_demo = False
    else:
        is_good_demo = True
    
    cart_coord_dmp.setParams(cart_coord_dmp_baseline_params['W'], cart_coord_dmp_baseline_params['A_learn'])
    
    demo_obs_avoid_traj_local = cart_coord_dmp.cart_coord_transformer.computeCTrajAtNewCoordSys(demo_obs_avoid_traj_global,
                                                                                                cart_coord_dmp_baseline_params['T_global_to_local_H'])
    
    point_obstacles_cart_position_local = cart_coord_dmp.cart_coord_transformer.computeCPosAtNewCoordSys(point_obstacles_cart_position_global.T,
                                                                                                         cart_coord_dmp_baseline_params['T_global_to_local_H'])
    point_obstacles_cart_position_local = point_obstacles_cart_position_local.T
    
    [sub_Ct_target, _, 
     sub_phase_PSI, sub_phase_X, sub_phase_V, 
     _, _, 
     _] = cart_coord_dmp.transform_sys_discrete.getTargetCouplingTermTraj(demo_obs_avoid_traj_local, 
                                                                          1.0/dt,
                                                                          cart_coord_dmp_baseline_params['mean_goal_local_position'])
    
    sub_X = constructObsAvoidViconFeatMat(demo_obs_avoid_traj_local,
                                          point_obstacles_cart_position_local,
                                          dt,
                                          cart_coord_dmp_baseline_params,
                                          cart_coord_dmp)
    
    return sub_X, sub_Ct_target, sub_phase_PSI, sub_phase_V, sub_phase_X, is_good_demo