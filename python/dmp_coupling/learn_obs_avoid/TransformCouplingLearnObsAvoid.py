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
sys.path.append(os.path.join(os.path.dirname(__file__), '../base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../cart_dmp/cart_coord_dmp/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
from DMPState import *
from DMPTrajectory import *
from TransformSystemDiscrete import *
from CartesianCoordDMP import *
from CartesianCoordTransformer import *
from DataIO import *
from utilities import *

class TransformCouplingLearnObsAvoid(TransformCoupling, object):
    'Class defining learnable obstacle avoidance coupling terms for DMP transformation systems.'
    
    def __init__(self, loa_parameters, tau_system,
                 endeff_cart_state_global, point_obstacles_cart_state_global, 
                 cart_traj_homogeneous_transform_global_to_local_matrix, name=""):
        super(TransformCouplingLearnObsAvoid, self).__init__(3, name)
        self.loa_param = loa_parameters
        self.tau_sys = tau_system
        self.cart_coord_transformer = CartesianCoordTransformer()
        self.endeff_ccstate_global = endeff_cart_state_global
        self.endeff_ccstate_local = DMPState(np.zeros((3,1)))
        self.point_obstacles_ccstate_global = point_obstacles_cart_state_global
        self.ctraj_hmg_transform_global_to_local_matrix = cart_traj_homogeneous_transform_global_to_local_matrix
    
    def computeSubFeatMatAndSubTargetCt(self, demo_obs_avoid_traj_global, point_obstacles_cart_position_global, 
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
        
        sub_X = self.constructObsAvoidViconFeatMat(demo_obs_avoid_traj_local,
                                                   point_obstacles_cart_position_local,
                                                   dt,
                                                   cart_coord_dmp_baseline_params,
                                                   cart_coord_dmp)
        
        return sub_X, sub_Ct_target, sub_phase_PSI, sub_phase_V, sub_phase_X, is_good_demo
    
    def constructObsAvoidViconFeatMat(self, demo_obs_avoid_traj_local,
                                      point_obstacles_cart_position_local,
                                      dt,
                                      cart_coord_dmp_baseline_params,
                                      cart_coord_dmp):
        
        cart_coord_dmp.setParams(cart_coord_dmp_baseline_params['W'], cart_coord_dmp_baseline_params['A_learn'])
        
        Y_obs_local = demo_obs_avoid_traj_local.getX()
        Yd_obs_local = demo_obs_avoid_traj_local.getXd()
        Ydd_obs_local = demo_obs_avoid_traj_local.getXdd()
        
        # based on previous test on learning on synthetic dataset
        # (see for example in amd_clmc_dmp/matlab/dmp_coupling/learn_obs_avoid/learn_obs_avoid_fixed_learning_algo/main.m ), 
        # the ground-truth feature matrix is attained if
        # the trajectory is delayed 1 time step:
        is_demo_traj_shifted = True
        
        traj_length = Y_obs_local.shape[1]
        tau = (traj_length-1)*dt
        
        Y_obs_local_shifted = Y_obs_local
        Yd_obs_local_shifted = Yd_obs_local
        Ydd_obs_local_shifted = Ydd_obs_local
        if (is_demo_traj_shifted):
            Y_obs_local_shifted[:,1:traj_length] = Y_obs_local_shifted[:,0:(traj_length-1)]
            Yd_obs_local_shifted[:,1:traj_length] = Yd_obs_local_shifted[:,0:(traj_length-1)]
            Ydd_obs_local_shifted[:,1:traj_length] = Ydd_obs_local_shifted[:,0:(traj_length-1)]
        
        list_sub_X_vectors = [None] * traj_length
        
        endeff_state = {}
        
        for i in xrange(traj_length):
            endeff_state['X'] = Y_obs_local_shifted[:,i]
            endeff_state['Xd'] = Yd_obs_local_shifted[:,i]
            endeff_state['Xdd'] = Ydd_obs_local_shifted[:,i]
            
            sub_X_vector = self.computeObsAvoidCtFeat(point_obstacles_cart_position_local,
                                                      endeff_state,
                                                      tau,
                                                      cart_coord_dmp_baseline_params,
                                                      cart_coord_dmp)
            
            list_sub_X_vectors[i] = sub_X_vector
        
        sub_X = np.hstack(list_sub_X_vectors)
        
        return sub_X
    
    def computeObsAvoidCtFeat(self, point_obstacles_cart_position_local,
                              endeff_state,
                              tau,
                              cart_coord_dmp_baseline_params,
                              cart_coord_dmp):
        
        return sub_X_vector