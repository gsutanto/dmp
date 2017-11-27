#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import numpy as np
import numpy.matlib
import os
import sys
import copy
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../cart_dmp/cart_coord_dmp/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_coupling/base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
from DMPState import *
from DMPTrajectory import *
from ObstacleStates import *
from TransformSystemDiscrete import *
from CartesianCoordDMP import *
from CartesianCoordTransformer import *
from DataIO import *
from utilities import *

class TransformCouplingLearnObsAvoid(TransformCoupling, object):
    'Class defining learnable obstacle avoidance coupling terms for DMP transformation systems.'
    
    def __init__(self, loa_parameters, tau_system, cart_coord_dmp=None,
                 point_obstacles_cart_state_global=None, name=""):
        super(TransformCouplingLearnObsAvoid, self).__init__(3, name)
        self.loa_param = loa_parameters
        self.tau_sys = tau_system
        self.point_obstacles_ccstate_global = point_obstacles_cart_state_global
        self.ccdmp = cart_coord_dmp
        if ((self.ccdmp is not None) and (self.ccdmp.isValid())):
            self.cart_coord_transformer = self.ccdmp.cart_coord_transformer
            self.ctraj_hmg_transform_global_to_local_matrix = self.ccdmp.ctraj_hmg_transform_global_to_local_matrix
            self.endeff_ccstate_local = self.ccdmp.transform_sys_discrete_cart_coord.current_state
            if (self.point_obstacles_ccstate_global is not None):
                self.point_obstacles_ccstate_local = self.cart_coord_transformer.computeCTrajAtNewCoordSys(self.point_obstacles_ccstate_global, 
                                                                                                           self.ctraj_hmg_transform_global_to_local_matrix)
            else:
                self.point_obstacles_ccstate_local = None
        else:
            self.cart_coord_transformer = None
            self.ctraj_hmg_transform_global_to_local_matrix = None
            self.endeff_ccstate_local = None
            self.point_obstacles_ccstate_local = None
    
    def isValid(self):
        assert (super(TransformCouplingLearnObsAvoid, self).isValid())
        assert (self.loa_param is not None)
        assert (self.tau_sys is not None)
        assert (self.ccdmp is not None)
        assert (self.cart_coord_transformer is not None)
        assert (self.ctraj_hmg_transform_global_to_local_matrix is not None)
        assert (self.endeff_ccstate_local is not None)
        assert (self.point_obstacles_ccstate_global is not None)
        assert (self.point_obstacles_ccstate_local is not None)
        assert (self.loa_param.isValid())
        assert (self.tau_sys.isValid())
        assert (self.ccdmp.isValid())
        assert (self.cart_coord_transformer.isValid())
        assert (self.ctraj_hmg_transform_global_to_local_matrix.shape == (4, 4))
        assert (self.endeff_ccstate_local.isValid())
        assert (self.point_obstacles_ccstate_global.isValid())
        assert (self.point_obstacles_ccstate_local.isValid())
        return True
    
    def setCartCoordDMP(self, cart_coord_dmp):
        assert (cart_coord_dmp is not None)
        assert (cart_coord_dmp.isValid())
        
        self.ccdmp = cart_coord_dmp
        self.cart_coord_transformer = self.ccdmp.cart_coord_transformer
        self.ctraj_hmg_transform_global_to_local_matrix = self.ccdmp.ctraj_hmg_transform_global_to_local_matrix
        self.endeff_ccstate_local = self.ccdmp.transform_sys_discrete_cart_coord.current_state
        if (self.point_obstacles_ccstate_global is not None):
            self.point_obstacles_ccstate_local = self.cart_coord_transformer.computeCTrajAtNewCoordSys(self.point_obstacles_ccstate_global, 
                                                                                                       self.ctraj_hmg_transform_global_to_local_matrix)
        else:
            self.point_obstacles_ccstate_local = None
        
        return None
    
    def computeSubFeatMatAndSubTargetCt(self, demo_obs_avoid_traj_global, point_obstacles_cart_position_global, 
                                        dt, cart_coord_dmp_baseline_params, cart_coord_dmp):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this TransformCouplingLearnObsAvoid is invalid!"
        assert (self.tau_sys == cart_coord_dmp.tau_sys)
        
        max_critical_point_distance_baseline_vs_oa_demo = 0.1 # in meter
        
        traj_length = demo_obs_avoid_traj_global.getLength()
        
        start_state_global_obs_avoid_demo = demo_obs_avoid_traj_global.getDMPStateAtIndex(0)
        goal_state_global_obs_avoid_demo = demo_obs_avoid_traj_global.getDMPStateAtIndex(traj_length-1)
        
        traj_dt = (goal_state_global_obs_avoid_demo.time[0,0] - start_state_global_obs_avoid_demo.time[0,0])/(traj_length - 1.0)
        if (traj_dt <= 0.0):
            traj_dt = dt
        
        start_position_global_obs_avoid_demo = start_state_global_obs_avoid_demo.X
        goal_position_global_obs_avoid_demo = goal_state_global_obs_avoid_demo.X
        
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
        
        demo_obs_avoid_traj_local = self.cart_coord_transformer.computeCTrajAtNewCoordSys(demo_obs_avoid_traj_global,
                                                                                          cart_coord_dmp_baseline_params['T_global_to_local_H'])
        
        point_obstacles_cart_position_local = self.cart_coord_transformer.computeCPosAtNewCoordSys(point_obstacles_cart_position_global.T,
                                                                                                   cart_coord_dmp_baseline_params['T_global_to_local_H'])
        point_obstacles_cart_position_local = point_obstacles_cart_position_local.T
        
        [sub_Ct_target, _, 
         sub_phase_PSI, sub_phase_X, sub_phase_V, 
         _, _, 
         _] = cart_coord_dmp.transform_sys_discrete.getTargetCouplingTermTraj(demo_obs_avoid_traj_local, 
                                                                              1.0/traj_dt,
                                                                              cart_coord_dmp_baseline_params['mean_goal_local_position'])
        
        sub_X = self.constructObsAvoidViconFeatMat(demo_obs_avoid_traj_local,
                                                   point_obstacles_cart_position_local,
                                                   traj_dt)
        
        return sub_X, sub_Ct_target, sub_phase_PSI, sub_phase_V, sub_phase_X, is_good_demo
    
    def constructObsAvoidViconFeatMat(self, demo_obs_avoid_traj_local,
                                      point_obstacles_cart_position_local,
                                      dt):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this TransformCouplingLearnObsAvoid is invalid!"
        
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
        
        self.point_obstacles_ccstate_local.X = point_obstacles_cart_position_local.T
        self.point_obstacles_ccstate_local.Xd = np.zeros(point_obstacles_cart_position_local.T.shape)
        self.point_obstacles_ccstate_local.Xdd = np.zeros(point_obstacles_cart_position_local.T.shape)
        
        self.tau_sys.setTauBase(tau)
        
        for i in xrange(traj_length):
            self.endeff_ccstate_local.X = Y_obs_local_shifted[:,i].reshape(3,1)
            self.endeff_ccstate_local.Xd = Yd_obs_local_shifted[:,i].reshape(3,1)
            self.endeff_ccstate_local.Xdd = Ydd_obs_local_shifted[:,i].reshape(3,1)
            self.endeff_ccstate_local.time = demo_obs_avoid_traj_local.time[:,i].reshape(1,1)
            
            self.point_obstacles_ccstate_local.time = demo_obs_avoid_traj_local.time[:,i].reshape(1,1)
            
            sub_X_vector = self.computeObsAvoidCtFeat(self.point_obstacles_ccstate_local,
                                                      self.endeff_ccstate_local)
            
            list_sub_X_vectors[i] = sub_X_vector
        
        sub_X = np.hstack(list_sub_X_vectors)
        
        return sub_X
    
    def computeObsAvoidCtFeat(self, point_obstacles_cart_coord_state_local,
                              endeff_cart_coord_state_local):
        tau_rel = self.tau_sys.getTauRelative()
        
        x3 = endeff_cart_coord_state_local.X
        v3 = endeff_cart_coord_state_local.Xd
        
        tau_v3 = tau_rel * v3
        
        O3 = point_obstacles_cart_coord_state_local.X
        
        o3_center = np.mean(O3, axis=1).reshape(3,1)
        
        N_obs = point_obstacles_cart_coord_state_local.getLength()
        
        N_closest_obs = 3
        
        o3_centerx = o3_center - x3
        
        OX3 = (O3 - np.matlib.repmat(x3, 1, N_obs))
        
        # norm of euclidean distances between the endeff and each obstacle
        OX3_norm = np.linalg.norm(OX3, axis=0)
        
        closest_obs_idx = np.argsort(OX3_norm)
        
        selected_idx_OX3 = closest_obs_idx[0:N_closest_obs]
        
        reshaped_sel_OX3_vecs = OX3[:,selected_idx_OX3].reshape((3*N_closest_obs),1)
        
        ox3 = OX3[:,closest_obs_idx[0]].reshape(3,1)
        num_thresh = 0.0
        if ((np.linalg.norm(v3) > num_thresh) and (np.linalg.norm(ox3) > num_thresh)):
            cos_theta = (np.matmul(ox3.T, v3)[0,0])/(np.linalg.norm(ox3)*np.linalg.norm(v3))
#            clamped_cos_theta = cos_theta
#            if (clamped_cos_theta < 0.0):
#                clamped_cos_theta = 0.0
        else:
            cos_theta = -1.0
#            clamped_cos_theta = 0.0
        
        sub_X_vector = np.vstack([o3_centerx, 
                                  reshaped_sel_OX3_vecs, 
                                  tau_v3,
                                  np.linalg.norm(o3_centerx),
                                  cos_theta])
        
        return sub_X_vector