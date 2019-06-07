#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import numpy as np
import tensorflow as tf
import os
import sys
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
from CartesianCoordDMP import *
from CartesianCoordTransformer import *
from DMPDiscrete import *
from TransformSystemDiscrete import *
from DMPTrajectory import *
from DMPState import *
from utilities import *

class CartesianCoordDMPTensorflow(CartesianCoordDMP, object):
    """
    Class for discrete Cartesian Coordinate (x-y-z) DMPs, implemented with TensorFlow.
    Only valid for 2nd-order canonical system.
    Terminology:
    z_vec             = augmented_state_vector               (size ((3*dmp_num_dimensions)+1+1)X1)
    x_vec             = state_position_vector                (size           dmp_num_dimensionsX1)
    v_vec             = state_velocity_vector**              (size           dmp_num_dimensionsX1)
    g_vec             = goal_state_position_vector           (size           dmp_num_dimensionsX1)
    p_vec             = phase_variable_vector                (size                            1X1)
    u_vec             = phase_velocity_vector                (size                            1X1)
    G_vec             = steady_state_goal_position_vector    (size           dmp_num_dimensionsX1)
    ft                = forcing_term
    ft_kernel_mu_vec  = forcing_term_kernel_center_vector    (size                   model_sizeX1)
    ft_kernel_chi_vec = forcing_term_kernel_bandwidth_vector (size                   model_sizeX1)
    ft_W_mat          = forcing_term_weight_matrix           (size           dmp_num_dimensionsXmodel_size)
    ct                = coupling_term
    
    ** Please note that the v_vec here is the time derivative of x_vec,
       which is different from 
       v = TransformSystemDiscrete.current_velocity_state.X 
         = tau * xd 
         = tau * TransformSystemDiscrete.current_state.Xd
       as implemented in TransformSystemDiscrete.
    """
    
    def isValid(self):
        assert (super(CartesianCoordDMPTensorflow, self).isValid())
        assert (self.canonical_sys_discrete.order == 2)
        return True
    
    def TFExtractStateComponents(self, 
                                 tf_z_vec):
        tf.assert_equal(tf.shape(tf_z_vec)[0], ((3 * self.dmp_num_dimensions) + 1 + 1))
        tf.assert_equal(tf.shape(tf_z_vec)[1], 1)
        
        tf_x_vec = tf.slice(tf_z_vec, [0 * self.dmp_num_dimensions, 0], [self.dmp_num_dimensions, 1])
        tf_v_vec = tf.slice(tf_z_vec, [1 * self.dmp_num_dimensions, 0], [self.dmp_num_dimensions, 1])
        tf_g_vec = tf.slice(tf_z_vec, [2 * self.dmp_num_dimensions, 0], [self.dmp_num_dimensions, 1])
        tf_p_vec = tf.slice(tf_z_vec, [(3 * self.dmp_num_dimensions) + 0, 0], [1, 1])
        tf_u_vec = tf.slice(tf_z_vec, [(3 * self.dmp_num_dimensions) + 1, 0], [1, 1])
        
        return [tf_x_vec, 
                tf_v_vec, 
                tf_g_vec, 
                tf_p_vec, 
                tf_u_vec]
        
    def TFAugmentStateComponents(self, 
                                 tf_x_vec, 
                                 tf_v_vec, 
                                 tf_g_vec, 
                                 tf_p_vec, 
                                 tf_u_vec):
        tf.assert_equal(tf.shape(tf_x_vec)[0], self.dmp_num_dimensions)
        tf.assert_equal(tf.shape(tf_x_vec)[1], 1)
        
        tf.assert_equal(tf.shape(tf_v_vec)[0], self.dmp_num_dimensions)
        tf.assert_equal(tf.shape(tf_v_vec)[1], 1)
        
        tf.assert_equal(tf.shape(tf_g_vec)[0], self.dmp_num_dimensions)
        tf.assert_equal(tf.shape(tf_g_vec)[1], 1)
        
        tf.assert_equal(tf.shape(tf_p_vec)[0], 1)
        tf.assert_equal(tf.shape(tf_p_vec)[1], 1)
        
        tf.assert_equal(tf.shape(tf_u_vec)[0], 1)
        tf.assert_equal(tf.shape(tf_u_vec)[1], 1)
        
        tf_z_vec = tf.concat([tf_x_vec, 
                              tf_v_vec, 
                              tf_g_vec, 
                              tf_p_vec, 
                              tf_u_vec
                              ], axis=0)
        
        tf.assert_equal(tf.shape(tf_z_vec)[0], ((3 * self.dmp_num_dimensions) + 1 + 1))
        tf.assert_equal(tf.shape(tf_z_vec)[1], 1)
        
        return tf_z_vec
    
    def TFComputeNextAugmentedState(self,
                                    tf_curr_z_vec, 
                                    tf_G_vec, 
                                    tf_ft_kernel_mu_vec, 
                                    tf_ft_kernel_chi_vec, 
                                    tf_ft_W_mat, 
                                    tf_ft_scaler_vec, 
                                    tf_ct_vec, 
                                    tf_tau_relative_sca, 
                                    tf_dt_sca, 
                                    tf_alpha_v_sca=25.0, 
                                    tf_beta_v_sca=25.0/4.0, 
                                    tf_alpha_g_sca=25.0/2.0, 
                                    tf_alpha_u_sca=25.0, 
                                    tf_beta_u_sca=25.0/4.0):
        [tf_curr_x_vec, 
         tf_curr_v_vec, 
         tf_curr_g_vec, 
         tf_curr_p_vec, 
         tf_curr_u_vec
         ] = self.TFExtractStateComponents(tf_curr_z_vec)
        
        tf.assert_equal(tf.shape(tf_G_vec)[0], self.dmp_num_dimensions)
        tf.assert_equal(tf.shape(tf_G_vec)[1], 1)
        
        tf.assert_equal(tf.shape(tf_ft_kernel_mu_vec)[0], self.model_size)
        tf.assert_equal(tf.shape(tf_ft_kernel_mu_vec)[1], 1)
        
        tf.assert_equal(tf.shape(tf_ft_kernel_chi_vec)[0], self.model_size)
        tf.assert_equal(tf.shape(tf_ft_kernel_chi_vec)[1], 1)
        
        tf.assert_equal(tf.shape(tf_ft_W_mat)[0], self.dmp_num_dimensions)
        tf.assert_equal(tf.shape(tf_ft_W_mat)[1], self.model_size)
        
        tf.assert_equal(tf.shape(tf_ft_scaler_vec)[0], self.dmp_num_dimensions)
        tf.assert_equal(tf.shape(tf_ft_scaler_vec)[1], 1)
        
        tf_fdync_curr_x_vec = tf_curr_v_vec
        
        tf_ft_kernel_vec = tf.exp(-0.5 * 
                                  tf.square(tf.matmul(tf.ones([self.model_size, 1], tf.float32), tf_curr_p_vec) - 
                                            tf_ft_kernel_mu_vec) * 
                                  tf_ft_kernel_chi_vec)
        tf_normalized_ft_kernel_matmul_u_vec = tf.matmul((tf_ft_kernel_vec / 
                                                          tf.reduce_sum(tf_ft_kernel_vec + 1.e-10, axis=0)[0]), 
                                                         tf_curr_u_vec)
        tf_ft_vec = tf.matmul(tf_ft_W_mat, tf_normalized_ft_kernel_matmul_u_vec)
        tf_fdync_curr_v_vec = (((tf_alpha_v_sca / (tf_tau_relative_sca * tf_tau_relative_sca)) * 
                                ((tf_beta_v_sca * (tf_curr_g_vec - tf_curr_x_vec))
                                - (tf_tau_relative_sca * tf_curr_v_vec)))
                               + ((1.0 / (tf_tau_relative_sca * tf_tau_relative_sca)) * 
                                  tf_ft_scaler_vec * tf_ft_vec))
        
        tf_fdync_curr_g_vec = ((tf_alpha_g_sca / tf_tau_relative_sca) * (tf_G_vec - tf_curr_g_vec))
        
        tf_fdync_curr_p_vec = (1.0 / tf_tau_relative_sca) * tf_curr_u_vec
        
        tf_fdync_curr_u_vec = ((tf_alpha_u_sca / tf_tau_relative_sca) * 
                               ((tf_beta_u_sca * (- tf_curr_p_vec)) - tf_curr_u_vec))
        
        tf_fdync_curr_z_vec = tf.concat([tf_fdync_curr_x_vec, 
                                         tf_fdync_curr_v_vec, 
                                         tf_fdync_curr_g_vec, 
                                         tf_fdync_curr_p_vec, 
                                         tf_fdync_curr_u_vec
                                         ], axis=0)
        
        tf_u_curr_x_vec = tf.zeros([self.dmp_num_dimensions, 1], tf.float32)
        tf_u_curr_v_vec = ((1.0 / (tf_tau_relative_sca * tf_tau_relative_sca)) * tf_ct_vec)
        tf_u_curr_g_vec = tf.zeros([self.dmp_num_dimensions, 1], tf.float32)
        tf_u_curr_p_vec = tf.zeros([1, 1], tf.float32)
        tf_u_curr_u_vec = tf.zeros([1, 1], tf.float32)
        
        tf_u_curr_z_vec = tf.concat([tf_u_curr_x_vec, 
                                     tf_u_curr_v_vec, 
                                     tf_u_curr_g_vec, 
                                     tf_u_curr_p_vec, 
                                     tf_u_curr_u_vec
                                     ], axis=0)
        
        tf_fdynd_curr_z_vec = (tf_curr_z_vec + (tf_fdync_curr_z_vec * tf_dt_sca))
        tf_ud_curr_z_vec = (tf_u_curr_z_vec * tf_dt_sca)
        
        tf_next_z_vec = tf_fdynd_curr_z_vec + tf_ud_curr_z_vec
        
        tf.assert_equal(tf.shape(tf_next_z_vec)[0], ((3 * self.dmp_num_dimensions) + 1 + 1))
        tf.assert_equal(tf.shape(tf_next_z_vec)[1], 1)
        
        tf.assert_equal(tf.shape(tf_fdynd_curr_z_vec)[0], ((3 * self.dmp_num_dimensions) + 1 + 1))
        tf.assert_equal(tf.shape(tf_fdynd_curr_z_vec)[1], 1)
        
        tf.assert_equal(tf.shape(tf_ud_curr_z_vec)[0], ((3 * self.dmp_num_dimensions) + 1 + 1))
        tf.assert_equal(tf.shape(tf_ud_curr_z_vec)[1], 1)
        
        return [tf_next_z_vec, tf_fdynd_curr_z_vec, tf_ud_curr_z_vec]
    
    def GetTFGraphComputeNextAugmentedState(self, is_returning_as_graph_def=True):
        g = tf.Graph()
        with g.as_default() as g:
            tf_curr_z_vec_ph = tf.placeholder(tf.float32, shape=[((3*self.dmp_num_dimensions)+1+1), 1], name="tf_curr_z_vec_placeholder")
            tf_G_vec_ph = tf.placeholder(tf.float32, shape=[self.dmp_num_dimensions, 1], name="tf_G_vec_placeholder")
            tf_ft_kernel_mu_vec_ph = tf.placeholder(tf.float32, shape=[self.model_size, 1], name="tf_ft_kernel_mu_vec_placeholder")
            tf_ft_kernel_chi_vec_ph = tf.placeholder(tf.float32, shape=[self.model_size, 1], name="tf_ft_kernel_chi_vec_placeholder")
            tf_ft_W_mat_ph = tf.placeholder(tf.float32, shape=[self.dmp_num_dimensions, self.model_size], name="tf_ft_W_mat_placeholder")
            tf_ft_scaler_vec_ph = tf.placeholder(tf.float32, shape=[self.dmp_num_dimensions, 1], name="tf_ft_scaler_vec_placeholder")
            tf_ct_vec_ph = tf.placeholder(tf.float32, shape=[self.dmp_num_dimensions, 1], name="tf_ct_vec_placeholder")
            tf_tau_relative_sca_ph = tf.placeholder(tf.float32, shape=[], name="tf_tau_relative_sca_placeholder")
            tf_dt_sca_ph = tf.placeholder(tf.float32, shape=[], name="tf_dt_sca_placeholder")
            tf_alpha_v_sca_ph = tf.placeholder_with_default(25.0, shape=[], name="tf_alpha_v_sca_placeholder")
            tf_beta_v_sca_ph = tf.placeholder_with_default(25.0/4.0, shape=[], name="tf_beta_v_sca_placeholder")
            tf_alpha_g_sca_ph = tf.placeholder_with_default(25.0/2.0, shape=[], name="tf_alpha_g_sca_placeholder")
            tf_alpha_u_sca_ph = tf.placeholder_with_default(25.0, shape=[], name="tf_alpha_u_sca_placeholder")
            tf_beta_u_sca_ph = tf.placeholder_with_default(25.0/4.0, shape=[], name="tf_beta_u_sca_placeholder")
            
            [tf_next_z_vec,
             tf_fdynd_curr_z_vec,
             tf_ud_curr_z_vec
             ] = self.TFComputeNextAugmentedState(tf_curr_z_vec_ph, 
                                                  tf_G_vec_ph, 
                                                  tf_ft_kernel_mu_vec_ph, 
                                                  tf_ft_kernel_chi_vec_ph, 
                                                  tf_ft_W_mat_ph, 
                                                  tf_ft_scaler_vec_ph, 
                                                  tf_ct_vec_ph, 
                                                  tf_tau_relative_sca_ph, 
                                                  tf_dt_sca_ph, 
                                                  tf_alpha_v_sca_ph, 
                                                  tf_beta_v_sca_ph, 
                                                  tf_alpha_g_sca_ph, 
                                                  tf_alpha_u_sca_ph, 
                                                  tf_beta_u_sca_ph)
            
            tf_next_z_vec_ph = tf.identity(tf_next_z_vec, name="tf_next_z_vec_placeholder") # output placeholder
            tf_fdynd_curr_z_vec_ph = tf.identity(tf_fdynd_curr_z_vec, name="tf_fdynd_curr_z_vec_placeholder") # output placeholder
            tf_ud_curr_z_vec_ph = tf.identity(tf_ud_curr_z_vec, name="tf_ud_curr_z_vec_placeholder") # output placeholder
        if (is_returning_as_graph_def):
            return g.as_graph_def()
        else:
            return g
    
    def getNextState(self, dt, update_canonical_state, is_also_returning_local_next_state=False):
        assert (self.is_started == True)
        assert (self.isValid()), "Pre-condition(s) checking is failed: this CartesianCoordDMP is invalid!"
        
        assert (self.transform_sys_discrete_cart_coord.is_started)
        assert (self.transform_sys_discrete_cart_coord.isValid()), "Pre-condition(s) checking is failed: the TransformSystemDiscrete of this CartesianCoordDMP is invalid!"
        assert (dt > 0.0)
        
        x0 = self.transform_sys_discrete_cart_coord.start_state.X
        
        x = self.transform_sys_discrete_cart_coord.current_state.X
        v = self.transform_sys_discrete_cart_coord.current_state.Xd
        g = self.transform_sys_discrete_cart_coord.goal_sys.current_goal_state.X
        p = np.ones((1,1))
        u = np.ones((1,1))
        
        curr_z = np.vstack([x, v, g, p, u])
        G = self.transform_sys_discrete_cart_coord.goal_sys.G
        ft_kernel_mu = self.func_approx.centers
        ft_kernel_chi = self.func_approx.bandwidths
        ft_W = self.func_approx.weights
        ft_scaler = G - x0
        for d in range(self.dmp_num_dimensions):
            if (self.transform_sys_discrete_cart_coord.is_using_scaling[d]):
                if (np.fabs(self.transform_sys_discrete_cart_coord.A_learn[d,0]) < MIN_FABS_AMPLITUDE):
                    ft_scaler[d,0] = 1.0
                else:
                    ft_scaler[d,0] = ft_scaler[d,0] * 1.0 / self.transform_sys_discrete_cart_coord.A_learn[d,0]
            else:
                ft_scaler[d,0] = 1.0
        
        
        # TO BE CONTINUED ...
        # Please see example in file "test_tf_import_graph_def.py" in email "TensorFlow Tests" in account gsutanto@usc.edu .
        # This example shows how to chain/combine several graphs.
        # In our case, there can be two possibilities:
        # [1] If NOT using coupling term, then we will ONLY use the graph definition returned by 
        #     the function GetTFGraphComputeNextAugmentedState() here.
        # [2] If using coupling term, then we will chain the graph definition returned by
        #     the function GetTFGraphComputeNextAugmentedState() and 
        #     another graph representing the coupling term computation, into one graph
        #     (e.g. so that we can compute the gradient of the combined graph).
        # Please continue working on the commented code below 
        # (which was imported (and to be adapted) from getNextState() function of CartesianCoordDMP and TransformSystemDiscrete).
        
        
#        ct = 
#        tau_relative = self.tau_sys.getTauRelative()
#        
#        forcing_term, basis_function_vector = self.func_approx.getForcingTerm()
#        ct_acc, ct_vel = self.getCouplingTerm()
#        for d in range(self.dmp_num_dimensions):
#            if (self.is_using_coupling_term_at_dimension[d] == False):
#                ct_acc[d,0] = 0.0
#                ct_vel[d,0] = 0.0
#        
#        time = self.current_state.time
#        
#        
#        xdd = self.current_state.Xdd
#        
#        v = self.current_velocity_state.X
#        vd = self.current_velocity_state.Xd
#        
#        x = x + (xd * dt)
#        
#        
#        vd = ((self.alpha * ((self.beta * (g - x)) - v)) + (forcing_term * A) + ct_acc) * 1.0 / tau
#        assert (np.isnan(vd).any() == False), "vd contains NaN!"
#        
#        xdd = vd * 1.0 / tau
#        xd = (v + ct_vel) * 1.0 / tau
#        
#        v = v + (vd * dt)
#        
#        time = time + dt
#        
#        # self.current_state = DMPState(x, xd, xdd, time)
#        # self.current_velocity_state = DMPState(v, vd, np.zeros((self.dmp_num_dimensions,1)), time)
#        self.current_state.X = x
#        self.current_state.Xd = xd
#        self.current_state.Xdd = xdd
#        self.current_state.time = time
#        self.current_velocity_state.X = v
#        self.current_velocity_state.Xd = vd
#        self.current_velocity_state.time = time
#        next_state = self.current_state
#        
#        return next_state, forcing_term, ct_acc, ct_vel, basis_function_vector
#        
##        [result_dmp_state_local, 
##         transform_sys_forcing_term, 
##         transform_sys_coupling_term_acc, 
##         transform_sys_coupling_term_vel, 
##         func_approx_basis_function_vector] = self.transform_sys_discrete_cart_coord.getNextState(dt)
##        
#        if (update_canonical_state):
#            assert (self.is_started), "CanonicalSystemDiscrete is NOT yet started!"
#            assert (self.isValid()), "Pre-condition(s) checking is failed: this CanonicalSystemDiscrete is invalid!"
#            assert (dt > 0.0), "dt=" + str(dt) + " <= 0.0 (invalid!)"
#            
#            tau = self.tau_sys.getTauRelative()
#            C_c = self.getCouplingTerm()
#            if (self.order == 2):
#                self.xdd = self.vd * 1.0 / tau
#                self.vd = ((self.alpha * ((self.beta * (0 - self.x)) - self.v)) + C_c) * 1.0 / tau
#                self.xd = self.v * 1.0 / tau
#            elif (self.order == 1):
#                self.xdd = self.vd * 1.0 / tau
#                self.vd = 0.0
#                self.xd = ((self.alpha * (0 - self.x)) + C_c) * 1.0 / tau
#            self.x = self.x + (self.xd * dt)
#            self.v = self.v + (self.vd * dt)
#            
#            assert (self.x >= 0.0), "self.x=" + str(self.x) + " < 0.0 (invalid!)"
#            return None
##            self.canonical_sys_discrete.updateCanonicalState(dt)
#        
#        assert (self.is_started), "GoalSystem is NOT yet started!"
#        assert (self.isValid()), "Pre-condition(s) checking is failed: this GoalSystem is invalid!"
#        assert (dt > 0.0), "dt=" + str(dt) + " <= 0.0 (invalid!)"
#        
#        tau = self.tau_sys.getTauRelative()
#        g = self.current_goal_state.getX()
#        gd = (self.alpha * 1.0 / tau) * (self.G - g)
#        g = g + (gd * dt)
#        self.current_goal_state.setX(g)
#        self.current_goal_state.setXd(gd)
#        self.current_goal_state.setTime(self.current_goal_state.getTime() + dt)
#        
#        assert (self.isValid()), "Post-condition(s) checking is failed: this GoalSystem became invalid!"
#        return None
##        self.transform_sys_discrete_cart_coord.updateCurrentGoalState(dt)
#        
#        # update critical states in local coordinate frame (only current state and current goal state changes)
#        self.ctraj_critical_states_local_coord.setDMPStateAtIndex(1, result_dmp_state_local)
#        self.ctraj_critical_states_local_coord.setDMPStateAtIndex(2, self.transform_sys_discrete_cart_coord.getCurrentGoalState())
#        
#        self.ctraj_critical_states_global_coord = self.cart_coord_transformer.convertCTrajAtOldToNewCoordSys(self.ctraj_critical_states_local_coord,
#                                                                                                             self.ctraj_hmg_transform_local_to_global_matrix)
#        
#        result_dmp_state_global = self.ctraj_critical_states_global_coord.getDMPStateAtIndex(1)
#        
#        assert (self.isValid()), "Post-condition(s) checking is failed: this CartesianCoordDMP became invalid!"
#        if (is_also_returning_local_next_state):
#            return result_dmp_state_global, result_dmp_state_local, transform_sys_forcing_term, transform_sys_coupling_term_acc, transform_sys_coupling_term_vel, func_approx_basis_function_vector
#        else:
#            return result_dmp_state_global, transform_sys_forcing_term, transform_sys_coupling_term_acc, transform_sys_coupling_term_vel, func_approx_basis_function_vector
        
        return None