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
    x = state_position_vector             (size 3X1)
    v = state_velocity_vector             (size 3X1)
    g = goal_state_position_vector        (size 3X1)
    p = phase_variable_vector             (size 1X1)
    u = phase_velocity_vector             (size 1X1)
    G = steady_state_goal_position_vector (size 3X1)
    """
    
    def TFExtractStateComponents(self, 
                                 tf_augmented_state_vector):
        tf.assert_equal(tf.shape(tf_augmented_state_vector)[0], ((3 * self.dmp_num_dimensions) + 1 + 1))
        tf.assert_equal(tf.shape(tf_augmented_state_vector)[1], 1)
        
        tf_state_position_vector = tf.slice(tf_augmented_state_vector, [0 * self.dmp_num_dimensions, 0], [self.dmp_num_dimensions, 1])
        tf_state_velocity_vector = tf.slice(tf_augmented_state_vector, [1 * self.dmp_num_dimensions, 0], [self.dmp_num_dimensions, 1])
        tf_goal_state_position_vector = tf.slice(tf_augmented_state_vector, [2 * self.dmp_num_dimensions, 0], [self.dmp_num_dimensions, 1])
        tf_phase_variable_vector = tf.slice(tf_augmented_state_vector, [(3 * self.dmp_num_dimensions) + 0, 0], [1, 1])
        tf_phase_velocity_vector = tf.slice(tf_augmented_state_vector, [(3 * self.dmp_num_dimensions) + 1, 0], [1, 1])
        
        return [tf_state_position_vector, 
                tf_state_velocity_vector, 
                tf_goal_state_position_vector, 
                tf_phase_variable_vector, 
                tf_phase_velocity_vector]
        
    def TFAugmentStateComponents(self, 
                                 tf_state_position_vector, 
                                 tf_state_velocity_vector, 
                                 tf_goal_state_position_vector, 
                                 tf_phase_variable_vector, 
                                 tf_phase_velocity_vector):
        tf.assert_equal(tf.shape(tf_state_position_vector)[0], self.dmp_num_dimensions)
        tf.assert_equal(tf.shape(tf_state_position_vector)[1], 1)
        
        tf.assert_equal(tf.shape(tf_state_velocity_vector)[0], self.dmp_num_dimensions)
        tf.assert_equal(tf.shape(tf_state_velocity_vector)[1], 1)
        
        tf.assert_equal(tf.shape(tf_goal_state_position_vector)[0], self.dmp_num_dimensions)
        tf.assert_equal(tf.shape(tf_goal_state_position_vector)[1], 1)
        
        tf.assert_equal(tf.shape(tf_phase_variable_vector)[0], 1)
        tf.assert_equal(tf.shape(tf_phase_variable_vector)[1], 1)
        
        tf.assert_equal(tf.shape(tf_phase_velocity_vector)[0], 1)
        tf.assert_equal(tf.shape(tf_phase_velocity_vector)[1], 1)
        
        tf_augmented_state_vector = tf.concat([tf_state_position_vector, 
                                               tf_state_velocity_vector, 
                                               tf_goal_state_position_vector, 
                                               tf_phase_variable_vector, 
                                               tf_phase_velocity_vector
                                               ], axis=0)
        
        tf.assert_equal(tf.shape(tf_augmented_state_vector)[0], ((3 * self.dmp_num_dimensions) + 1 + 1))
        tf.assert_equal(tf.shape(tf_augmented_state_vector)[1], 1)
        
        return tf_augmented_state_vector
    
    def TFComputeNextAugmentedState(self,
                                    tf_curr_augmented_state_vector, 
                                    tf_steady_state_goal_position_vector, 
                                    tf_forcing_term_kernel_center_vector, 
                                    tf_forcing_term_kernel_bandwidth_vector, 
                                    tf_forcing_term_weight_matrix, 
                                    tf_forcing_term_scaler_vector, 
                                    tf_coupling_term_vector, 
                                    tf_tau_base_scalar, 
                                    tf_dt_scalar, 
                                    tf_alpha_v_scalar=25.0, 
                                    tf_beta_v_scalar=25.0/4.0, 
                                    tf_alpha_g_scalar=25.0/2.0, 
                                    tf_alpha_u_scalar=25.0, 
                                    tf_beta_u_scalar=25.0/4.0):
        [tf_curr_state_position_vector, 
         tf_curr_state_velocity_vector, 
         tf_curr_goal_state_position_vector, 
         tf_curr_phase_variable_vector, 
         tf_curr_phase_velocity_vector
         ] = self.TFExtractStateComponents(tf_curr_augmented_state_vector)
        
        tf.assert_equal(tf.shape(tf_steady_state_goal_position_vector)[0], self.dmp_num_dimensions)
        tf.assert_equal(tf.shape(tf_steady_state_goal_position_vector)[1], 1)
        
        tf.assert_equal(tf.shape(tf_forcing_term_kernel_center_vector)[0], self.model_size)
        tf.assert_equal(tf.shape(tf_forcing_term_kernel_center_vector)[1], 1)
        
        tf.assert_equal(tf.shape(tf_forcing_term_kernel_bandwidth_vector)[0], self.model_size)
        tf.assert_equal(tf.shape(tf_forcing_term_kernel_bandwidth_vector)[1], 1)
        
        tf.assert_equal(tf.shape(tf_forcing_term_weight_matrix)[0], self.dmp_num_dimensions)
        tf.assert_equal(tf.shape(tf_forcing_term_weight_matrix)[1], self.model_size)
        
        tf.assert_equal(tf.shape(tf_forcing_term_scaler_vector)[0], self.dmp_num_dimensions)
        tf.assert_equal(tf.shape(tf_forcing_term_scaler_vector)[1], 1)
        
        tf_fdync_curr_state_position_vector = tf_curr_state_velocity_vector
        
        tau_reference = 0.5
        tf_tau_relative_scalar = tf_tau_base_scalar * 1.0 / tau_reference
        tf_forcing_term_kernel_vector = tf.exp(-0.5 * 
                                               tf.square(tf.matmul(tf.ones([self.model_size, 1], tf.float32), tf_curr_phase_variable_vector) - 
                                                         tf_forcing_term_kernel_center_vector) * 
                                               tf_forcing_term_kernel_bandwidth_vector)
        tf_normalized_forcing_term_kernel_matmul_phase_velocity_vector = tf.matmul((tf_forcing_term_kernel_vector / 
                                                                                    tf.reduce_sum(tf_forcing_term_kernel_vector + 1.e-10, axis=0)[0]), 
                                                                                   tf_curr_phase_velocity_vector)
        tf_forcing_term_vector = tf.matmul(tf_forcing_term_weight_matrix, tf_normalized_forcing_term_kernel_matmul_phase_velocity_vector)
        tf_fdync_curr_state_velocity_vector = (((tf_alpha_v_scalar / (tf_tau_relative_scalar * tf_tau_relative_scalar)) * 
                                                ((tf_beta_v_scalar * (tf_curr_goal_state_position_vector - tf_curr_state_position_vector))
                                                 - (tf_tau_relative_scalar * tf_curr_state_velocity_vector)))
                                               + ((1.0 / (tf_tau_relative_scalar * tf_tau_relative_scalar)) * 
                                                  tf_forcing_term_scaler_vector * tf_forcing_term_vector))
        
        tf_fdync_curr_goal_state_position_vector = ((tf_alpha_g_scalar / tf_tau_relative_scalar) * 
                                                    (tf_steady_state_goal_position_vector - tf_curr_goal_state_position_vector))
        
        tf_fdync_curr_phase_variable_vector = (1.0 / tf_tau_relative_scalar) * tf_curr_phase_velocity_vector
        
        tf_fdync_curr_phase_velocity_vector = ((tf_alpha_u_scalar / tf_tau_relative_scalar) * 
                                               ((tf_beta_u_scalar * (- tf_curr_phase_variable_vector)) - tf_curr_phase_velocity_vector))
        
        tf_fdync_curr_augmented_state_vector = tf.concat([tf_fdync_curr_state_position_vector, 
                                                          tf_fdync_curr_state_velocity_vector, 
                                                          tf_fdync_curr_goal_state_position_vector, 
                                                          tf_fdync_curr_phase_variable_vector, 
                                                          tf_fdync_curr_phase_velocity_vector
                                                          ], axis=0)
        
        tf_u_curr_state_position_vector = tf.zeros([self.dmp_num_dimensions, 1], tf.float32)
        tf_u_curr_state_velocity_vector = ((1.0 / (tf_tau_relative_scalar * tf_tau_relative_scalar)) * tf_coupling_term_vector)
        tf_u_curr_goal_state_position_vector = tf.zeros([self.dmp_num_dimensions, 1], tf.float32)
        tf_u_curr_phase_variable_vector = tf.zeros([1, 1], tf.float32)
        tf_u_curr_phase_velocity_vector = tf.zeros([1, 1], tf.float32)
        
        tf_u_curr_augmented_state_vector = tf.concat([tf_u_curr_state_position_vector, 
                                                      tf_u_curr_state_velocity_vector, 
                                                      tf_u_curr_goal_state_position_vector, 
                                                      tf_u_curr_phase_variable_vector, 
                                                      tf_u_curr_phase_velocity_vector
                                                      ], axis=0)
        
        tf_fdynd_curr_augmented_state_vector = (tf_curr_augmented_state_vector + (tf_fdync_curr_augmented_state_vector * tf_dt_scalar))
        tf_ud_curr_augmented_state_vector = (tf_u_curr_augmented_state_vector * tf_dt_scalar)
        
        tf_next_augmented_state_vector = tf_fdynd_curr_augmented_state_vector + tf_ud_curr_augmented_state_vector
        
        tf.assert_equal(tf.shape(tf_next_augmented_state_vector)[0], ((3 * self.dmp_num_dimensions) + 1 + 1))
        tf.assert_equal(tf.shape(tf_next_augmented_state_vector)[1], 1)
        
        tf.assert_equal(tf.shape(tf_fdynd_curr_augmented_state_vector)[0], ((3 * self.dmp_num_dimensions) + 1 + 1))
        tf.assert_equal(tf.shape(tf_fdynd_curr_augmented_state_vector)[1], 1)
        
        tf.assert_equal(tf.shape(tf_ud_curr_augmented_state_vector)[0], ((3 * self.dmp_num_dimensions) + 1 + 1))
        tf.assert_equal(tf.shape(tf_ud_curr_augmented_state_vector)[1], 1)
        
        return [tf_next_augmented_state_vector, tf_fdynd_curr_augmented_state_vector, tf_ud_curr_augmented_state_vector]