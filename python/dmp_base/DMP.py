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
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities/'))
from TauSystem import *
from CanonicalSystem import *
from FunctionApproximator import *
from TransformationSystem import *
from DMPState import *
from DMPTrajectory import *
from DataIO import *
from utility_states_trajectories import smoothStartEndNDTrajectoryBasedOnPosition

class DMP:
    'Base class for DMPs.'
    
    def __init__(self, canonical_system, function_approximator, transform_system, learning_system, name=""):
        self.name = name
        self.learning_sys = learning_system
        self.transform_sys = transform_system
        self.func_approx = function_approximator
        self.canonical_sys = canonical_system
        self.tau_sys = self.canonical_sys.tau_sys
        self.dmp_num_dimensions = self.func_approx.dmp_num_dimensions
        self.model_size = self.func_approx.model_size
        self.is_started = False
        self.mean_start_position = np.zeros((self.dmp_num_dimensions,1))
        self.mean_goal_position = np.zeros((self.dmp_num_dimensions,1))
        self.mean_tau = 0.0
    
    def isValid(self):
        assert (self.dmp_num_dimensions > 0), "self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + " <= 0 (invalid!)"
        assert (self.model_size > 0), "self.model_size=" + str(self.model_size) + "<= 0 (invalid!)"
        assert (self.tau_sys != None)
        assert (self.canonical_sys != None)
        assert (self.func_approx != None)
        assert (self.transform_sys != None)
        assert (self.tau_sys.isValid())
        assert (self.canonical_sys.isValid())
        assert (self.func_approx.isValid())
        assert (self.transform_sys.isValid())
        assert (self.func_approx.dmp_num_dimensions == self.dmp_num_dimensions), "self.func_approx.dmp_num_dimensions=" + str(self.func_approx.dmp_num_dimensions) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
        assert (self.transform_sys.dmp_num_dimensions == self.dmp_num_dimensions), "self.transform_sys.dmp_num_dimensions=" + str(self.transform_sys.dmp_num_dimensions) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
        assert (self.func_approx.model_size == self.model_size), "self.func_approx.model_size=" + str(self.func_approx.model_size) + " is mis-matched with self.model_size=" + str(self.model_size) + "!"
        assert (self.tau_sys == self.canonical_sys.tau_sys)
        assert (self.tau_sys == self.transform_sys.tau_sys)
        assert (self.canonical_sys == self.func_approx.canonical_sys)
        assert (self.canonical_sys == self.transform_sys.canonical_sys)
        assert (self.func_approx == self.transform_sys.func_approx)
        assert (self.mean_start_position.shape[0] == self.transform_sys.getCurrentGoalState().getX().shape[0]), "self.mean_start_position.shape[0]=" + str(self.mean_start_position.shape[0]) + " is mis-matched with self.transform_sys.getCurrentGoalState().getX().shape[0]=" + str(self.transform_sys.getCurrentGoalState().getX().shape[0]) + "!"
        assert (self.mean_goal_position.shape[0] == self.transform_sys.getCurrentGoalState().getX().shape[0]), "self.mean_goal_position.shape[0]=" + str(self.mean_goal_position.shape[0]) + " is mis-matched with self.transform_sys.getCurrentGoalState().getX().shape[0]=" + str(self.transform_sys.getCurrentGoalState().getX().shape[0]) + "!"
        assert (self.mean_tau >= 0.0), "self.mean_tau=" + str(self.mean_tau) + " < 0.0 (invalid!)"
        return True
    
    def preprocess(self, list_dmp_trajectory):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this DMP is invalid!"
        N_traj = len(list_dmp_trajectory)
        
        self.mean_start_position = np.zeros((self.dmp_num_dimensions,1))
        self.mean_goal_position = np.zeros((self.dmp_num_dimensions,1))
        for dmp_trajectory in list_dmp_trajectory:
            self.mean_start_position = self.mean_start_position + dmp_trajectory.X[:,[0]]
            self.mean_goal_position = self.mean_goal_position + dmp_trajectory.X[:,[-1]]
        self.mean_start_position = self.mean_start_position * 1.0 / N_traj
        self.mean_goal_position = self.mean_goal_position * 1.0 / N_traj
        preprocessed_list_dmp_trajectory = list_dmp_trajectory
        assert (self.isValid()), "Post-condition(s) checking is failed: this DMP became invalid!"
        return preprocessed_list_dmp_trajectory
    
    def smoothStartEndTrajectoryBasedOnPosition(self, traj, percentage_padding, percentage_smoothing_points, mode, dt, smoothing_cutoff_frequency):
        if (dt is None):
            traj_length = traj.time.shape[1]
            assert (traj_length > 1)
            dt = (traj.time[0, traj_length-1] - traj.time[0, 0])/(1.0 * (traj_length-1))
            assert (dt > 0.0)
        return smoothStartEndNDTrajectoryBasedOnPosition(ND_traj=traj, 
                                                         percentage_padding=percentage_padding, 
                                                         percentage_smoothing_points=percentage_smoothing_points, 
                                                         mode=mode, dt=dt, 
                                                         fc=smoothing_cutoff_frequency)
    
    def learnFromPath(self, training_data_dir_or_file_path, robot_task_servo_rate, start_column_idx=1, time_column_idx=0, 
                      is_smoothing_training_traj_before_learning=False, 
                      percentage_padding=None, percentage_smoothing_points=None, smoothing_mode=None, dt=None, smoothing_cutoff_frequency=None):
        set_traj_input = self.extractSetTrajectories(training_data_dir_or_file_path, start_column_idx, time_column_idx)
        if (is_smoothing_training_traj_before_learning):
            processed_set_traj_input = list()
            for traj_input in set_traj_input:
                processed_set_traj_input.append(self.smoothStartEndTrajectoryBasedOnPosition(traj=traj_input, 
                                                                                             percentage_padding=percentage_padding, 
                                                                                             percentage_smoothing_points=percentage_smoothing_points, 
                                                                                             mode=smoothing_mode, 
                                                                                             dt=dt, 
                                                                                             smoothing_cutoff_frequency=smoothing_cutoff_frequency))
        else:
            processed_set_traj_input = set_traj_input
        return self.learnGetDefaultUnrollParams(processed_set_traj_input, robot_task_servo_rate)
    
    def learnGetDefaultUnrollParams(self, set_traj_input, robot_task_servo_rate):
        W, mean_A_learn, mean_tau, Ft, Fp, G, cX, cV, PSI = self.learn(set_traj_input, robot_task_servo_rate)
        critical_states_list_learn = [None] * 2
        critical_states_list_learn[0] = DMPState(self.mean_start_position)
        critical_states_list_learn[-1] = DMPState(self.mean_goal_position)
        critical_states_learn = convertDMPStatesListIntoDMPTrajectory(critical_states_list_learn)
        return critical_states_learn, W, mean_A_learn, self.mean_tau, Ft, Fp, G, cX, cV, PSI
    
    def startWithUnrollParams(self, dmp_unroll_init_parameters):
        assert (dmp_unroll_init_parameters.isValid())
        return (self.start(dmp_unroll_init_parameters.critical_states, dmp_unroll_init_parameters.tau))
    
    def getMeanStartPosition(self):
        return copy.copy(self.mean_start_position)
    
    def getMeanGoalPosition(self):
        return copy.copy(self.mean_goal_position)
    
    def getMeanTau(self):
        return copy.copy(self.mean_tau)
    
    def setTransformSystemCouplingTermUsagePerDimensions(self, is_using_transform_sys_coupling_term_at_dimension_init):
        return self.transform_sys.setCouplingTermUsagePerDimensions(is_using_transform_sys_coupling_term_at_dimension_init)