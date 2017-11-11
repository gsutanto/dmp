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
    
    def learnFromPath(self, training_data_dir_or_file_path, robot_task_servo_rate):
        set_traj_input = self.extractSetTrajectories(training_data_dir_or_file_path)
        self.learn(set_traj_input, robot_task_servo_rate)
        tau_learn = self.mean_tau
        critical_states_list_learn = []
        critical_states_list_learn.append(DMPState(self.mean_start_position))
        critical_states_list_learn.append(DMPState(self.mean_goal_position))
        critical_states_learn = convertDMPStatesListIntoDMPTrajectory(critical_states_list_learn)
        return tau_learn, critical_states_learn
    
    def start(self, dmp_unroll_init_parameters):
        assert (dmp_unroll_init_parameters.isValid())
        return (self.start(dmp_unroll_init_parameters.critical_states, dmp_unroll_init_parameters.tau))
    
    def getTargetCouplingTermAndUpdateStates(self, current_state_demo_local, dt):
        assert (self.is_started)
        ct_acc_target = self.transform_sys.getTargetCouplingTerm(current_state_demo_local)
        self.canonical_sys.updateCanonicalState(dt)
        self.transform_sys.updateCurrentGoalState(dt)
        return ct_acc_target
    
    def setTransformSystemCouplingTermUsagePerDimensions(self, is_using_transform_sys_coupling_term_at_dimension_init):
        return self.transform_sys.setCouplingTermUsagePerDimensions(is_using_transform_sys_coupling_term_at_dimension_init)