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
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities/'))
from TauSystem import *
from CanonicalSystem import *
from FunctionApproximator import *
from TransformationSystem import *
from LearningSystem import *
from DataIO import *

class DMP:
    'Base class for DMPs.'
    
    def __init__(self, learning_system, name=""):
        self.name = name
        self.learning_sys = learning_system
        self.transform_sys = self.learning_sys.transform_sys
        self.func_approx = self.transform_sys.func_approx
        self.canonical_sys = self.transform_sys.canonical_sys
        self.tau_sys = self.canonical_sys.tau_sys
        self.dmp_num_dimensions = self.learning_sys.dmp_num_dimensions
        self.model_size = self.learning_sys.model_size
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
        assert (self.learning_sys != None)
        assert (self.tau_sys.isValid() == True)
        assert (self.canonical_sys.isValid() == True)
        assert (self.func_approx.isValid() == True)
        assert (self.transform_sys.isValid() == True)
        assert (self.learning_sys.isValid() == True)
        assert (self.func_approx.dmp_num_dimensions == self.dmp_num_dimensions), "self.func_approx.dmp_num_dimensions=" + str(self.func_approx.dmp_num_dimensions) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
        assert (self.transform_sys.dmp_num_dimensions == self.dmp_num_dimensions), "self.transform_sys.dmp_num_dimensions=" + str(self.transform_sys.dmp_num_dimensions) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
        assert (self.learning_sys.dmp_num_dimensions == self.dmp_num_dimensions), "self.learning_sys.dmp_num_dimensions=" + str(self.learning_sys.dmp_num_dimensions) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
        assert (self.func_approx.model_size == self.model_size), "self.func_approx.model_size=" + str(self.func_approx.model_size) + " is mis-matched with self.model_size=" + str(self.model_size) + "!"
        assert (self.learning_sys.model_size == self.model_size), "self.learning_sys.model_size=" + str(self.learning_sys.model_size) + " is mis-matched with self.model_size=" + str(self.model_size) + "!"
        assert (self.tau_sys == self.canonical_sys.tau_sys)
        assert (self.tau_sys == self.transform_sys.tau_sys)
        assert (self.canonical_sys == self.func_approx.canonical_sys)
        assert (self.canonical_sys == self.transform_sys.canonical_sys)
        assert (self.func_approx == self.transform_sys.func_approx)
        assert (self.transform_sys == self.learning_sys.transform_sys)
        assert (self.mean_start_position.shape[0] == self.transform_sys.getCurrentGoalState().getX().shape[0]), "self.mean_start_position.shape[0]=" + str(self.mean_start_position.shape[0]) + " is mis-matched with self.transform_sys.getCurrentGoalState().getX().shape[0]=" + str(self.transform_sys.getCurrentGoalState().getX().shape[0]) + "!"
        assert (self.mean_goal_position.shape[0] == self.transform_sys.getCurrentGoalState().getX().shape[0]), "self.mean_goal_position.shape[0]=" + str(self.mean_goal_position.shape[0]) + " is mis-matched with self.transform_sys.getCurrentGoalState().getX().shape[0]=" + str(self.transform_sys.getCurrentGoalState().getX().shape[0]) + "!"
        assert (self.mean_tau >= 0.0), "self.mean_tau=" + str(self.mean_tau) + " < 0.0 (invalid!)"
        return True
    
    def preprocess(self, dmp_trajectories_list):
        assert (self.isValid() == True), "Pre-condition(s) checking is failed: this DMP is invalid!"
        N_traj = len(dmp_trajectories_list)
        
        self.mean_start_position = np.zeros((self.dmp_num_dimensions,1))
        self.mean_goal_position = np.zeros((self.dmp_num_dimensions,1))
        for dmp_trajectory in dmp_trajectories_list:
            self.mean_start_position = self.mean_start_position + dmp_trajectory.X[:,[0]]
            self.mean_goal_position = self.mean_goal_position + dmp_trajectory.X[:,[-1]]
        self.mean_start_position = self.mean_start_position/N_traj
        self.mean_goal_position = self.mean_goal_position/N_traj
        assert (self.isValid() == True), "Post-condition(s) checking is failed: this DMP became invalid!"
        return None
    
    def learn(self, training_data_dir_or_file_path, robot_task_servo_rate):
        