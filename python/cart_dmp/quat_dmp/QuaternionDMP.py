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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
from DMPDiscrete import *
from TransformSystemQuaternion import *
from QuaternionDMPTrajectory import *
from QuaternionDMPState import *
from utilities import *
import utility_quaternion as util_quat
from utility_states_trajectories import *

class QuaternionDMP(DMPDiscrete, object):
    'Class for QuaternionDMPs.'
    
    def __init__(self, model_size_init, canonical_system_discrete, 
                 transform_couplers_list=[], name=""):
        self.transform_sys_discrete_quat = TransformSystemQuaternion(canonical_system_discrete=canonical_system_discrete, 
                                                                     func_approximator_discrete=None, # this will be initialized during the initialization of DMPDiscrete
                                                                     is_using_scaling_init=[True] * 3, 
                                                                     transform_couplers_list=transform_couplers_list, 
                                                                     ts_alpha=25.0, 
                                                                     ts_beta=25.0/4.0, 
                                                                     name="")
        super(QuaternionDMP, self).__init__(dmp_num_dimensions_init=3, 
                                            model_size_init=model_size_init, 
                                            canonical_system_discrete=canonical_system_discrete, 
                                            transform_system_discrete=self.transform_sys_discrete_quat, 
                                            name=name)
        self.mean_start_position = np.zeros((4,1))
        self.mean_goal_position = np.zeros((4,1))
        print("QuaternionDMP is created.")
    
    def isValid(self):
        assert (self.transform_sys_discrete_quat.isValid())
        assert (super(QuaternionDMP, self).isValid())
        assert (self.transform_sys_discrete == self.transform_sys_discrete_quat)
        assert (self.dmp_num_dimensions == 3)
        return True
    
    def preprocess(self, list_quat_dmp_trajectory):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this QuaternionDMP is invalid!"
        N_traj = len(list_quat_dmp_trajectory)
        
        Q0s = np.zeros((N_traj, 4))
        QGs = np.zeros((N_traj, 4))
        for i in range(N_traj):
            Q0s[i,:] = list_quat_dmp_trajectory[i].X[:,0]
            QGs[i,:] = list_quat_dmp_trajectory[i].X[:,-1]
        self.mean_start_position = util_quat.computeAverageQuaternions(Q0s)
        self.mean_goal_position = util_quat.computeAverageQuaternions(QGs)
        preprocessed_list_quat_dmp_trajectory = list_quat_dmp_trajectory
        assert (self.isValid()), "Post-condition(s) checking is failed: this QuaternionDMP became invalid!"
        return preprocessed_list_quat_dmp_trajectory
    
    def start(self, critical_states, tau_init):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this QuaternionDMP is invalid!"
        
        critical_states_length = critical_states.getLength()
        assert (critical_states_length >= 2)
        assert (critical_states.isValid())
        assert (critical_states.dmp_num_dimensions == self.dmp_num_dimensions)
        start_state_init = critical_states.getQuaternionDMPStateAtIndex(0)
        goal_state_init = critical_states.getQuaternionDMPStateAtIndex(critical_states_length-1)
        assert (start_state_init.isValid())
        assert (goal_state_init.isValid())
        assert (start_state_init.dmp_num_dimensions == self.dmp_num_dimensions)
        assert (goal_state_init.dmp_num_dimensions == self.dmp_num_dimensions)
        assert (tau_init >= MIN_TAU)
        self.tau_sys.setTauBase(tau_init)
        self.canonical_sys_discrete.start()
        self.transform_sys_discrete_quat.start(start_state_init, goal_state_init)
        self.is_started = True
        
        assert (self.isValid()), "Post-condition(s) checking is failed: this QuaternionDMP became invalid!"
        return None
    
    def extractSetTrajectories(self, training_data_dir_or_file_path, 
                               start_column_idx=1, time_column_idx=0, 
                               is_omega_and_omegad_provided=True):
        return extractSetQuaternionTrajectories(training_data_dir_or_file_path, 
                                                start_column_idx, time_column_idx, 
                                                is_omega_and_omegad_provided)
    
    def smoothStartEndTrajectoryBasedOnPosition(self, traj, percentage_padding, 
                                                percentage_smoothing_points, mode, 
                                                dt, smoothing_cutoff_frequency):
        if (dt is None):
            traj_length = traj.time.shape[1]
            assert (traj_length > 1)
            dt = (traj.time[0, traj_length-1] - traj.time[0, 0])/(1.0 * (traj_length-1))
            assert (dt > 0.0)
        return smoothStartEndQuatTrajectoryBasedOnQuaternion(Quat_traj=traj, 
                                                             percentage_padding=percentage_padding, 
                                                             percentage_smoothing_points=percentage_smoothing_points, 
                                                             mode=mode, dt=dt, 
                                                             fc=smoothing_cutoff_frequency, 
                                                             is_plotting_smoothing_comparison=False)
    
    def learnGetDefaultUnrollParams(self, set_traj_input, robot_task_servo_rate):
        W, mean_A_learn, mean_tau, Ft, Fp, QgT, cX, cV, PSI = self.learn(set_traj_input, robot_task_servo_rate)
        critical_states_list_learn = [None] * 2
        critical_states_list_learn[0] = QuaternionDMPState(self.mean_start_position)
        critical_states_list_learn[-1] = QuaternionDMPState(self.mean_goal_position)
        critical_states_learn = convertQuaternionDMPStatesListIntoQuaternionDMPTrajectory(critical_states_list_learn)
        return critical_states_learn, W, mean_A_learn, self.mean_tau, Ft, Fp, QgT, cX, cV, PSI