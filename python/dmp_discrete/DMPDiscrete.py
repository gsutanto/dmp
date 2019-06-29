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
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_base/'))
from DMP import *
from TransformSystemDiscrete import *
from FuncApproximatorDiscrete import *
from CanonicalSystemDiscrete import *
from LearningSystemDiscrete import *
from TauSystem import *

class DMPDiscrete(DMP, object):
    'Class for discrete DMPs.'
    
    def __init__(self, dmp_num_dimensions_init, model_size_init,
                 canonical_system_discrete, transform_system_discrete, name=""):
        self.canonical_sys_discrete = canonical_system_discrete
        self.transform_sys_discrete = transform_system_discrete
        self.func_approx_discrete = FuncApproximatorDiscrete(dmp_num_dimensions_init, model_size_init,
                                                             self.canonical_sys_discrete, name)
        self.transform_sys_discrete.func_approx = self.func_approx_discrete
        self.learning_sys_discrete = LearningSystemDiscrete(self.transform_sys_discrete, name)
        super(DMPDiscrete, self).__init__(self.canonical_sys_discrete, self.func_approx_discrete,
                                          self.transform_sys_discrete, self.learning_sys_discrete, 
                                          name)
    
    def isValid(self):
        assert (super(DMPDiscrete, self).isValid())
        assert (self.canonical_sys_discrete != None)
        assert (self.transform_sys_discrete != None)
        assert (self.canonical_sys_discrete.isValid())
        assert (self.transform_sys_discrete.isValid())
        assert (self.func_approx_discrete.isValid())
        assert (self.learning_sys_discrete.isValid())
        return True
    
    def learn(self, list_dmptrajectory_demo_global, robot_task_servo_rate):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this DMPDiscrete is invalid!"
        assert (robot_task_servo_rate > 0.0)
        
        list_dmptrajectory_demo_local = self.preprocess(list_dmptrajectory_demo_global)
        W, mean_A_learn, mean_tau, Ft, Fp, G, cX, cV, PSI = self.learning_sys_discrete.learnApproximator(list_dmptrajectory_demo_local, robot_task_servo_rate)
        self.mean_tau = mean_tau
        assert (self.mean_tau >= MIN_TAU)
        assert (self.isValid()), "Post-condition(s) checking is failed: this DMPDiscrete became invalid!"
        return W, mean_A_learn, mean_tau, Ft, Fp, G, cX, cV, PSI
    
    def start(self, critical_states, tau_init):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this DMPDiscrete is invalid!"
        
        traj_size = critical_states.getLength()
        assert (traj_size >= 2)
        start_state_init = critical_states.getDMPStateAtIndex(0)
        goal_state_init = critical_states.getDMPStateAtIndex(traj_size-1)
        assert (start_state_init.isValid())
        assert (goal_state_init.isValid())
        assert (start_state_init.dmp_num_dimensions == self.dmp_num_dimensions)
        assert (goal_state_init.dmp_num_dimensions == self.dmp_num_dimensions)
        assert (tau_init >= MIN_TAU)
        self.tau_sys.setTauBase(tau_init)
        self.canonical_sys_discrete.start()
        self.transform_sys_discrete.start(start_state_init, goal_state_init)
        self.is_started = True
        
        assert (self.isValid()), "Post-condition(s) checking is failed: this DMPDiscrete became invalid!"
        return None
    
    def getNextState(self, dt, update_canonical_state):
        assert (self.is_started == True)
        assert (self.isValid()), "Pre-condition(s) checking is failed: this DMPDiscrete is invalid!"
        
        next_state, forcing_term, ct_acc, ct_vel, basis_function_vector = self.transform_sys_discrete.getNextState(dt)
        
        if (update_canonical_state):
            self.canonical_sys_discrete.updateCanonicalState(dt)
        self.transform_sys_discrete.updateCurrentGoalState(dt)
        
        assert (self.isValid()), "Post-condition(s) checking is failed: this DMPDiscrete became invalid!"
        return next_state, forcing_term, ct_acc, ct_vel, basis_function_vector
    
    def getCurrentState(self):
        return self.transform_sys_discrete.getCurrentState()
    
    def getCurrentGoalPosition(self):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this DMPDiscrete is invalid!"
        return self.transform_sys_discrete.getCurrentGoalState().getX()
    
    def getSteadyStateGoalPosition(self):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this DMPDiscrete is invalid!"
        return self.transform_sys_discrete.getSteadyStateGoalPosition()
    
    def setNewSteadyStateGoalPosition(self, new_G):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this DMPDiscrete is invalid!"
        return self.transform_sys_discrete.setSteadyStateGoalPosition(new_G)
    
    def setScalingUsage(self, is_using_scaling_init):
        return self.transform_sys_discrete.setScalingUsage(is_using_scaling_init)
    
    def getParams(self):
        weights = self.func_approx_discrete.getWeights()
        A_learn = self.transform_sys_discrete.getLearningAmplitude()
        return weights, A_learn
    
    def setParams(self, new_weights, new_A_learn):
        self.transform_sys_discrete.setLearningAmplitude(new_A_learn)
        return self.func_approx_discrete.setWeights(new_weights)
    
    def getParamsAsDict(self):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this DMPDiscrete is invalid!"
        dmp_params = {}
        dmp_params['canonical_order'] = self.canonical_sys_discrete.order
        dmp_params['W'], dmp_params['A_learn'] = self.getParams()
        dmp_params['mean_start_position'] = copy.copy(self.mean_start_position)
        dmp_params['mean_goal_position'] = copy.copy(self.mean_goal_position)
        dmp_params['mean_tau'] = self.mean_tau
        assert (self.isValid()), "Post-condition(s) checking is failed: this DMPDiscrete became invalid!"
        return dmp_params
    
    def setParamsFromDict(self, dmp_params):
        # gsutanto's Remarks:
        # The following setting of the canonical system order may be risky.
        # Please re-think/consider it (for all possible cases) more carefully!!!
        # (for now, user must set this order manually during the initialization of 
        #  the canonical system...once set, it won't be changed/updated!!!
        #  In all cases so far, canonical system order==2 works best, 
        #  and can be considered as the default value selection for this.)
#        if ('canonical_order' in dmp_params.keys()):
#            self.canonical_sys_discrete.setOrder(dmp_params['canonical_order'])
        self.setParams(dmp_params['W'], dmp_params['A_learn'])
        self.mean_start_position = dmp_params['mean_start_position']
        self.mean_goal_position = dmp_params['mean_goal_position']
        self.mean_tau = dmp_params['mean_tau']
        assert (self.isValid()), "Post-condition(s) checking is failed: this DMPDiscrete became invalid!"
        return None
    
    def loadParamsAsDict(self, dir_path, 
                         file_name_weights="f_weights_matrix.txt", 
                         file_name_A_learn="f_A_learn_matrix.txt",
                         file_name_mean_start_position="mean_start_position.txt", 
                         file_name_mean_goal_position="mean_goal_position.txt", 
                         file_name_mean_tau="mean_tau.txt", 
                         file_name_canonical_system_order="canonical_order.txt", 
                         file_name_mean_start_position_global=None, 
                         file_name_mean_goal_position_global=None, 
                         file_name_mean_start_position_local=None, 
                         file_name_mean_goal_position_local=None, 
                         file_name_ctraj_local_coordinate_frame_selection=None, 
                         file_name_ctraj_hmg_transform_local_to_global_matrix=None, 
                         file_name_ctraj_hmg_transform_global_to_local_matrix=None):
        assert (file_name_mean_start_position_global is None)
        assert (file_name_mean_goal_position_global is None)
        assert (file_name_mean_start_position_local is None)
        assert (file_name_mean_goal_position_local is None)
        assert (file_name_ctraj_local_coordinate_frame_selection is None)
        assert (file_name_ctraj_hmg_transform_local_to_global_matrix is None)
        assert (file_name_ctraj_hmg_transform_global_to_local_matrix is None)
        assert (os.path.isdir(dir_path)), dir_path + " is NOT a directory!"
        dmp_params = {}
        canonical_system_order_file_path = dir_path + "/" + file_name_canonical_system_order
        if (os.path.isfile(canonical_system_order_file_path)):
            dmp_params['canonical_order'] = int(round(np.loadtxt(canonical_system_order_file_path)))
        dmp_params['W'] = np.loadtxt(dir_path + "/" + file_name_weights)
        dmp_params['A_learn'] = np.loadtxt(dir_path + "/" + file_name_A_learn)
        dmp_params['mean_start_position'] = np.loadtxt(dir_path + "/" + file_name_mean_start_position)
        dmp_params['mean_goal_position'] = np.loadtxt(dir_path + "/" + file_name_mean_goal_position)
        dmp_params['mean_tau'] = np.loadtxt(dir_path + "/" + file_name_mean_tau)
        assert (self.isValid()), "Post-condition(s) checking is failed: this DMPDiscrete became invalid!"
        return dmp_params
    
    def saveParamsFromDict(self, dir_path, dmp_params, 
                           file_name_weights="f_weights_matrix.txt", 
                           file_name_A_learn="f_A_learn_matrix.txt",
                           file_name_mean_start_position="mean_start_position.txt", 
                           file_name_mean_goal_position="mean_goal_position.txt", 
                           file_name_mean_tau="mean_tau.txt", 
                           file_name_canonical_system_order="canonical_order.txt", 
                           file_name_mean_start_position_global=None, 
                           file_name_mean_goal_position_global=None, 
                           file_name_mean_start_position_local=None, 
                           file_name_mean_goal_position_local=None, 
                           file_name_ctraj_local_coordinate_frame_selection=None, 
                           file_name_ctraj_hmg_transform_local_to_global_matrix=None, 
                           file_name_ctraj_hmg_transform_global_to_local_matrix=None):
        assert (file_name_mean_start_position_global is None)
        assert (file_name_mean_goal_position_global is None)
        assert (file_name_mean_start_position_local is None)
        assert (file_name_mean_goal_position_local is None)
        assert (file_name_ctraj_local_coordinate_frame_selection is None)
        assert (file_name_ctraj_hmg_transform_local_to_global_matrix is None)
        assert (file_name_ctraj_hmg_transform_global_to_local_matrix is None)
        assert (self.isValid()), "Pre-condition(s) checking is failed: this DMPDiscrete is invalid!"
        assert (os.path.isdir(dir_path)), dir_path + " is NOT a directory!"
        if ('canonical_order' in dmp_params.keys()):
            np.savetxt(dir_path + "/" + file_name_canonical_system_order, [dmp_params['canonical_order']])
        np.savetxt(dir_path + "/" + file_name_weights, dmp_params['W'])
        np.savetxt(dir_path + "/" + file_name_A_learn, dmp_params['A_learn'])
        np.savetxt(dir_path + "/" + file_name_mean_start_position, dmp_params['mean_start_position'])
        np.savetxt(dir_path + "/" + file_name_mean_goal_position, dmp_params['mean_goal_position'])
        np.savetxt(dir_path + "/" + file_name_mean_tau, [dmp_params['mean_tau']])
        assert (self.isValid()), "Post-condition(s) checking is failed: this DMPDiscrete became invalid!"
        return None
    
    def loadParams(self, dir_path, 
                   file_name_weights="f_weights_matrix.txt", 
                   file_name_A_learn="f_A_learn_matrix.txt",
                   file_name_mean_start_position="mean_start_position.txt", 
                   file_name_mean_goal_position="mean_goal_position.txt", 
                   file_name_mean_tau="mean_tau.txt", 
                   file_name_canonical_system_order="canonical_order.txt", 
                   file_name_mean_start_position_global=None, 
                   file_name_mean_goal_position_global=None, 
                   file_name_mean_start_position_local=None, 
                   file_name_mean_goal_position_local=None, 
                   file_name_ctraj_local_coordinate_frame_selection=None, 
                   file_name_ctraj_hmg_transform_local_to_global_matrix=None, 
                   file_name_ctraj_hmg_transform_global_to_local_matrix=None):
        dmp_params = self.loadParamsAsDict(dir_path, file_name_weights, 
                                           file_name_A_learn,
                                           file_name_mean_start_position, 
                                           file_name_mean_goal_position, 
                                           file_name_mean_tau, 
                                           file_name_canonical_system_order, 
                                           file_name_mean_start_position_global, 
                                           file_name_mean_goal_position_global, 
                                           file_name_mean_start_position_local, 
                                           file_name_mean_goal_position_local, 
                                           file_name_ctraj_local_coordinate_frame_selection, 
                                           file_name_ctraj_hmg_transform_local_to_global_matrix, 
                                           file_name_ctraj_hmg_transform_global_to_local_matrix)
        return self.setParamsFromDict(dmp_params)
    
    def saveParams(self, dir_path, 
                   file_name_weights="f_weights_matrix.txt", 
                   file_name_A_learn="f_A_learn_matrix.txt",
                   file_name_mean_start_position="mean_start_position.txt", 
                   file_name_mean_goal_position="mean_goal_position.txt", 
                   file_name_mean_tau="mean_tau.txt", 
                   file_name_canonical_system_order="canonical_order.txt", 
                   file_name_mean_start_position_global=None, 
                   file_name_mean_goal_position_global=None, 
                   file_name_mean_start_position_local=None, 
                   file_name_mean_goal_position_local=None, 
                   file_name_ctraj_local_coordinate_frame_selection=None, 
                   file_name_ctraj_hmg_transform_local_to_global_matrix=None, 
                   file_name_ctraj_hmg_transform_global_to_local_matrix=None):
        dmp_params = self.getParamsAsDict()
        return self.saveParamsFromDict(dir_path, dmp_params, 
                                       file_name_weights, 
                                       file_name_A_learn,
                                       file_name_mean_start_position, 
                                       file_name_mean_goal_position, 
                                       file_name_mean_tau, 
                                       file_name_canonical_system_order, 
                                       file_name_mean_start_position_global, 
                                       file_name_mean_goal_position_global, 
                                       file_name_mean_start_position_local, 
                                       file_name_mean_goal_position_local, 
                                       file_name_ctraj_local_coordinate_frame_selection, 
                                       file_name_ctraj_hmg_transform_local_to_global_matrix, 
                                       file_name_ctraj_hmg_transform_global_to_local_matrix)