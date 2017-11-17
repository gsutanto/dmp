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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
from CartesianCoordTransformer import *
from DMPDiscrete import *
from TransformSystemDiscrete import *
from DMPTrajectory import *
from DMPState import *
from utilities import *

class CartesianCoordDMP(DMPDiscrete, object):
    'Class for discrete Cartesian Coordinate (x-y-z) DMPs.'
    
    def __init__(self, model_size_init, canonical_system_discrete, ctraj_local_coordinate_frame_selection,
                 transform_couplers_list=[], name=""):
        self.transform_sys_discrete_cart_coord = TransformSystemDiscrete(3, canonical_system_discrete, 
                                                                         None, 
                                                                         [True] * 3, 25.0, 25.0/4.0,
                                                                         None, None, None, None,
                                                                         transform_couplers_list)
        super(CartesianCoordDMP, self).__init__(3, model_size_init, canonical_system_discrete, 
                                                self.transform_sys_discrete_cart_coord, name)
        self.cart_coord_transformer = CartesianCoordTransformer()
        self.ctraj_local_coord_selection = ctraj_local_coordinate_frame_selection
        is_using_scaling_init = [True] * 3
        if ((self.ctraj_local_coord_selection == _GSUTANTO_LOCAL_COORD_FRAME_) or 
            (self.ctraj_local_coord_selection == _SCHAAL_LOCAL_COORD_FRAME_)):
            is_using_scaling_init[1] = False
            is_using_scaling_init[2] = False
        elif (self.ctraj_local_coord_selection == _KROEMER_LOCAL_COORD_FRAME_):
            is_using_scaling_init[2] = False
        self.setScalingUsage(is_using_scaling_init)
        self.ctraj_hmg_transform_local_to_global_matrix = np.zeros((4,4))
        self.ctraj_hmg_transform_global_to_local_matrix = np.zeros((4,4))
        self.ctraj_critical_states_global_coord = DMPTrajectory(np.zeros((3,5)),np.zeros((3,5)),np.zeros((3,5)),np.zeros((1,5)))
        self.ctraj_critical_states_local_coord = DMPTrajectory(np.zeros((3,5)),np.zeros((3,5)),np.zeros((3,5)),np.zeros((1,5)))
    
    def isValid(self):
        assert (self.transform_sys_discrete_cart_coord.isValid())
        assert (super(CartesianCoordDMP, self).isValid())
        assert (self.transform_sys_discrete == self.transform_sys_discrete_cart_coord)
        assert (self.dmp_num_dimensions == 3)
        assert (self.cart_coord_transformer.isValid())
        assert (self.ctraj_critical_states_global_coord.getLength() == 5)
        assert (self.ctraj_critical_states_local_coord.getLength() == 5)
        assert (self.ctraj_local_coord_selection >= MIN_CTRAJ_LOCAL_COORD_OPTION_NO)
        assert (self.ctraj_local_coord_selection <= MAX_CTRAJ_LOCAL_COORD_OPTION_NO)
        return True
    
    def preprocess(self, list_ctraj_global):
        # in this case preprocessing is conversion of the trajectories representation 
        # from the global to local coordinate system:
        list_intermed_processed_ctraj_global = super(CartesianCoordDMP, self).preprocess(list_ctraj_global)
        list_ctraj_local = []
        for intermed_processed_ctraj_global in list_intermed_processed_ctraj_global:
            # computing the transformation to local coordinate system (for each individual trajectory):
            [self.ctraj_hmg_transform_local_to_global_matrix,
             self.ctraj_hmg_transform_global_to_local_matrix] = self.cart_coord_transformer.computeCTrajCoordTransforms(intermed_processed_ctraj_global, 
                                                                                                                        self.ctraj_local_coord_selection)
            ctraj_local = self.cart_coord_transformer.convertCTrajAtOldToNewCoordSys(intermed_processed_ctraj_global,
                                                                                     self.ctraj_hmg_transform_global_to_local_matrix)
            list_ctraj_local.append(ctraj_local)
        return list_ctraj_local
    
    def learnFromPath(self, training_data_dir_or_file_path, robot_task_servo_rate):
        return super(CartesianCoordDMP, self).learnFromPath(training_data_dir_or_file_path, robot_task_servo_rate)
    
    def start(self, critical_states, tau_init):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this CartesianCoordDMP is invalid!"
        
        critical_states_length = critical_states.getLength()
        assert (critical_states_length >= 2)
        assert (((self.ctraj_local_coord_selection == _KROEMER_LOCAL_COORD_FRAME_) and 
                 (critical_states_length < 3)) == False)
        assert (critical_states.isValid())
        assert (critical_states.dmp_num_dimensions == 3)
        start_state_global_init = critical_states.getDMPStateAtIndex(0)
        approaching_ss_goal_state_global_init = critical_states.getDMPStateAtIndex(critical_states_length-2)
        ss_goal_state_global_init = critical_states.getDMPStateAtIndex(critical_states_length-1)
        assert (tau_init >= MIN_TAU)
        
        current_state_global_init = copy.copy(start_state_global_init)
        if (self.canonical_sys_discrete.order == 2):
            current_goal_state_global_init = copy.copy(start_state_global_init)
        else:
            current_goal_state_global_init = copy.copy(ss_goal_state_global_init)
        list_cart_critical_states_global_coord = [None] * 5
        list_cart_critical_states_global_coord[0] = start_state_global_init
        list_cart_critical_states_global_coord[1] = current_state_global_init
        list_cart_critical_states_global_coord[2] = current_goal_state_global_init
        list_cart_critical_states_global_coord[-2] = approaching_ss_goal_state_global_init
        list_cart_critical_states_global_coord[-1] = ss_goal_state_global_init
        self.ctraj_critical_states_global_coord = convertDMPStatesListIntoDMPTrajectory(list_cart_critical_states_global_coord)
        
        self.mean_start_global_position = start_state_global_init.getX()
        self.mean_goal_global_position = ss_goal_state_global_init.getX()
        [self.ctraj_hmg_transform_local_to_global_matrix,
         self.ctraj_hmg_transform_global_to_local_matrix] = self.cart_coord_transformer.computeCTrajCoordTransforms(self.ctraj_critical_states_global_coord,
                                                                                                                    self.ctraj_local_coord_selection)
        self.ctraj_critical_states_local_coord = self.cart_coord_transformer.convertCTrajAtOldToNewCoordSys(self.ctraj_critical_states_global_coord,
                                                                                                            self.ctraj_hmg_transform_global_to_local_matrix)
        
        self.tau_sys.setTauBase(tau_init)
        self.canonical_sys_discrete.start()
        start_state_local_init = self.ctraj_critical_states_local_coord.getDMPStateAtIndex(0)
        ss_goal_state_local_init = self.ctraj_critical_states_local_coord.getDMPStateAtIndex(self.ctraj_critical_states_local_coord.getLength()-1)
        self.transform_sys_discrete_cart_coord.start(start_state_local_init, ss_goal_state_local_init)
        
        self.mean_start_local_position = start_state_local_init.getX()
        self.mean_goal_local_position = ss_goal_state_local_init.getX()
        
        self.is_started = True
        
        assert (self.isValid()), "Post-condition(s) checking is failed: this CartesianCoordDMP became invalid!"
        return None
    
    def getNextState(self, dt, update_canonical_state):
        assert (self.is_started == True)
        assert (self.isValid()), "Pre-condition(s) checking is failed: this CartesianCoordDMP is invalid!"
        
        [result_dmp_state_local, 
         transform_sys_forcing_term, 
         transform_sys_coupling_term_acc, 
         transform_sys_coupling_term_vel, 
         func_approx_basis_function_vector] = self.transform_sys_discrete_cart_coord.getNextState(dt)
        
        if (update_canonical_state):
            self.canonical_sys_discrete.updateCanonicalState(dt)
        self.transform_sys_discrete_cart_coord.updateCurrentGoalState(dt)
        
        # update critical states in local coordinate frame (only current state and current goal state changes)
        self.ctraj_critical_states_local_coord.setDMPStateAtIndex(1, result_dmp_state_local)
        self.ctraj_critical_states_local_coord.setDMPStateAtIndex(2, self.transform_sys_discrete_cart_coord.getCurrentGoalState())
        
        self.ctraj_critical_states_global_coord = self.cart_coord_transformer.convertCTrajAtOldToNewCoordSys(self.ctraj_critical_states_local_coord,
                                                                                                             self.ctraj_hmg_transform_local_to_global_matrix)
        
        result_dmp_state_global = self.ctraj_critical_states_global_coord.getDMPStateAtIndex(1)
        
        assert (self.isValid()), "Post-condition(s) checking is failed: this CartesianCoordDMP became invalid!"
        return result_dmp_state_global, result_dmp_state_local, transform_sys_forcing_term, transform_sys_coupling_term_acc, transform_sys_coupling_term_vel, func_approx_basis_function_vector
    
    def getCurrentState(self):
        # update critical states in local coordinate frame (only current state and current goal state changes)
        self.ctraj_critical_states_local_coord.setDMPStateAtIndex(1, self.transform_sys_discrete_cart_coord.getCurrentState())
        self.ctraj_critical_states_local_coord.setDMPStateAtIndex(2, self.transform_sys_discrete_cart_coord.getCurrentGoalState())
        
        self.ctraj_critical_states_global_coord = self.cart_coord_transformer.convertCTrajAtOldToNewCoordSys(self.ctraj_critical_states_local_coord,
                                                                                                             self.ctraj_hmg_transform_local_to_global_matrix)
        
        return self.ctraj_critical_states_global_coord.getDMPStateAtIndex(1)
    
    def getCurrentGoalPosition(self):
        current_goal_state_local = self.transform_sys_discrete_cart_coord.getCurrentGoalState()
        assert (current_goal_state_local.dmp_num_dimensions == 3)
        current_goal_state_global = self.cart_coord_transformer.convertCTrajAtOldToNewCoordSys(current_goal_state_local,
                                                                                               self.ctraj_hmg_transform_local_to_global_matrix)
        return current_goal_state_global.getX()
    
    def getSteadyStateGoalPosition(self):
        steady_state_goal_local = super(CartesianCoordDMP, self).getSteadyStateGoalPosition()
        steady_state_goal_local_H = np.vstack(steady_state_goal_local, np.ones((1,1)))
        steady_state_goal_global_H = self.cart_coord_transformer.computeCVecAtNewCoordSys(steady_state_goal_local_H, 
                                                                                          self.ctraj_hmg_transform_local_to_global_matrix)
        steady_state_goal_global = steady_state_goal_global_H[0:3,0].reshape(3,1)
        return steady_state_goal_global
    
    def setNewSteadyStateGoalPosition(self, new_G):
        return self.setNewSteadyStateGoalPositionAndApproachingSSGoalPosition(new_G, new_G)
    
    def setNewSteadyStateGoalPositionAndApproachingSSGoalPosition(self, new_G_global, new_approaching_G_global):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this CartesianCoordDMP is invalid!"
        
        self.ctraj_critical_states_local_coord.setDMPStateAtIndex(0, self.transform_sys_discrete_cart_coord.getStartState())
        self.ctraj_critical_states_local_coord.setDMPStateAtIndex(1, self.transform_sys_discrete_cart_coord.getCurrentState())
        self.ctraj_critical_states_local_coord.setDMPStateAtIndex(2, transform_sys_discrete_cart_coord.getCurrentGoalState())
        # self.ctraj_critical_states_local_coord.setDMPStateAtIndex(self.ctraj_critical_states_local_coord.getLength()-2, self.ctraj_critical_states_local_coord.getDMPStateAtIndex(3))
        self.ctraj_critical_states_local_coord.setDMPStateAtIndex(self.ctraj_critical_states_local_coord.getLength()-1, DMPState(self.transform_sys_discrete_cart_coord.getSteadyStateGoalPosition()))
        
        # Compute critical states on the global coordinate system
        # (using the OLD coordinate transformation):
        self.ctraj_critical_states_global_coord = self.cart_coord_transformer.convertCTrajAtOldToNewCoordSys(self.ctraj_critical_states_local_coord,
                                                                                                             self.ctraj_hmg_transform_local_to_global_matrix)
        
        # Compute the NEW coordinate system transformation (representing the new local coordinate system),
        # based on the original start state, the new approaching-steady-state goal,
        # the new steady-state goal (G, NOT the current/transient/non-steady-state goal),
        # both on global coordinate system, for trajectory reproduction:
        self.ctraj_critical_states_global_coord.setDMPStateAtIndex(self.ctraj_critical_states_global_coord.getLength()-2, DMPState(new_approaching_G_global))
        self.ctraj_critical_states_global_coord.setDMPStateAtIndex(self.ctraj_critical_states_global_coord.getLength()-1, DMPState(new_G_global))
        [self.ctraj_hmg_transform_local_to_global_matrix,
         self.ctraj_hmg_transform_global_to_local_matrix] = self.cart_coord_transformer.computeCTrajCoordTransforms(self.ctraj_critical_states_global_coord,
                                                                                                                    self.ctraj_local_coord_selection)
        self.ctraj_critical_states_local_coord = self.cart_coord_transformer.convertCTrajAtOldToNewCoordSys(self.ctraj_critical_states_global_coord,
                                                                                                            self.ctraj_hmg_transform_global_to_local_matrix)
        self.transform_sys_discrete_cart_coord.setStartState(self.ctraj_critical_states_local_coord.getDMPStateAtIndex(0))
        self.transform_sys_discrete_cart_coord.setCurrentState(self.ctraj_critical_states_local_coord.getDMPStateAtIndex(1))
        self.transform_sys_discrete_cart_coord.setCurrentGoalState(self.ctraj_critical_states_local_coord.getDMPStateAtIndex(2))
        self.transform_sys_discrete_cart_coord.setSteadyStateGoalPosition(self.ctraj_critical_states_local_coord.getDMPStateAtIndex(self.ctraj_critical_states_local_coord.getLength()-1).getX())
        
        assert (self.isValid()), "Post-condition(s) checking is failed: this CartesianCoordDMP became invalid!"
        return None
    
    def convertToGlobalDMPState(self, dmp_state_local):
        return self.cart_coord_transformer.convertCTrajAtOldToNewCoordSys(dmp_state_local,
                                                                          self.ctraj_hmg_transform_local_to_global_matrix)
        
    def convertToLocalDMPState(self, dmp_state_global):
        return self.cart_coord_transformer.convertCTrajAtOldToNewCoordSys(dmp_state_global,
                                                                          self.ctraj_hmg_transform_global_to_local_matrix)
    
    def extractSetTrajectories(self, training_data_dir_or_file_path):
        return extractSetNDTrajectories(training_data_dir_or_file_path, 3)
    
    def saveParamsCartCoordDMP(self, dir_path, 
                               file_name_weights="f_weights_matrix.txt",
                               file_name_A_learn="f_A_learn_matrix.txt",
                               file_name_mean_start_position_local="mean_start_position_local.txt",
                               file_name_mean_goal_position_local="mean_goal_position_local.txt",
                               file_name_mean_start_position_global="mean_start_position_global.txt",
                               file_name_mean_goal_position_global="mean_goal_position_global.txt",
                               file_name_ctraj_hmg_transform_local_to_global_matrix="ctraj_hmg_transform_local_to_global_matrix.txt",
                               file_name_ctraj_hmg_transform_global_to_local_matrix="ctraj_hmg_transform_global_to_local_matrix.txt",
                               file_name_mean_tau="mean_tau.txt"):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this CartesianCoordDMP is invalid!"
        super(CartesianCoordDMP, self).saveParams(dir_path,
                                                  file_name_weights,
                                                  file_name_A_learn,
                                                  file_name_mean_start_position_global,
                                                  file_name_mean_goal_position_global,
                                                  file_name_mean_tau)
        np.savetxt(dir_path + "/" + file_name_mean_start_position_local, self.mean_start_local_position)
        np.savetxt(dir_path + "/" + file_name_mean_goal_position_local, self.mean_goal_local_position)
        np.savetxt(dir_path + "/" + file_name_mean_start_position_global, self.mean_start_global_position)
        np.savetxt(dir_path + "/" + file_name_mean_goal_position_global, self.mean_goal_global_position)
        np.savetxt(dir_path + "/" + file_name_ctraj_hmg_transform_local_to_global_matrix, self.ctraj_hmg_transform_local_to_global_matrix)
        np.savetxt(dir_path + "/" + file_name_ctraj_hmg_transform_global_to_local_matrix, self.ctraj_hmg_transform_global_to_local_matrix)
        return None