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
from DMPTrajectory import *
from DMPState import *
from utilities import *

class CartesianCoordDMP:
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
        assert (self.ctraj_critical_states_global_coord.getTrajectoryLength() == 5)
        assert (self.ctraj_critical_states_local_coord.getTrajectoryLength() == 5)
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
        
        return None