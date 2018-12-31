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
    
    def isValid(self):
        assert (self.transform_sys_discrete_quat.isValid())
        assert (super(QuaternionDMP, self).isValid())
        assert (self.transform_sys_discrete == self.transform_sys_discrete_quat)
        assert (self.dmp_num_dimensions == 3)
        return True
    
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