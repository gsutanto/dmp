#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 21:00:00 2018

@author: gsutanto
"""

import numpy as np
import numpy.linalg as npla
import numpy.matlib as npma
import os
import sys
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_goal_system/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
from TauSystem import *
from TransformSystemDiscrete import *
from FuncApproximatorDiscrete import *
from CanonicalSystemDiscrete import *
from QuaternionGoalSystem import *
from DMPState import *
from QuaternionDMPState import *
import utility_quaternion as util_quat

class TransformSystemQuaternion(TransformSystemDiscrete, object):
    'Class for transformation systems of QuaternionDMPs.'
    
    def __init__(self, canonical_system_discrete, func_approximator_discrete, 
                 is_using_scaling_init, transform_couplers_list=[], 
                 ts_alpha=25.0, ts_beta=25.0/4.0, name=""):
        self.start_quat_state = QuaternionDMPState()
        self.current_quat_state = QuaternionDMPState()
        self.current_angular_velocity_state = DMPState(X_init=np.zeros((3,1)))
        self.quat_goal_sys = QuaternionGoalSystem(canonical_system_discrete.tau_sys)
        super(TransformSystemQuaternion, self).__init__(dmp_num_dimensions_init=3, 
                                                        canonical_system_discrete=canonical_system_discrete, 
                                                        func_approximator_discrete=func_approximator_discrete, 
                                                        is_using_scaling_init=is_using_scaling_init, 
                                                        ts_alpha=ts_alpha, 
                                                        ts_beta=ts_beta,
                                                        start_dmpstate_discrete=self.start_quat_state, 
                                                        current_dmpstate_discrete=self.current_quat_state, 
                                                        current_velocity_dmpstate_discrete=self.current_angular_velocity_state, 
                                                        goal_system_discrete=self.quat_goal_sys,
                                                        transform_couplers_list=[], name)
    
    def isValid(self):
        assert (super(TransformSystemQuaternion, self).isValid())
        assert (self.dmp_num_dimensions == 3)
        assert (self.start_quat_state.isValid())
        assert (self.current_quat_state.isValid())
        assert (self.current_angular_velocity_state.isValid())
        assert (self.quat_goal_sys.isValid())
        assert (self.start_state == self.start_quat_state)
        assert (self.current_state == self.current_quat_state)
        assert (self.current_velocity_state == self.current_angular_velocity_state)
        assert (self.goal_sys == self.quat_goal_sys)
        return True
    
    def start(self, start_quat_state_init, goal_quat_state_init):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this TransformSystemQuaternion is invalid!"
        assert (start_quat_state_init.isValid())
        assert (goal_quat_state_init.isValid())
        assert (start_quat_state_init.dmp_num_dimensions == self.dmp_num_dimensions)
        assert (goal_quat_state_init.dmp_num_dimensions == self.dmp_num_dimensions)
        
        self.start_quat_state = start_quat_state_init
        self.setCurrentState(start_quat_state_init)
        
        QG_init = goal_quat_state_init.getQ()
        current_goal_quat_state_init = QuaternionDMPState()
        if (self.canonical_sys.order == 2):
            # Best option for Schaal's DMP Model using 2nd order canonical system:
            # Using goal evolution system initialized with the start position (state) as goal position (state),
            # which over time evolves toward a steady-state goal position (state).
            # The main purpose is to avoid discontinuous initial acceleration (see the math formula
            # of Schaal's DMP Model for more details).
            # Please also refer the initialization described in paper:
            # B. Nemec and A. Ude, “Action sequencing using dynamic movement
            # primitives,” Robotica, vol. 30, no. 05, pp. 837–846, 2012.
            x0 = start_state_init.getX()
            xd0 = start_state_init.getXd()
            xdd0 = start_state_init.getXdd()
            
            tau = self.tau_sys.getTauRelative()
            g0 = ((((tau*tau*xdd0) * 1.0 / self.alpha) + (tau*xd0)) * 1.0 / self.beta) + x0
            current_goal_state_init = DMPState(g0)
        elif (self.canonical_sys.order == 1):
            current_goal_state_init = goal_state_init
        
        self.goal_sys.start(current_goal_state_init, G_init)
        self.resetCouplingTerm()
        self.is_started = True
        
        assert (self.isValid()), "Post-condition(s) checking is failed: this TransformSystemDiscrete is invalid!"
        return None