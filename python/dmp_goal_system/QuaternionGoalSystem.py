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
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
from TauSystem import *
from GoalSystem import *
from QuaternionDMPState import *
import utility_quaternion as util_quat

class QuaternionGoalSystem(GoalSystem, object):
    'Class for Quaternion goal evolution systems of QuaternionDMPs.'
    
    def __init__(self, tau_system, alpha_init=25.0/2.0, name=""):
        self.current_quaternion_goal_state = QuaternionDMPState(name=name+"_current_quaternion_goal_state")
        super(QuaternionGoalSystem, self).__init__(dmp_num_dimensions_init=3, 
                                                   tau_system=tau_system, 
                                                   goal_num_dimensions_init=4, 
                                                   current_goal_state=self.current_quaternion_goal_state, 
                                                   alpha_init=alpha_init, 
                                                   name=name)
    
    def isValid(self):
        assert (self.current_quaternion_goal_state.isValid())
        assert (super(QuaternionGoalSystem, self).isValid())
        assert (self.current_goal_state == self.current_quaternion_goal_state)
        return True
    
    def updateCurrentGoalState(self, dt):
        assert (self.is_started), "QuaternionGoalSystem is NOT yet started!"
        assert (self.isValid()), "Pre-condition(s) checking is failed: this QuaternionGoalSystem is invalid!"
        assert (dt > 0.0), "dt=" + str(dt) + " <= 0.0 (invalid!)"
        
        tau = self.tau_sys.getTauRelative()
        QG = self.G
        Qg = self.current_quaternion_goal_state.getQ()
        omegag = self.current_quaternion_goal_state.getOmega()
        
        twice_log_quat_diff_g = util_quat.computeTwiceLogQuatDifference( QG, Qg )
        next_omegag = (self.alpha * 1.0 / tau) * twice_log_quat_diff_g
        next_Qg = util_quat.integrateQuat( Qg, next_omegag, dt )
        next_omegagd = (next_omegag - omegag)/dt
        self.current_quaternion_goal_state.setQ(next_Qg)
        self.current_quaternion_goal_state.setOmega(next_omegag)
        self.current_quaternion_goal_state.setOmegad(next_omegagd)
        self.current_quaternion_goal_state.computeQdAndQdd()
        self.current_quaternion_goal_state.setTime(self.current_quaternion_goal_state.getTime() + dt)
        
        assert (self.isValid()), "Post-condition(s) checking is failed: this QuaternionGoalSystem became invalid!"
        return None