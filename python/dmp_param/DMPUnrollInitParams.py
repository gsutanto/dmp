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
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_state/'))
from DMPState import *
from DMPTrajectory import *
from TauSystem import *

class DMPUnrollInitParams:
    'Class for DMP Unroll Initialization Parameters.'
    
    def __init__(self, critical_dmptraj_init, tau_init=None, robot_task_servo_rate=None, is_zeroing_out_velocity_and_acceleration=True):
        traj_size = critical_dmptraj_init.getLength()
        assert (traj_size >= 2)
        start_dmpstate = critical_dmptraj_init.getDMPStateAtIndex(0)
        goal_dmpstate = critical_dmptraj_init.getDMPStateAtIndex(traj_size-1)
        if (tau_init == None):
            if (robot_task_servo_rate == None):
                self.tau = goal_dmpstate.getTime()[0,0] - start_dmpstate.getTime()[0,0]
            else:
                self.tau = (1.0 * (traj_size-1))/robot_task_servo_rate
        else:
            self.tau = tau_init
        assert (self.tau >= MIN_TAU), "DMPUnrollInitParams.tau=" + str(self.tau) + " < MIN_TAU"
        critical_dmpstates_list = []
        critical_dmpstates_list.append(start_dmpstate)
        if (traj_size >= 3):
            before_goal_dmpstate = critical_dmptraj_init.getDMPStateAtIndex(traj_size-2)
            critical_dmpstates_list.append(before_goal_dmpstate)
        critical_dmpstates_list.append(goal_dmpstate)
        processed_critical_states_list = []
        for critical_dmpstate in critical_dmpstates_list:
            if (is_zeroing_out_velocity_and_acceleration):
                processed_critical_states_list.append(DMPState(critical_dmpstate.getX()))
            else:
                processed_critical_states_list.append(copy.copy(critical_dmpstate))
        self.critical_states = convertDMPStatesListIntoDMPTrajectory(processed_critical_states_list)
    
    def isValid(self):
        assert (self.tau >= MIN_TAU), "DMPUnrollInitParams.tau=" + str(self.tau) + " < MIN_TAU"
        assert (self.critical_states.getLength() >= 2)
        assert (self.critical_states.isValid())
        return True

def getDMPUnrollInitParams(start_dmpstate, goal_dmpstate, tau_init=None, robot_task_servo_rate=None, is_zeroing_out_velocity_and_acceleration=True):
    critical_dmptraj_init = convertDMPStatesListIntoDMPTrajectory([start_dmpstate, goal_dmpstate])
    return DMPUnrollInitParams(critical_dmptraj_init, tau_init, robot_task_servo_rate, 
                               is_zeroing_out_velocity_and_acceleration)