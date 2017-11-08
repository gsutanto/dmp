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
from TauSystem import *
from DMPState import *

class GoalSystem:
    'Base class for goal evolution systems of DMPs.'
    
    def __init__(self, dmp_num_dimensions_init, tau_system, goal_num_dimensions_init=0, current_goal_state=None, alpha_init=25.0/2.0, name=""):
        self.name = name
        self.dmp_num_dimensions = dmp_num_dimensions_init
        if (goal_num_dimensions_init > 0):
            self.goal_num_dimensions = goal_num_dimensions_init
        else:
            self.goal_num_dimensions = self.dmp_num_dimensions
        self.tau_sys = tau_system
        if (current_goal_state != None):
            self.current_goal_state = current_goal_state
        else:
            self.current_goal_state = DMPState(np.zeros((self.dmp_num_dimensions,1)))
        self.alpha = alpha_init
        self.G = np.zeros((self.goal_num_dimensions,1))
        self.is_started = False
    
    def isValid(self):
        assert (self.dmp_num_dimensions > 0), "self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + " <= 0 (invalid!)"
        assert (self.current_goal_state != None), "self.current_goal_state cannot be None!"
        assert (self.current_goal_state.isValid()), "DMPState current_goal_state is invalid!"
        assert (self.current_goal_state.dmp_num_dimensions == self.dmp_num_dimensions), "self.current_goal_state.dmp_num_dimensions=" + str(self.current_goal_state.dmp_num_dimensions) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
        assert (self.G.shape[0] == self.current_goal_state.X.shape[0])
        assert (self.tau_sys != None), "self.tau_sys cannot be None!"
        assert (self.tau_sys.isValid()), "TauSystem tau_sys is invalid!"
        assert (self.alpha > 0.0), "self.alpha=" + str(self.alpha) + " <= 0 (invalid!)"
        return True
    
    def start(self, current_goal_state_init, G_init):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this GoalSystem is invalid!"
        self.setCurrentGoalState(current_goal_state_init)
        self.setSteadyStateGoalPosition(G_init)
        self.is_started = True
        assert (self.isValid()), "Post-condition(s) checking is failed: this GoalSystem became invalid!"
        return None
    
    def getSteadyStateGoalPosition(self):
        return copy.copy(self.G)
    
    def setSteadyStateGoalPosition(self, new_G):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this GoalSystem is invalid!"
        assert (new_G.shape[0] == self.G.shape[0]), "new_G.shape[0]=" + str(new_G.shape[0]) + " is mis-matched with self.G.shape[0]=" + str(self.G.shape[0])
        self.G = copy.copy(new_G)
        assert (self.isValid()), "Post-condition(s) checking is failed: this GoalSystem became invalid!"
        return None
    
    def getCurrentGoalState(self):
        return copy.copy(self.current_goal_state)
    
    def setCurrentGoalState(self, new_current_goal_state):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this GoalSystem is invalid!"
        assert (new_current_goal_state.isValid()), "new_current_goal_state is invalid!"
        assert (new_current_goal_state.dmp_num_dimensions == self.dmp_num_dimensions), "new_current_goal_state.dmp_num_dimensions=" + str(new_current_goal_state.dmp_num_dimensions) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
        assert (new_current_goal_state.X.shape[0] == self.current_goal_state.X.shape[0]), "new_current_goal_state.X.shape[0]=" + str(new_current_goal_state.X.shape[0]) + " is mis-matched with self.current_goal_state.X.shape[0]=" + str(self.current_goal_state.X.shape[0])
        self.current_goal_state = copy.copy(new_current_goal_state)
        assert (self.isValid()), "Post-condition(s) checking is failed: this GoalSystem became invalid!"
        return None
    
    def updateCurrentGoalState(self, dt):
        assert (self.is_started), "GoalSystem is NOT yet started!"
        assert (self.isValid()), "Pre-condition(s) checking is failed: this GoalSystem is invalid!"
        assert (dt > 0.0), "dt=" + str(dt) + " <= 0.0 (invalid!)"
        
        tau = self.tau_sys.getTauRelative()
        g = self.current_goal_state.getX()
        gd = (self.alpha/tau) * (self.G - g)
        g = g + (gd * dt)
        self.current_goal_state.setX(g)
        self.current_goal_state.setXd(gd)
        self.current_goal_state.setTime(self.current_goal_state.getTime() + dt)
        
        assert (self.isValid()), "Post-condition(s) checking is failed: this GoalSystem became invalid!"
        return None