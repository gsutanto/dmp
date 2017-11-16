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
from DMPState import *

class DMPTrajectory(DMPState, object):
    'Class for DMP trajectories.'
    
    def getTrajectoryLength(self):
        return (super(DMPTrajectory, self).getLength())
    
    def getDMPStateAtIndex(self, i):
        assert (super(DMPTrajectory, self).isValid()), "DMPTrajectory is invalid!"
        assert ((i >= 0) and (i < self.time.shape[1])), "Index i=" + str(i) + " is out-of-range (TrajectoryLength=" + str(self.time.shape[1]) + ")!"
        return DMPState(self.X[:,i:i+1], self.Xd[:,i:i+1], self.Xdd[:,i:i+1], self.time[:,i:i+1])

def convertDMPStatesListIntoDMPTrajectory(dmpstates_list):
    X_list = []
    Xd_list = []
    Xdd_list = []
    time_list = []
    for dmpstate in dmpstates_list:
        X_list.append(dmpstate.X)
        Xd_list.append(dmpstate.Xd)
        Xdd_list.append(dmpstate.Xdd)
        time_list.append(dmpstate.time)
    return DMPTrajectory(np.concatenate(X_list,axis=1),
                         np.concatenate(Xd_list,axis=1),
                         np.concatenate(Xdd_list,axis=1),
                         np.concatenate(time_list,axis=1))