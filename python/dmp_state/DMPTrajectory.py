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
        assert (super(DMPTrajectory, self).isValid() == True), "DMPTrajectory is invalid!"
        return self.time.shape[1]
    
    def getDMPStateAtIndex(self, i):
        assert (super(DMPTrajectory, self).isValid() == True), "DMPTrajectory is invalid!"
        assert ((i >= 0) and (i < self.time.shape[1])), "Index i=" + str(i) + " is out-of-range (TrajectoryLength=" + str(self.time.shape[1]) + ")!"
        return DMPState(self.X[:,i:i+1], self.Xd[:,i:i+1], self.Xdd[:,i:i+1], self.time[:,i:i+1])