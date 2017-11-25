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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_state/'))
from DMPState import *

class ObstacleStates(DMPState, object):
    'Class for a group/bundle of states of obstacles.'
    
    def isValid(self):
        assert (super(ObstacleStates, self).isValid())
        assert (self.time.shape[1] == 1)
        return True