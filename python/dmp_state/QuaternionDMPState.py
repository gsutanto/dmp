#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 22:00:00 2018

@author: gsutanto
"""

import numpy as np
import os
import sys
import copy
from DMPState import *
from utilities import *

class QuaternionDMPState(DMPState, object):
    'Class for Quaternion DMP states.'
    
    def __init__(self, Q_init=None, Qd_init=None, Qdd_init=None, 
                 omega_init=None, omegad_init=None, 
                 time_init=np.zeros((1,1)), name=""):
        super(QuaternionDMPState, self).__init__(Q_init, Qd_init, Qdd_init, time_init, name)
        self.dmp_num_dimensions = 3