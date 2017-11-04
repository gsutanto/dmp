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
from TauSystem import *
from CanonicalSystem import *
from FunctionApproximator import *
from TransformationSystem import *
from LearningSystem import *

class DMP:
    'Base class for DMPs.'
    
    def __init__(self, dmp_num_dimensions_init, model_size_init,
                 canonical_system, function_approximator,
                 transform_system, learning_system, name=""):
        self.name = name
        self.dmp_num_dimensions = dmp_num_dimensions_init
        self.model_size = model_size_init
        self.canonical_sys = canonical_system
        self.tau_sys = self.canonical_sys.tau_sys
        self.func_approx = function_approximator
        self.transform_sys = transform_system
        self.learning_sys = learning_system
        self.is_started = False
        self.mean_start_position = np.zeros((self.dmp_num_dimensions,1))
        self.mean.goal_position = np.zeros((self.dmp_num_dimensions,1))
        self.mean_tau = 0.0
    
    def isValid(self):
        
        return True