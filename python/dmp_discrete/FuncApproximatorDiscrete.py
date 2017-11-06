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
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_base/'))
from FunctionApproximator import *
from CanonicalSystemDiscrete import *

class FuncApproximatorDiscrete(FunctionApproximator):
    'Class for function approximators of discrete DMPs.'
    
    def __init__(self, dmp_num_dimensions_init, num_basis_functions,
                 canonical_system_discrete, name=""):
        super(FuncApproximatorDiscrete, self).__init__(dmp_num_dimensions_init, num_basis_functions, canonical_system_discrete, name)
        self.centers = np.zeros((self.model_size,1))
        self.bandwidths = np.zeros((self.model_size,1))
        self.psi = np.zeros((self.model_size,1))
        self.initBasisFunctions()
    
    def initBasisFunctions(self):
        