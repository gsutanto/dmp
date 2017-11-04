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
from CanonicalSystem import *

class FunctionApproximator:
    'Base class for function approximators of DMPs.'
    
    def __init__(self, dmp_num_dimensions_init, model_size_init, canonical_system, name=""):
        self.name = name
        self.dmp_num_dimensions = dmp_num_dimensions_init
        self.model_size = model_size_init
        self.canonical_sys = canonical_system
        self.weights = np.zeros((self.dmp_num_dimensions, self.model_size))
    
    def isValid(self):
        assert (self.dmp_num_dimensions > 0), "self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "<= 0 (invalid!)"
        assert (self.model_size > 0), "self.model_size=" + str(self.model_size) + "<= 0 (invalid!)"
        assert (self.canonical_sys != None), "CanonicalSystem canonical_sys does NOT exist!"
        assert (self.canonical_sys.isValid() == True), "CanonicalSystem canonical_sys is invalid!"
        assert ((self.weights.shape[0] == self.dmp_num_dimensions) and (self.weights.shape[1] == self.model_size)), "Weights matrix dimensions=" + str(self.weights.shape[0]) + "X" + self.weights.shape[1] + " is/are mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + " and/or self.model_size=" + str(self.model_size)
        return True