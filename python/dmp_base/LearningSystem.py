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

class LearningSystem:
    'Base class for learning systems of DMPs.'
    
    def __init__(self, dmp_num_dimensions_init, model_size_init, transformation_system, name=""):
        self.name = name
        self.dmp_num_dimensions = dmp_num_dimensions_init
        self.model_size = model_size_init
        self.transform_sys = transformation_system
        self.learned_weights = np.zeros((self.dmp_num_dimensions, self.model_size))
    
    def isValid(self):
        assert (self.dmp_num_dimensions > 0), "self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "<= 0 (invalid!)"
        assert (self.model_size > 0), "self.model_size=" + str(self.model_size) + "<= 0 (invalid!)"
        assert (self.transform_sys != None), "TransformationSystem transform_sys does NOT exist!"
        assert (self.transform_sys.isValid() == True), "TransformationSystem transform_sys is invalid!"
        assert (self.learned_weights != None), "learned_weights does NOT exist!"
        assert ((self.learned_weights.shape[0] == self.dmp_num_dimensions) and (self.learned_weights.shape[1] == self.model_size)), "learned_weights matrix dimensions=" + str(self.learned_weights.shape[0]) + "X" + self.learned_weights.shape[1] + " is/are mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + " and/or self.model_size=" + str(self.model_size)
        assert (self.model_size == self.transform_sys.func_approx.model_size), "self.model_size=" + str(self.model_size) + " is/are mis-matched with self.transform_sys.func_approx.model_size=" + str(self.transform_sys.func_approx.model_size)
        return True