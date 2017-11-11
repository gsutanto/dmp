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
    
    def __init__(self, transformation_system, name=""):
        self.name = name
        self.transform_sys = transformation_system
    
    def isValid(self):
        assert (self.transform_sys != None), "TransformationSystem transform_sys does NOT exist!"
        assert (self.transform_sys.isValid()), "TransformationSystem transform_sys is invalid!"
        return True