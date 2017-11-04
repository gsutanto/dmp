#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import numpy as np
import copy

class Coupling:
    'Common base class for DMP coupling terms.'
    
    def __init__(self, name=""):
        self.name = name
    
    def isValid(self):
        return True
    
    def getValue(self):
        coupling_value = 0.0
        return coupling_value