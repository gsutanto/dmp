#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import numpy as np

class Coupling:
    'Common base class for DMP coupling terms.'
    
    def __init__(self, name=None):
        self.name = name
    
    def isValid(self):
        return True