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

class TCLearnObsAvoidFeatureParameter:
    'Class for a group/bundle of states of obstacles.'
    
    def __init__(self, name=""):
        self.name = name
    
    def isValid(self):
        return True