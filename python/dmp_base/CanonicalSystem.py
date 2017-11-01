#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_coupling/base/'))
from TauSystem import *
from CanonicalCoupling import *

class CanonicalSystem:
    'Base class for canonical systems of DMPs.'
    
    def __init__(self, name, tau_system, canonical_couplers_list):
        self.name = name
        self.tau_sys = tau_system
        self.canonical_couplers_list = canonical_couplers_list
        self.is_started = False
    
    def isValid(self):
        assert (self.tau_sys != None), "TauSystem tau_sys does NOT exist!"
        assert (self.tau_sys.isValid() == True), "TauSystem tau_sys is invalid!"
        return True