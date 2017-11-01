#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import math
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_coupling/base/'))
from TauCoupling import *

MIN_TAU = 0.01

class TauSystem:
    'Class defining computation of tau parameter for canonical system, transformation system, and goal evolution system, which sometime is coupled with an external variable.'
    
    def __init__(self, name, tau_base_init, tau_ref=0.5, tau_couplers_list=[]):
        self.name = name
        self.tau_base = tau_base_init
        self.tau_reference = tau_ref
        self.tau_couplers_list = tau_couplers_list
    
    def isValid(self):
        assert (self.tau_base >= MIN_TAU), "TauSystem.tau_base have to be >= MIN_TAU"
        assert (self.tau_reference >= MIN_TAU), "TauSystem.tau_reference have to be >= MIN_TAU"
        return True
    
    def setTauBase(self, tau_base_new):
        assert (tau_base_new >= MIN_TAU), "tau_base_new have to be >= MIN_TAU"
        self.tau_base = tau_base_new
        return None
    
    def getTauRelative(self):
        assert (self.isValid() == True), "TauSystem is invalid!"
        C_tau = self.getCouplingTerm()
        tau_relative = (1.0 + C_tau) * (self.tau_base / self.tau_reference)
        assert (tau_relative * self.tau_reference >= MIN_TAU), 'tau_relative is too small!'
        return tau_relative
    
    def getCouplingTerm(self):
        accumulated_ctau = 0.0
        for tau_coupler_idx in range(len(self.tau_couplers_list)):
            tau_coupling_value = self.tau_couplers_list[tau_coupler_idx].getValue()
            assert (math.isnan(tau_coupling_value) == False), 'tau_coupling_value['+str(tau_coupler_idx)+'] is NaN!'
            accumulated_ctau = accumulated_ctau + tau_coupling_value
        return accumulated_ctau