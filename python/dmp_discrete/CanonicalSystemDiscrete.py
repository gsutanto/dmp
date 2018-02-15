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
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_coupling/base/'))
from CanonicalSystem import *

class CanonicalSystemDiscrete(CanonicalSystem, object):
    'Class for canonical systems of discrete DMPs.'
    
    def __init__(self, tau_system, cs_order=2, cs_alpha=None, cs_beta=None, canonical_couplers_list=[], name=""):
        super(CanonicalSystemDiscrete, self).__init__(tau_system, canonical_couplers_list, name)
        self.order = cs_order
        assert ((self.order == 1) or (self.order == 2)), "Discrete canonical system order=" + str(self.order) + " is not supported! The only supported discrete canonical system order is either order==1 or order==2."
        if (cs_alpha != None):
            self.alpha = cs_alpha
        else:
            if (self.order == 2):
                self.alpha = 25.0
            elif (self.order == 1):
                self.alpha = 25.0/3.0
        if (cs_beta != None):
            self.beta = cs_beta
        else:
            if (self.order == 2):
                self.beta = self.alpha/4.0
            elif (self.order == 1):
                self.beta = 0.0
        self.x = 1.0
        self.xd = 0.0
        self.xdd = 0.0
        self.v = 0.0
        self.vd = 0.0
    
    def isValid(self):
        assert (super(CanonicalSystemDiscrete, self).isValid()), "CanonicalSystem is invalid!"
        assert ((self.order == 1) or (self.order == 2)), "Discrete canonical system order=" + str(self.order) + " is not supported! The only supported discrete canonical system order is either order==1 or order==2."
        assert (self.alpha > 0.0), "self.alpha=" + str(self.alpha) + " <= 0.0 (invalid!)"
        assert (self.beta >= 0.0), "self.beta=" + str(self.beta) + " < 0.0 (invalid!)"
        return True
    
    def start(self):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this CanonicalSystemDiscrete is invalid!"
        self.x = 1.0
        self.xd = 0.0
        self.xdd = 0.0
        self.v = 0.0
        self.vd = 0.0
        self.resetCouplingTerm()
        self.is_started = True
        return None
    
    def getCanonicalPosition(self):
        return copy.copy(self.x)
    
    def getCanonicalVelocity(self):
        return copy.copy(self.v)
    
    def getCanonicalMultiplier(self):
        assert ((self.order == 1) or (self.order == 2)), "Discrete canonical system order=" + str(self.order) + " is not supported! The only supported discrete canonical system order is either order==1 or order==2."
        if (self.order == 1):
            return copy.copy(self.x)
        elif (self.order == 2):
            return copy.copy(self.v)
        else: # should NEVER reach here
            return None
    
    def updateCanonicalState(self, dt):
        assert (self.is_started), "CanonicalSystemDiscrete is NOT yet started!"
        assert (self.isValid()), "Pre-condition(s) checking is failed: this CanonicalSystemDiscrete is invalid!"
        assert (dt > 0.0), "dt=" + str(dt) + " <= 0.0 (invalid!)"
        
        tau = self.tau_sys.getTauRelative()
        C_c = self.getCouplingTerm()
        if (self.order == 2):
            self.xdd = self.vd * 1.0 / tau
            self.vd = ((self.alpha * ((self.beta * (0 - self.x)) - self.v)) + C_c) * 1.0 / tau
            self.xd = self.v * 1.0 / tau
        elif (self.order == 1):
            self.xdd = self.vd * 1.0 / tau
            self.vd = 0.0
            self.xd = ((self.alpha * (0 - self.x)) + C_c) * 1.0 / tau
        self.x = self.x + (self.xd * dt)
        self.v = self.v + (self.vd * dt)
        
        assert (self.x >= 0.0), "self.x=" + str(self.x) + " < 0.0 (invalid!)"
        return None