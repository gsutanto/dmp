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

class FuncApproximatorDiscrete(FunctionApproximator, object):
    'Class for function approximators of discrete DMPs.'
    
    def __init__(self, dmp_num_dimensions_init, num_basis_functions,
                 canonical_system_discrete, name=""):
        super(FuncApproximatorDiscrete, self).__init__(dmp_num_dimensions_init, num_basis_functions, canonical_system_discrete, name)
        self.centers = np.zeros((self.model_size,1))
        self.bandwidths = np.zeros((self.model_size,1))
        self.psi = np.zeros((self.model_size,1))
        self.initBasisFunctions()
    
    def initBasisFunctions(self):
        canonical_order = self.canonical_sys.order
        assert ((canonical_order == 1) or (canonical_order == 2)), "Discrete canonical system order=" + str(canonical_order) + " is not supported! The only supported discrete canonical system order is either order==1 or order==2."
        alpha_canonical = self.canonical_sys.alpha
        
        t = np.array(np.linspace(0.0,1.0,self.model_size)).reshape((self.model_size,1)) * self.canonical_sys.tau_sys.tau_reference
        if (canonical_order == 2):
            self.centers = (1.0+((alpha_canonical/2.0)*t)) * np.exp(-(alpha_canonical/2.0)*t)
        elif (canonical_order == 1):
            self.centers = np.exp(-alpha_canonical * t)
        self.bandwidths = np.square(np.diff(self.centers,axis=0)*0.55)
        self.bandwidths = 1.0/np.vstack((self.bandwidths,self.bandwidths[-1,0]))
        
        return None
    
    def isValid(self):
        assert (super(FuncApproximatorDiscrete, self).isValid())
        assert (self.canonical_sys.isValid())
        assert (self.centers != None)
        assert (self.bandwidths != None)
        assert (self.psi != None)
        assert (self.centers.shape[0] == self.model_size)
        assert (self.bandwidths.shape[0] == self.model_size)
        assert (self.psi.shape[0] == self.model_size)
        return True
    
    def getForcingTerm(self):
        assert (self.isValid())
        
        self.psi = self.getBasisFunctionTensor(self.canonical_sys.getCanonicalPosition())
        assert (self.psi.shape == (self.model_size,1)), "self.psi must be a column vector of size self.model_size!"
        assert (np.isnan(self.psi).any()), "self.psi contains NaN!"
        sum_psi = np.sum(self.psi) + (self.model_size * 1.e-10)
        forcing_term = self.weights * self.psi * self.canonical_sys.getCanonicalMultiplier() / sum_psi
        assert (np.isnan(forcing_term).any()), "forcing_term contains NaN!"
        basis_function_vector = copy.copy(self.psi)
        
        return forcing_term, basis_function_vector
    
    def getBasisFunctionTensor(self, canonical_X):
        assert (self.isValid())
        
        if (type(canonical_X) != np.ndarray):
            cX = canonical_X * np.ones((1,1))
        else:
            assert (canonical_X.shape[0] == 1), "canonical_X must be a row vector!"
            cX = canonical_X
        basis_function_tensor = np.exp(-0.5 * np.square((np.ones(self.centers.shape) * cX) - (self.centers * np.ones(cX.shape))) * (self.bandwidths * np.ones(cX.shape)))
        return basis_function_tensor