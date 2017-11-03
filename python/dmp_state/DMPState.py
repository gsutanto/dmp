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

class DMPState:
    'Base class for DMP states.'
    
    def __init__(self, X_init=None, Xd_init=None, Xdd_init=None, time_init=0.0, name=None):
        self.name = name
        self.time = time_init
        if ((X_init is None) and (Xd_init is None) and (Xdd_init is None)):
            self.dmp_num_dimensions = 0
            self.X = np.empty((0,0))
            self.Xd = np.empty((0,0))
            self.Xdd = np.empty((0,0))
        else:
            if (X_init is not None):
                self.dmp_num_dimensions = X_init.shape[0]
                self.X = X_init
            if (Xd_init is not None):
                assert (self.dmp_num_dimensions == Xd_init.shape[0]), "Dimension Xd_init.shape[0]=" + str(Xd_init.shape[0]) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
                self.Xd = Xd_init
            else:
                self.Xd = np.zeros((self.dmp_num_dimensions,1))
            if (Xdd_init is not None):
                assert (self.dmp_num_dimensions == Xdd_init.shape[0]), "Dimension Xdd_init.shape[0]=" + str(Xdd_init.shape[0]) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
                self.Xdd = Xdd_init
            else:
                self.Xdd = np.zeros((self.dmp_num_dimensions,1))
    
    def isValid(self):
        assert (self.X.shape[0] == self.dmp_num_dimensions), "Dimension self.X.shape[0]=" + str(self.X.shape[0]) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
        assert (self.Xd.shape[0] == self.dmp_num_dimensions), "Dimension self.Xd.shape[0]=" + str(self.Xd.shape[0]) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
        assert (self.Xdd.shape[0] == self.dmp_num_dimensions), "Dimension self.Xdd.shape[0]=" + str(self.Xdd.shape[0]) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
        assert (self.time >= 0.0), "self.time=" + str(self.time) + " < 0!"
        return True
    
    def getX(self):
        return copy.copy(self.X)
    
    def getXd(self):
        return copy.copy(self.Xd)
    
    def getXdd(self):
        return copy.copy(self.Xdd)
    
    def getTime(self):
        return copy.copy(self.time)
    
    def setX(self, new_X):
        assert (self.isValid() == True), "Pre-condition(s) checking is failed: this DMPState is invalid!"
        assert (self.dmp_num_dimensions == new_X.shape[0]), "self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + " is mis-matched with new_X.shape[0]=" + str(new_X.shape[0]) + "!"
        self.X = copy.copy(new_X)
        assert (self.isValid() == True), "Post-condition(s) checking is failed: this DMPState became invalid!"
        return None
    
    def setXd(self, new_Xd):
        assert (self.isValid() == True), "Pre-condition(s) checking is failed: this DMPState is invalid!"
        assert (self.dmp_num_dimensions == new_Xd.shape[0]), "self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + " is mis-matched with new_Xd.shape[0]=" + str(new_Xd.shape[0]) + "!"
        self.Xd = copy.copy(new_Xd)
        assert (self.isValid() == True), "Post-condition(s) checking is failed: this DMPState became invalid!"
        return None
    
    def setXdd(self, new_Xdd):
        assert (self.isValid() == True), "Pre-condition(s) checking is failed: this DMPState is invalid!"
        assert (self.dmp_num_dimensions == new_Xdd.shape[0]), "self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + " is mis-matched with new_Xdd.shape[0]=" + str(new_Xdd.shape[0]) + "!"
        self.Xdd = copy.copy(new_Xdd)
        assert (self.isValid() == True), "Post-condition(s) checking is failed: this DMPState became invalid!"
        return None
    
    def setTime(self, new_time):
        assert (self.isValid() == True), "Pre-condition(s) checking is failed: this DMPState is invalid!"
        assert (new_time >= 0.0), "new_time=" + str(new_time) + " < 0.0 (invalid!)"
        self.time = new_time
        assert (self.isValid() == True), "Post-condition(s) checking is failed: this DMPState became invalid!"
        return None