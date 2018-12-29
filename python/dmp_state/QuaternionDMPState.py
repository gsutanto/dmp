#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 22:00:00 2018

@author: gsutanto
"""

import numpy as np
import numpy.linalg as npla
import numpy.matlib as npma
import os
import sys
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities/'))
from DMPState import *
from utilities import *
import utility_quaternion as util_quat

class QuaternionDMPState(DMPState, object):
    'Class for Quaternion DMP states.'
    
    def __init__(self, Q_init=None, Qd_init=None, Qdd_init=None, 
                 omega_init=None, omegad_init=None, 
                 time_init=np.zeros((1,1)), name=""):
        self.name = name
        self.time = time_init
        self.dmp_num_dimensions = 3
        if ((Q_init is None) and (Qd_init is None) and (Qdd_init is None) and 
            (omega_init is None) and (omegad_init in None)):
            self.X = np.empty((4,0))
            self.Xd = np.empty((4,0))
            self.Xdd = np.empty((4,0))
            self.omega = np.empty((self.dmp_num_dimensions,0))
            self.omegad = np.empty((self.dmp_num_dimensions,0))
        else:
            if (Q_init is not None):
                assert (Q_init.shape[0] == 4), "Dimension Q_init.shape[0]=" + str(Q_init.shape[0]) + " is NOT 4 (invalid Quaternion)!"
                self.X = Q_init
            else:
                assert (False), 'Q_init CANNOT be empty!'
            if (Qd_init is not None):
                assert (Qd_init.shape[0] == 4), "Dimension Qd_init.shape[0]=" + str(Qd_init.shape[0]) + " is NOT 4!"
                self.Xd = Qd_init
            else:
                self.Xd = np.zeros((4,self.X.shape[1]))
            if (Qdd_init is not None):
                assert (Qdd_init.shape[0] == 4), "Dimension Qdd_init.shape[0]=" + str(Qdd_init.shape[0]) + " is NOT 4!"
                self.Xdd = Qdd_init
            else:
                self.Xdd = np.zeros((4,self.X.shape[1]))
            if (omega_init is not None):
                assert (omega_init.shape[0] == self.dmp_num_dimensions), "Dimension omega_init.shape[0]=" + str(omega_init.shape[0]) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
                self.omega = omega_init
            else:
                self.omega = np.zeros((self.dmp_num_dimensions,self.X.shape[1]))
            if (omegad_init is not None):
                assert (omegad_init.shape[0] == self.dmp_num_dimensions), "Dimension omegad_init.shape[0]=" + str(omegad_init.shape[0]) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
                self.omegad = omegad_init
            else:
                self.omegad = np.zeros((self.dmp_num_dimensions,self.X.shape[1]))
            if (len(self.X.shape) == 1):
                self.X.reshape((4, 1))
            if (len(self.Xd.shape) == 1):
                self.Xd.reshape((4, 1))
            if (len(self.Xdd.shape) == 1):
                self.Xdd.reshape((4, 1))
            if (len(self.omega.shape) == 1):
                self.omega.reshape((self.dmp_num_dimensions, 1))
            if (len(self.omegad.shape) == 1):
                self.omegad.reshape((self.dmp_num_dimensions, 1))
            if (len(self.time.shape) == 1):
                self.time.reshape((1, 1))
            assert (self.isValid())
    
    def isValid(self):
        assert (self.dmp_num_dimensions == 3), "self.dmp_num_dimensions =" + str(self.dmp_num_dimensions)
        assert (len(self.X.shape) == 2), "self.X.shape =" + str(self.X.shape)
        assert (len(self.Xd.shape) == 2), "self.Xd.shape =" + str(self.Xd.shape)
        assert (len(self.Xdd.shape) == 2), "self.Xdd.shape =" + str(self.Xdd.shape)
        assert (len(self.omega.shape) == 2), "self.omega.shape =" + str(self.omega.shape)
        assert (len(self.omegad.shape) == 2), "self.omegad.shape =" + str(self.omegad.shape)
        assert (len(self.time.shape) == 2), "self.time.shape =" + str(self.time.shape)
        assert (self.X.shape[0] == 4), "Dimension self.X.shape[0]=" + str(self.X.shape[0]) + " is NOT 4 (invalid Quaternion)!"
        assert (npla.norm(self.X, ord=2, axis=0) > 0.0).all(), "npla.norm(self.X, ord=2, axis=0) =" + str(npla.norm(self.X, ord=2, axis=0))
        assert (self.Xd.shape[0] == 4), "Dimension self.Xd.shape[0]=" + str(self.Xd.shape[0]) + " is NOT 4!"
        assert (self.Xdd.shape[0] == 4), "Dimension self.Xdd.shape[0]=" + str(self.Xdd.shape[0]) + " is NOT 4!"
        assert (self.omega.shape[0] == self.dmp_num_dimensions), "Dimension self.omega.shape[0]=" + str(self.omega.shape[0]) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
        assert (self.omegad.shape[0] == self.dmp_num_dimensions), "Dimension self.omegad.shape[0]=" + str(self.omegad.shape[0]) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
        assert (self.time.shape[0] == 1)
        assert (self.X.shape[1] == self.Xd.shape[1])
        assert (self.X.shape[1] == self.Xdd.shape[1])
        assert (self.X.shape[1] == self.omega.shape[1])
        assert (self.X.shape[1] == self.omegad.shape[1])
        assert (self.time.shape[1] >= 1)
        assert (np.amin(self.time) >= 0.0), "min(self.time)=" + str(np.amin(self.time)) + " < 0.0 (invalid!)"
        return True
    
    def getQ(self):
        return copy.copy(self.X)
    
    def getQd(self):
        return copy.copy(self.Xd)
    
    def getQdd(self):
        return copy.copy(self.Xdd)
    
    def getOmega(self):
        return copy.copy(self.omega)
    
    def getOmegad(self):
        return copy.copy(self.omegad)
    
    def setQ(self, new_Q):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this QuaternionDMPState is invalid!"
        assert (new_Q.shape[0] == 4), "new_Q.shape[0]=" + str(new_Q.shape[0]) + " is NOT 4 (invalid Quaternion)!"
        self.X = copy.copy(new_Q)
        self.normalizeQuaternion()
        assert (self.isValid()), "Post-condition(s) checking is failed: this QuaternionDMPState became invalid!"
        return None
    
    def setQd(self, new_Qd):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this QuaternionDMPState is invalid!"
        assert (new_Qd.shape[0] == 4), "new_Qd.shape[0]=" + str(new_Qd.shape[0]) + " is NOT 4!"
        self.Xd = copy.copy(new_Qd)
        assert (self.isValid()), "Post-condition(s) checking is failed: this QuaternionDMPState became invalid!"
        return None
    
    def setQdd(self, new_Qdd):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this QuaternionDMPState is invalid!"
        assert (new_Qdd.shape[0] == 4), "new_Qdd.shape[0]=" + str(new_Qdd.shape[0]) + " is NOT 4!"
        self.Xdd = copy.copy(new_Qdd)
        assert (self.isValid()), "Post-condition(s) checking is failed: this QuaternionDMPState became invalid!"
        return None
    
    def setOmega(self, new_omega):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this QuaternionDMPState is invalid!"
        assert (new_omega.shape[0] == self.dmp_num_dimensions), "new_omega.shape[0]=" + str(new_omega.shape[0]) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
        self.omega = copy.copy(new_omega)
        assert (self.isValid()), "Post-condition(s) checking is failed: this QuaternionDMPState became invalid!"
        return None
    
    def setOmegad(self, new_omegad):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this QuaternionDMPState is invalid!"
        assert (new_omegad.shape[0] == self.dmp_num_dimensions), "new_omegad.shape[0]=" + str(new_omegad.shape[0]) + " is mis-matched with self.dmp_num_dimensions=" + str(self.dmp_num_dimensions) + "!"
        self.omegad = copy.copy(new_omegad)
        assert (self.isValid()), "Post-condition(s) checking is failed: this QuaternionDMPState became invalid!"
        return None
    
    def setX(self, new_X):
        return self.setQ(new_X)
    
    def setXd(self, new_Xd):
        return self.setQd(new_Xd)
    
    def setXdd(self, new_Xdd):
        return self.setQdd(new_Xdd)
    
    def normalizeQuaternion(self):
        traj_length = self.getLength()
        self.X = util_quat.normalizeQuaternion(self.X.T).reshape(traj_length, 4).T
    
    def computeQdAndQdd(self):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this QuaternionDMPState is invalid!"
        traj_length = self.getLength()
        self.normalizeQuaternion()
        [new_QdT, new_QddT] = util_quat.computeQDotAndQDoubleDotTrajectory( self.X.T, self.omega.T, self.omegad.T )
        self.Xd = new_QdT.reshape(traj_length, 4).T
        self.Xdd = new_QddT.reshape(traj_length, 4).T
        assert (self.isValid()), "Post-condition(s) checking is failed: this QuaternionDMPState became invalid!"
        return None