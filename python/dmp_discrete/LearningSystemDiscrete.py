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
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities/'))
from LearningSystem import *
from TransformSystemDiscrete import *
from FuncApproximatorDiscrete import *
from utilities import *

class LearningSystemDiscrete(LearningSystem, object):
    'Class for learning systems of discrete DMPs.'
    
    def __init__(self, transformation_system_discrete, name=""):
        super(LearningSystemDiscrete, self).__init__(transformation_system_discrete, name)
    
    def isValid(self):
        assert (super(LearningSystemDiscrete, self).isValid())
        return True
    
    def learnApproximator(self, list_dmptrajectory_demo_local, robot_task_servo_rate):
        assert (self.isValid())
        assert (robot_task_servo_rate > 0.0)
        
        N_traj = len(list_dmptrajectory_demo_local)
        assert (N_traj > 0)
        
        list_Ft = []
        list_cX = []
        list_cV = []
        list_tau = []
        list_A_learn = []
        list_PSI = []
        for dmptrajectory_demo_local in list_dmptrajectory_demo_local:
            Ft_inst, cX_inst, cV_inst, tau_inst, A_learn_inst = getTargetForcingTermTraj(self, dmptrajectory_demo_local, robot_task_servo_rate)
            list_Ft.append(Ft_inst)
            list_cX.append(cX_inst)
            list_cV.append(cV_inst)
            list_tau.append(tau_inst)
            list_A_learn.append(A_learn_inst)
            PSI_inst = getBasisFunctionTensor(cX_inst)
            list_PSI.append(PSI_inst)
        mean_tau = np.mean(list_tau)
        mean_A_learn = np.mean(list_A_learn, axis=0)
        Ft = np.hstack(list_Ft)
        cX = np.hstack(list_cX)
        cV = np.hstack(list_cV)
        PSI = np.hstack(list_PSI)
        
        if (self.transform_sys.canonical_sys.order == 2):
            MULT = cV
        elif (self.transform_sys.canonical_sys.order == 1):
            MULT = cX
        sx2 = np.sum(np.matmul(np.ones(self.transform_sys.func_approx.model_size,1), np.square(MULT)) * PSI, axis=1)
        list_w = []
        for i in range(self.transform_sys.dmp_num_dimensions):
            sxtd = np.sum(np.matmul(np.ones(self.transform_sys.func_approx.model_size,1), (MULT * Ft[[i],:])) * PSI, axis=1)
            w_dim = sxtd * 1.0 / (sx2+1.e-10)
            list_w.append(w_dim.T)
        W = np.vstack(list_w)
        assert (np.isnan(W).any()), "Learned W contains NaN!"
        self.transform_sys.func_approx.weights = W
        self.transform_sys.A_learn = mean_A_learn
        Fp = np.matmul(W, PSI) * np.matmul(np.ones((1, self.transform_sys.dmp_num_dimensions)), (MULT * 1.0 / np.sum(PSI, axis=0).reshape((1,PSI.shape[1])))) # predicted forcing term
        NMSE = computeNMSE(Fp.T, Ft.T)
        print ('NMSE = '+ str(NMSE))
        
        return W, mean_A_learn, mean_tau