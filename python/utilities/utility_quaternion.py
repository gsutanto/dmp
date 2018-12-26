#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24  21:00:00 2018

@author: gsutanto
"""

import re
import warnings as wa
import numpy as np
import numpy.linalg as npla
import numpy.matlib as npma
import os
import sys
import copy

division_epsilon = 1.0e-100


def normalizeQuaternion(Q_input, warning_threshold = 0.98):
    assert (Q_input.shape[1] == 4), "Each row of Q_input has to be 4-dimensional!!!"
    tensor_length = Q_input.shape[0]
    Q_input_norm = npla.norm(Q_input, ord=2, axis=1).reshape(tensor_length, 1)
    if ((Q_input_norm < warning_threshold).any()):
        wa.warn("(Q_input_norm < %f).any() == True ; Q_input_norm=\n"%warning_threshold + str(Q_input_norm))
    # Normalize (make sure that norm(Quaternion) == 1)
    Q_output = Q_input / npma.repmat(Q_input_norm, 1, 4)
    return Q_output

def standardizeNormalizeQuaternion(Q_input):
    assert (Q_input.shape[1] == 4), "Each row of Q_input has to be 4-dimensional!!!"
    
    Q_output = copy.deepcopy(Q_input)
    
    # Standardize (make sure that unique Quaternion represents 
    # unique orientation)
    Q_idx_tobe_std = np.where(Q_output[:,0] < 0.0)[0]
    if (len(Q_idx_tobe_std) > 0):
        print('Standardizing some Quaternions for uniqueness ...');
        Q_output[Q_idx_tobe_std,:] = -Q_output[Q_idx_tobe_std,:]
    
    Q_output = normalizeQuaternion(Q_output)
    return Q_output

def computeQuaternionLogMap(Q_input, div_epsilon=division_epsilon):
    assert (Q_input.shape[1] == 4), "Each row of Q_input has to be 4-dimensional!!!"
    
    tensor_length = Q_input.shape[0]
    
    # normalize the input Quaternion first:
    Q_prep = normalizeQuaternion(Q_input)
    
    u = Q_prep[:,0].reshape(tensor_length,1)
    q = Q_prep[:,1:4]
    
    arccos_u = np.arccos(u)
    sin_arccos_u = np.sin(arccos_u)
    
    arccos_u_div_sin_arccos_u = (arccos_u + div_epsilon)/(sin_arccos_u + div_epsilon)
    
    log_Q_output = npma.repmat(arccos_u_div_sin_arccos_u, 1, 3) * q
    return log_Q_output

def computeQuaternionExpMap(log_Q_input, div_epsilon=division_epsilon):
    assert (log_Q_input.shape[1] == 3), "Each row of log_Q_input has to be 3-dimensional!!!"
    
    tensor_length = log_Q_input.shape[0]
    
    r = log_Q_input
    norm_r = npla.norm(r, ord=2, axis=1).reshape(tensor_length, 1)
    cos_norm_r = np.cos(norm_r)
    sin_norm_r = np.sin(norm_r)
    sin_norm_r_div_norm_r = (sin_norm_r + div_epsilon)/(norm_r + div_epsilon)
    
    Q_output = np.hstack([cos_norm_r, (npma.repmat(sin_norm_r_div_norm_r, 1, 3) * r)])
    
    # don't forget to normalize the resulting Quaternion:
    Q_output = normalizeQuaternion(Q_output);
    return Q_output