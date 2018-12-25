#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24  21:00:00 2018

@author: gsutanto
"""

import re
import numpy as np
import os
import sys
import copy


def convertVector3ToSkewSymmetricMatrix(V):
    assert (len(V.shape) == 2), 'V must be a 2D tensor!!!'
    assert (V.shape[1] == 3), 'V must have 3 columns!!!'
    
    V_length = V.shape[0]
    skew_symm_tensor = np.zeros((V_length,3,3))
    skew_symm_tensor[:,0,1] = -V[:,2]
    skew_symm_tensor[:,1,0] = V[:,2]
    skew_symm_tensor[:,0,2] = V[:,1]
    skew_symm_tensor[:,2,0] = -V[:,1]
    skew_symm_tensor[:,1,2] = -V[:,0]
    skew_symm_tensor[:,2,1] = V[:,0]
    if (V_length == 1):
        skew_symm_matrix = skew_symm_tensor[0,:,:]
        return skew_symm_matrix
    else:
        return skew_symm_tensor

#def computeHomogeneousTransformMatrix(t, Q):