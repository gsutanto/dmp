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


def normalizeQuaternion(Q_input, warning_threshold = 0.98):
    assert (Q_input.shape[1] == 4), "Each row of Q_input has to be 4-dimensional!!!"
    Q_input_length = Q_input.shape[0]
    Q_input_norm = npla.norm(Q_input, ord=2, axis=1).reshape(Q_input_length, 1)
    if ((Q_input_norm < warning_threshold).any()):
        wa.warn("(Q_input_norm < %f).any() == True ; Q_input_norm=\n"%warning_threshold + str(Q_input_norm))
    # Normalize (make sure that norm(Quaternion) == 1)
    Q_output = Q_input / npma.repmat(Q_input_norm, 1, 4)
    return Q_output

#def standardizeNormalizeQuaternion():