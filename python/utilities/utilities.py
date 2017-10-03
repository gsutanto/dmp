#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30  9:30:00 2017

@author: gsutanto
"""

import numpy as np

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def computeNMSE(predictions, ground_truth):
    mse = np.mean(np.square(predictions - ground_truth), axis=0)    # Mean-Squared Error (MSE)
    var_ground_truth = np.var(ground_truth, axis=0)                 # Variance of the Ground-Truth
    nmse = np.divide(mse, var_ground_truth)                         # Normalized Mean-Squared Error (NMSE)
    return nmse

def computeWNMSE(predictions, ground_truth, weight):
    N_data = ground_truth.shape[0]
    N_dims = ground_truth.shape[1]
    wmse = np.mean(np.multiply(np.tile(weight, (1, N_dims)), np.square(predictions - ground_truth)), axis=0)        # Weighted Mean-Squared Error (WMSE)
    mean_gt = np.mean(ground_truth, axis=0)
    zero_mean_gt = ground_truth - np.tile(mean_gt, (N_data, 1))
    wvar_gt = (1.0/(N_data-1)) * np.sum(np.multiply(np.tile(weight, (1, N_dims)), np.square(zero_mean_gt)), axis=0)    # Weighted Variance of the Ground-Truth
    wnmse = np.divide(wmse, wvar_gt)    # Normalized Weighted Mean-Squared Error (NWMSE)
    return wnmse