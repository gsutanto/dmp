#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30  9:30:00 2017

@author: gsutanto
"""

import re
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

def compareTwoNumericFiles(file_1_path, file_2_path, scalar_max_abs_diff_threshold=1.001e-5, scalar_max_rel_abs_diff_threshold=1.501e-3):
    file_1 = np.loadtxt(file_1_path)
    file_2 = np.loadtxt(file_2_path)
    assert (file_1.shape == file_2.shape), 'File dimension mis-match!'
    
    file_diff = file_1 - file_2
    abs_diff = np.abs(file_diff)
    rowvec_max_abs_diff = np.max(abs_diff,axis=0)
    rowvec_max_idx_abs_diff = np.argmax(abs_diff,axis=0)
    scalar_max_abs_diff = np.max(rowvec_max_abs_diff)
    scalar_max_abs_diff_col = np.argmax(rowvec_max_abs_diff)
    scalar_max_abs_diff_row = rowvec_max_idx_abs_diff[scalar_max_abs_diff_col]
    
    if (scalar_max_abs_diff > scalar_max_abs_diff_threshold):
        print ('Comparing:')
        print (file_1_path)
        print ('and')
        print (file_2_path)
        assert (False), ('Two files are NOT similar: scalar_max_abs_diff=' + str(scalar_max_abs_diff) + 
                         ' is beyond threshold at [row,col]=[' + str(scalar_max_abs_diff_row) + ',' + str(scalar_max_abs_diff_col) + '], i.e. ' + 
                         str(file_1[scalar_max_abs_diff_row, scalar_max_abs_diff_col]) + ' vs ' + 
                         str(file_2[scalar_max_abs_diff_row, scalar_max_abs_diff_col]) + ' !')
    return None

def naturalSort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)