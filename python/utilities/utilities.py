#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30  9:30:00 2017

@author: gsutanto
"""

import re
import numpy as np
import os
import sys
import copy
import glob
import pickle
import shutil
from scipy import signal
from scipy.interpolate import interp1d

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def computeNMSE(predictions, ground_truth, axis=0):
    mse = np.mean(np.square(predictions - ground_truth), axis=axis) # Mean-Squared Error (MSE)
    var_ground_truth = np.var(ground_truth, axis=axis)              # Variance of the Ground-Truth
    nmse = np.divide(mse, var_ground_truth)                         # Normalized Mean-Squared Error (NMSE)
    return nmse

def computeWNMSE(predictions, ground_truth, weight, axis=0):
    N_data = ground_truth.shape[0]
    N_dims = ground_truth.shape[1]
    wmse = np.mean(np.multiply(np.tile(weight, (1, N_dims)), np.square(predictions - ground_truth)), axis=axis)        # Weighted Mean-Squared Error (WMSE)
    mean_gt = np.mean(ground_truth, axis=axis)
    zero_mean_gt = ground_truth - np.tile(mean_gt, (N_data, 1))
    wvar_gt = (1.0/(N_data-1)) * np.sum(np.multiply(np.tile(weight, (1, N_dims)), np.square(zero_mean_gt)), axis=axis) # Weighted Variance of the Ground-Truth
    wnmse = np.divide(wmse, wvar_gt)    # Normalized Weighted Mean-Squared Error (NWMSE)
    return wnmse

def compareTwoNumericFiles(file_1_path, file_2_path, 
                           scalar_max_abs_diff_threshold=1.001e-5, 
                           scalar_max_rel_abs_diff_threshold=1.501e-3):
    file_1 = np.loadtxt(file_1_path)
    file_2 = np.loadtxt(file_2_path)
    return compareTwoMatrices(file_1, file_2, 
                              scalar_max_abs_diff_threshold, 
                              scalar_max_rel_abs_diff_threshold,
                              file_1_path, file_2_path)

def compareTwoMatrices(matrix1, matrix2, 
                       scalar_max_abs_diff_threshold=1.001e-5, 
                       scalar_max_rel_abs_diff_threshold=1.501e-3,
                       name1='', name2=''):
    assert (matrix1.shape == matrix2.shape), 'File dimension mis-match!'
    
    file_diff = matrix1 - matrix2
    abs_diff = np.abs(file_diff)
    rowvec_max_abs_diff = np.max(abs_diff,axis=0)
    rowvec_max_idx_abs_diff = np.argmax(abs_diff,axis=0)
    scalar_max_abs_diff = np.max(rowvec_max_abs_diff)
    scalar_max_abs_diff_col = np.argmax(rowvec_max_abs_diff)
    scalar_max_abs_diff_row = rowvec_max_idx_abs_diff[scalar_max_abs_diff_col]
    
    if (scalar_max_abs_diff > scalar_max_abs_diff_threshold):
        print ('Comparing:')
        print (name1)
        print ('and')
        print (name2)
        assert (False), ('Two files are NOT similar: scalar_max_abs_diff=' + str(scalar_max_abs_diff) + 
                         ' is beyond threshold=' + str(scalar_max_abs_diff_threshold) + ' at [row,col]=[' + str(scalar_max_abs_diff_row) + ',' + str(scalar_max_abs_diff_col) + '], i.e. ' + 
                         str(matrix1[scalar_max_abs_diff_row, scalar_max_abs_diff_col]) + ' vs ' + 
                         str(matrix2[scalar_max_abs_diff_row, scalar_max_abs_diff_col]) + ' !')
    return None

def naturalSort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def countNumericSubdirs(dir_path):
    num_subdirs_count = 1
    subdir_path = dir_path + "/" + str(num_subdirs_count) + "/"
    while (os.path.isdir(subdir_path)):
        num_subdirs_count += 1
        subdir_path = dir_path + "/" + str(num_subdirs_count) + "/"
    num_subdirs_count -= 1
    return num_subdirs_count

def saveObj(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadObj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def recreateDir(dir_path):
    if (os.path.isdir(dir_path)):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def diffnc(X, dt):
    '''
    [X] = diffc(X,dt) does non causal differentiation with time interval
    dt between data points. The returned vector (matrix) is of the same length
    as the original one
    '''
    [traj_length, D] = X.shape
    XX = np.zeros((traj_length+2, D))
    for d in range(D):
        XX[:, d] = np.convolve(X[:, d], np.array([1,0,-1])/2.0/dt)
    
    X = XX[1:traj_length+1, :]
    X[0, :] = X[1, :]
    X[traj_length-1, :] = X[traj_length-2, :]
    
    return X

def stretchTrajectory( input_trajectory, new_traj_length ):
    if (len(input_trajectory.shape) == 1):
        input_trajectory = input_trajectory.reshape(1, input_trajectory.shape[0])
    
    D = input_trajectory.shape[0]
    traj_length = input_trajectory.shape[1]
    
    stretched_trajectory = np.zeros((D, new_traj_length))
    
    for d in range(D):
        xi = np.linspace(1.0, traj_length * 1.0, num=traj_length)
        vi = input_trajectory[d,:]
        xq = np.linspace(1.0, traj_length * 1.0, num=new_traj_length)
        vq = interp1d(xi,vi,kind='cubic')(xq)
        
        stretched_trajectory[d,:] = vq
    
    if (D == 1):
        stretched_trajectory = stretched_trajectory.reshape(new_traj_length, )
    
    return stretched_trajectory