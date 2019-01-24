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

def smoothStartEnd1DPositionProfile(oneD_position_prof, 
                                    percentage_padding, 
                                    percentage_smoothing_points, 
                                    mode, 
                                    b, a):
    traj_length = oneD_position_prof.shape[0]
    is_originally_a_vector = False
    if (len(oneD_position_prof.shape) == 1):
        oneD_position_prof = oneD_position_prof.reshape(traj_length, 1)
        is_originally_a_vector = True
    
    num_padding = int(round((percentage_padding/100.0) * traj_length))
    if (num_padding <= 2):
        num_padding = 3 # minimum number of padding
    
    num_smoothing_points = int(round((percentage_smoothing_points/100.0) * traj_length))
    if (num_smoothing_points <= (num_padding+2)):
        num_smoothing_points = num_padding + 3 # minimum number of smoothing points
    
    smoothed_1D_position_prof = oneD_position_prof
    if ((mode >= 1) and (mode <= 3)):
        assert (num_padding > 2), 'num_padding must be greater than 2!'
        assert (num_smoothing_points > (num_padding+2)), '# of smoothing points must be greater than (num_padding+2)!'
        assert (len(smoothed_1D_position_prof.shape) == 2), 'Input tensor must be 2-dimensional'
        assert (min(smoothed_1D_position_prof.shape) == 1), 'Input matrix must be 1-dimensional, i.e. a vector!'
    
    # mode == 1: smooth start only
    # mode == 2: smooth end only
    # mode == 3: smooth both start and end
    # otherwise: no smoothing
    
    if ((mode == 1) or (mode == 3)):
        smoothed_1D_position_prof[1:num_padding,:] = smoothed_1D_position_prof[0,:]
        smoothed_1D_position_prof_idx = (range(0, num_padding) + range(num_smoothing_points, traj_length))
        interp_position_prof_idx = range(num_padding, num_smoothing_points+1)
        
        smoothed_1D_position_prof[interp_position_prof_idx,:] = interp1d(smoothed_1D_position_prof_idx, 
                                                                         smoothed_1D_position_prof[smoothed_1D_position_prof_idx,:], 
                                                                         kind='linear', axis=0)(interp_position_prof_idx)
    
    if ((mode == 2) or (mode == 3)):
        smoothed_1D_position_prof[traj_length-num_padding:traj_length-1,:] = smoothed_1D_position_prof[traj_length-1,:]
        smoothed_1D_position_prof_idx = (range(0, traj_length-num_smoothing_points) + range(traj_length-num_padding, traj_length))
        interp_position_prof_idx = range(traj_length-num_smoothing_points, traj_length-num_padding)
        
        smoothed_1D_position_prof[interp_position_prof_idx,:] = interp1d(smoothed_1D_position_prof_idx, 
                                                                         smoothed_1D_position_prof[smoothed_1D_position_prof_idx,:], 
                                                                         kind='linear', axis=0)(interp_position_prof_idx)
    
    # apply low-pass filter for smoothness:
    smoothed_1D_position_prof = signal.filtfilt(b, a, smoothed_1D_position_prof, axis=0, padlen=3*(max(len(a), len(b))-1)) # padlen here is adapted to follow what MATLAB's filtfilt() does (for code synchronization)
    
    if (is_originally_a_vector):
        smoothed_1D_position_prof = smoothed_1D_position_prof.reshape(traj_length,)
    
    return smoothed_1D_position_prof

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