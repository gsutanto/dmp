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
from DMPState import *
from DMPTrajectory import *
from QuaternionDMPState import *
from QuaternionDMPTrajectory import *
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities/'))
from utilities import *
import utility_quaternion as util_quat
import pyplot_util

def smoothStartEnd1DPositionProfile(oneD_position_prof, 
                                    percentage_padding, 
                                    percentage_smoothing_points, 
                                    mode, 
                                    b, a):
    assert ((percentage_padding >= 0.0) and (percentage_padding <= 100.0))
    assert ((percentage_smoothing_points >= 0.0) and (percentage_smoothing_points <= 100.0))
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

def smoothStartEndNDPositionProfile(ND_position_prof, 
                                    percentage_padding, 
                                    percentage_smoothing_points, 
                                    mode, 
                                    b, a):
    smoothed_ND_position_prof = np.zeros(ND_position_prof.shape)
    D = ND_position_prof.shape[1]
    for d in range(D):
        smoothed_ND_position_prof[:,d] = smoothStartEnd1DPositionProfile( ND_position_prof[:,d], 
                                                                          percentage_padding, 
                                                                          percentage_smoothing_points, 
                                                                          mode, 
                                                                          b, a )
    return smoothed_ND_position_prof

def smoothStartEndNDTrajectoryBasedOnPosition(ND_traj, 
                                              percentage_padding, 
                                              percentage_smoothing_points, 
                                              mode, dt, 
                                              fc=40.0 # cutoff frequency
                                              ):
    N_filter_order = 2
    fs = 1.0/dt # sampling frequency
    Wn = fc/(fs/2)
    b, a = signal.butter(N_filter_order, Wn)
    
    ND_position_prof = ND_traj.X.T
    D = ND_position_prof.shape[1]
    smoothed_position_prof = smoothStartEndNDPositionProfile(ND_position_prof, 
                                                             percentage_padding, 
                                                             percentage_smoothing_points, 
                                                             mode, 
                                                             b, a)
    smoothed_velocity_prof = diffnc(smoothed_position_prof, dt)
    smoothed_acceleration_prof = diffnc(smoothed_velocity_prof, dt)
    
    smoothed_ND_traj = DMPTrajectory(smoothed_position_prof.T, 
                                     smoothed_velocity_prof.T, 
                                     smoothed_acceleration_prof.T, 
                                     ND_traj.time)
    return smoothed_ND_traj

def smoothStartEndQuatTrajectoryBasedOnQuaternion(Quat_traj, 
                                                  percentage_padding, 
                                                  percentage_smoothing_points, 
                                                  mode, dt, 
                                                  fc=40.0, # cutoff frequency
                                                  is_plotting_smoothing_comparison=False):
    N_filter_order = 2
    fs = 1.0/dt # sampling frequency
    Wn = fc/(fs/2)
    b, a = signal.butter(N_filter_order, Wn)
    
    Quat_prof = Quat_traj.X.T
    log_Quat_prof = util_quat.computeQuaternionLogMap(Quat_prof)
    smoothed_log_Quat_prof = smoothStartEndNDPositionProfile(log_Quat_prof, 
                                                             percentage_padding, 
                                                             percentage_smoothing_points, 
                                                             mode, b, a )
    smoothed_Quat_prof = util_quat.computeQuaternionExpMap(smoothed_log_Quat_prof)
    
    traj_length = smoothed_Quat_prof.shape[0]
    
    omega_prof = util_quat.computeOmegaTrajectory(smoothed_Quat_prof, dt)
    
    smoothed_omega_prof = signal.filtfilt(b, a, omega_prof, axis=0, padlen=3*(max(len(a), len(b))-1)) # padlen here is adapted to follow what MATLAB's filtfilt() does (for code synchronization)
    smoothed_omegad_prof = diffnc(smoothed_omega_prof, dt)
    
    smoothed_Quat_traj = QuaternionDMPTrajectory(Q_init=smoothed_Quat_prof.T, Qd_init=None, Qdd_init=None, 
                                                 omega_init=smoothed_omega_prof.T, omegad_init=smoothed_omegad_prof.T, 
                                                 time_init=Quat_traj.time)
    smoothed_Quat_traj.computeQdAndQdd()
    
    if (is_plotting_smoothing_comparison):
        pyplot_util.subplot_ND(NDtraj_list=[Quat_prof, smoothed_Quat_prof], 
                               title='Quaternion', 
                               Y_label_list=['w','x','y','z'], 
                               fig_num=0, 
                               label_list=['original', 'smoothed'], 
                               is_auto_line_coloring_and_styling=True)
        
        pyplot_util.subplot_ND(NDtraj_list=[log_Quat_prof, smoothed_log_Quat_prof], 
                               title='log(Quat)', 
                               Y_label_list=['x','y','z'], 
                               fig_num=1, 
                               label_list=['original', 'smoothed'], 
                               is_auto_line_coloring_and_styling=True)
        
        pyplot_util.subplot_ND(NDtraj_list=[omega_prof, smoothed_omega_prof], 
                               title='Angular Velocity (Omega)', 
                               Y_label_list=['x','y','z'], 
                               fig_num=2, 
                               label_list=['original', 'smoothed'], 
                               is_auto_line_coloring_and_styling=True)
        
        pyplot_util.subplot_ND(NDtraj_list=[Quat_traj.omegad.T, smoothed_omegad_prof], 
                               title='Angular Acceleration (Omegadot)', 
                               Y_label_list=['x','y','z'], 
                               fig_num=3, 
                               label_list=['original', 'smoothed'], 
                               is_auto_line_coloring_and_styling=True)
    
    return smoothed_Quat_traj