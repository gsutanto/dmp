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


def smoothStartEnd1DPositionProfile(oneD_position_prof, percentage_padding,
                                    percentage_smoothing_points, mode, b, a):
  assert ((percentage_padding >= 0.0) and (percentage_padding <= 100.0))
  assert ((percentage_smoothing_points >= 0.0) and
          (percentage_smoothing_points <= 100.0))
  traj_length = oneD_position_prof.shape[0]
  is_originally_a_vector = False
  if (len(oneD_position_prof.shape) == 1):
    oneD_position_prof = oneD_position_prof.reshape(traj_length, 1)
    is_originally_a_vector = True

  num_padding = int(round((percentage_padding / 100.0) * traj_length))
  if (num_padding <= 2):
    num_padding = 3  # minimum number of padding

  num_smoothing_points = int(
      round((percentage_smoothing_points / 100.0) * traj_length))
  if (num_smoothing_points <= (num_padding + 2)):
    num_smoothing_points = num_padding + 3  # minimum number of smoothing points

  smoothed_1D_position_prof = oneD_position_prof
  if ((mode >= 1) and (mode <= 3)):
    assert (num_padding > 2), 'num_padding must be greater than 2!'
    assert (num_smoothing_points >
            (num_padding +
             2)), '# of smoothing points must be greater than (num_padding+2)!'
    assert (len(smoothed_1D_position_prof.shape) == 2
           ), 'Input tensor must be 2-dimensional'
    assert (min(smoothed_1D_position_prof.shape) == 1
           ), 'Input matrix must be 1-dimensional, i.e. a vector!'

  # mode == 1: smooth start only
  # mode == 2: smooth end only
  # mode == 3: smooth both start and end
  # otherwise: no smoothing

  if ((mode == 1) or (mode == 3)):
    smoothed_1D_position_prof[1:num_padding, :] = smoothed_1D_position_prof[
        0, :]
    smoothed_1D_position_prof_idx = (
        list(range(0, num_padding)) +
        list(range(num_smoothing_points, traj_length)))
    interp_position_prof_idx = range(num_padding, num_smoothing_points + 1)

    smoothed_1D_position_prof[interp_position_prof_idx, :] = interp1d(
        smoothed_1D_position_prof_idx,
        smoothed_1D_position_prof[smoothed_1D_position_prof_idx, :],
        kind='linear',
        axis=0)(
            interp_position_prof_idx)

  if ((mode == 2) or (mode == 3)):
    smoothed_1D_position_prof[traj_length - num_padding:traj_length -
                              1, :] = smoothed_1D_position_prof[traj_length -
                                                                1, :]
    smoothed_1D_position_prof_idx = (
        list(range(0, traj_length - num_smoothing_points)) +
        list(range(traj_length - num_padding, traj_length)))
    interp_position_prof_idx = range(traj_length - num_smoothing_points,
                                     traj_length - num_padding)

    smoothed_1D_position_prof[interp_position_prof_idx, :] = interp1d(
        smoothed_1D_position_prof_idx,
        smoothed_1D_position_prof[smoothed_1D_position_prof_idx, :],
        kind='linear',
        axis=0)(
            interp_position_prof_idx)

  # apply low-pass filter for smoothness:
  smoothed_1D_position_prof = signal.filtfilt(
      b,
      a,
      smoothed_1D_position_prof,
      axis=0,
      padlen=3 * (max(len(a), len(b)) - 1)
  )  # padlen here is adapted to follow what MATLAB's filtfilt() does (for code synchronization)

  if (is_originally_a_vector):
    smoothed_1D_position_prof = smoothed_1D_position_prof.reshape(traj_length,)

  return smoothed_1D_position_prof


def smoothStartEndNDPositionProfile(ND_position_prof, percentage_padding,
                                    percentage_smoothing_points, mode, b, a):
  smoothed_ND_position_prof = np.zeros(ND_position_prof.shape)
  D = ND_position_prof.shape[1]
  for d in range(D):
    smoothed_ND_position_prof[:, d] = smoothStartEnd1DPositionProfile(
        ND_position_prof[:, d], percentage_padding, percentage_smoothing_points,
        mode, b, a)
  return smoothed_ND_position_prof


def smoothStartEndNDTrajectoryBasedOnPosition(
    ND_traj,
    percentage_padding,
    percentage_smoothing_points,
    mode,
    dt,
    fc=40.0  # cutoff frequency
):
  N_filter_order = 2
  fs = 1.0 / dt  # sampling frequency
  Wn = fc / (fs / 2)
  b, a = signal.butter(N_filter_order, Wn)

  ND_position_prof = ND_traj.X.T
  D = ND_position_prof.shape[1]
  smoothed_position_prof = smoothStartEndNDPositionProfile(
      ND_position_prof, percentage_padding, percentage_smoothing_points, mode,
      b, a)
  smoothed_velocity_prof = diffnc(smoothed_position_prof, dt)
  smoothed_acceleration_prof = diffnc(smoothed_velocity_prof, dt)

  smoothed_ND_traj = DMPTrajectory(smoothed_position_prof.T,
                                   smoothed_velocity_prof.T,
                                   smoothed_acceleration_prof.T, ND_traj.time)
  return smoothed_ND_traj


def smoothStartEndQuatTrajectoryBasedOnQuaternion(
    Quat_traj,
    percentage_padding,
    percentage_smoothing_points,
    mode,
    dt,
    fc=40.0,  # cutoff frequency
    is_plotting_smoothing_comparison=False,
    is_plotting_Qsignal_preprocessing_comparison=False):
  N_filter_order = 2
  fs = 1.0 / dt  # sampling frequency
  Wn = fc / (fs / 2)
  b, a = signal.butter(N_filter_order, Wn)

  # GSutanto remarks: the following commented line has an issue,
  # in particular when suddenly there is a flip sign on the Quaternion signal
  # (remember that Quaternion Q and Quaternion -Q both represent
  #  the same orientation). Although there is a sign flipping,
  # the Quaternion signal is still valid, but it will be mistakenly
  # considered as a discontinuity, which by the low-pass filter will be
  # smoothened out (this smoothening out will make things even worse & WRONG).
  # Please note that we do low-pass filtering on the log(Q) signal
  # (not on the Q signal directly), however,
  # even though Q and -Q represent the same orientation, log(Q) and log(-Q)
  # is usually are not of the same (vector) value, which is why doing
  # low-pass filtering on log(Q) signal which have a sign flipping on
  # the Q signal is BAD and WRONG:
  #    Quat_prof = Quat_traj.X.T
  # That's why we do Quaternion signal preprocessing here, to make sure
  # the Q signal does NOT have a sign flipping:
  Quat_prof = util_quat.preprocessQuaternionSignal(Quat_traj.X.T,
                                                   Quat_traj.omega.T, dt)
  log_Quat_prof = util_quat.computeQuaternionLogMap(Quat_prof)

  if (is_plotting_Qsignal_preprocessing_comparison):
    all_QT_list = [None] * 2
    all_QT_list[0] = Quat_traj.X.T
    all_QT_list[1] = Quat_prof
    pyplot_util.subplot_ND(
        NDtraj_list=all_QT_list,
        title='Quaternion',
        Y_label_list=['Q%d' % Q_dim for Q_dim in range(4)],
        fig_num=100 + 0,
        label_list=['original', 'preprocessed'],
        color_style_list=[['b', '-'], ['g', '-']],
        is_auto_line_coloring_and_styling=False)

  smoothed_log_Quat_prof = smoothStartEndNDPositionProfile(
      log_Quat_prof, percentage_padding, percentage_smoothing_points, mode, b,
      a)
  smoothed_Quat_prof = util_quat.computeQuaternionExpMap(smoothed_log_Quat_prof)

  traj_length = smoothed_Quat_prof.shape[0]

  omega_prof = util_quat.computeOmegaTrajectory(smoothed_Quat_prof, dt)

  smoothed_omega_prof = signal.filtfilt(
      b, a, omega_prof, axis=0, padlen=3 * (max(len(a), len(b)) - 1)
  )  # padlen here is adapted to follow what MATLAB's filtfilt() does (for code synchronization)
  smoothed_omegad_prof = diffnc(smoothed_omega_prof, dt)

  smoothed_Quat_traj = QuaternionDMPTrajectory(
      Q_init=smoothed_Quat_prof.T,
      Qd_init=None,
      Qdd_init=None,
      omega_init=smoothed_omega_prof.T,
      omegad_init=smoothed_omegad_prof.T,
      time_init=Quat_traj.time)
  smoothed_Quat_traj.computeQdAndQdd()

  if (is_plotting_smoothing_comparison):
    pyplot_util.subplot_ND(
        NDtraj_list=[Quat_prof, smoothed_Quat_prof],
        title='Quaternion',
        Y_label_list=['w', 'x', 'y', 'z'],
        fig_num=0,
        label_list=['original', 'smoothed'],
        is_auto_line_coloring_and_styling=True)

    pyplot_util.subplot_ND(
        NDtraj_list=[log_Quat_prof, smoothed_log_Quat_prof],
        title='log(Quat)',
        Y_label_list=['x', 'y', 'z'],
        fig_num=1,
        label_list=['original', 'smoothed'],
        is_auto_line_coloring_and_styling=True)

    pyplot_util.subplot_ND(
        NDtraj_list=[omega_prof, smoothed_omega_prof],
        title='Angular Velocity (Omega)',
        Y_label_list=['x', 'y', 'z'],
        fig_num=2,
        label_list=['original', 'smoothed'],
        is_auto_line_coloring_and_styling=True)

    pyplot_util.subplot_ND(
        NDtraj_list=[Quat_traj.omegad.T, smoothed_omegad_prof],
        title='Angular Acceleration (Omegadot)',
        Y_label_list=['x', 'y', 'z'],
        fig_num=3,
        label_list=['original', 'smoothed'],
        is_auto_line_coloring_and_styling=True)

  return smoothed_Quat_traj
