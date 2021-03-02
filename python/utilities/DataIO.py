#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import numpy as np
import os
import sys
import copy
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities/'))
from DMPTrajectory import *
from QuaternionDMPTrajectory import *
from utilities import *


def extractSetNDTrajectories(dir_or_file_path,
                             N=-1,
                             start_column_idx=1,
                             time_column_idx=0):
  if (os.path.isdir(dir_or_file_path)):
    trajectory_files_list = naturalSort(glob.glob(dir_or_file_path + '/*.txt'))
  else:
    assert (
        dir_or_file_path.endswith('.txt')), 'Only *.txt files are supported.'
    trajectory_files_list = [dir_or_file_path]
  trajectories_list_size = len(trajectory_files_list)
  trajectories_list = [None] * trajectories_list_size
  for i in range(trajectories_list_size):
    trajectory_file = trajectory_files_list[i]
    trajectory_file_content = np.loadtxt(trajectory_file)
    if (N == -1):
      N = (trajectory_file_content.shape[1] - 1) / 3
    if (time_column_idx >= 0):
      timeT = np.transpose(trajectory_file_content[:, [time_column_idx]])
    else:
      timeT = np.zeros((trajectory_file_content.shape[0], 1))
    XT = np.transpose(
        trajectory_file_content[:, start_column_idx:start_column_idx + N])
    XdT = np.transpose(trajectory_file_content[:, start_column_idx +
                                               N:start_column_idx + (2 * N)])
    XddT = np.transpose(
        trajectory_file_content[:, start_column_idx + (2 * N):start_column_idx +
                                (3 * N)])
    trajectory = DMPTrajectory(XT, XdT, XddT, timeT)
    trajectories_list[i] = trajectory
  return trajectories_list


def extractSetCartCoordTrajectories(dir_or_file_path,
                                    start_column_idx=1,
                                    time_column_idx=0):
  return extractSetNDTrajectories(dir_or_file_path, 3, start_column_idx,
                                  time_column_idx)


def extractSetQuaternionTrajectories(dir_or_file_path,
                                     start_column_idx=1,
                                     time_column_idx=0,
                                     is_omega_and_omegad_provided=True):
  if (os.path.isdir(dir_or_file_path)):
    trajectory_files_list = naturalSort(glob.glob(dir_or_file_path + '/*.txt'))
  else:
    assert (
        dir_or_file_path.endswith('.txt')), 'Only *.txt files are supported.'
    trajectory_files_list = [dir_or_file_path]
  trajectories_list_size = len(trajectory_files_list)
  trajectories_list = [None] * trajectories_list_size
  for i in range(trajectories_list_size):
    trajectory_file = trajectory_files_list[i]
    trajectory_file_content = np.loadtxt(trajectory_file)
    N_Q = 4
    N_omega = 3
    if (time_column_idx >= 0):
      timeT = np.transpose(trajectory_file_content[:, [time_column_idx]])
    else:
      timeT = np.zeros((trajectory_file_content.shape[0], 1))
    QT = np.transpose(
        trajectory_file_content[:, start_column_idx:start_column_idx + N_Q])
    if (is_omega_and_omegad_provided):
      omegaT = np.transpose(
          trajectory_file_content[:, start_column_idx + N_Q:start_column_idx +
                                  (N_Q + N_omega)])
      omegadT = np.transpose(
          trajectory_file_content[:, start_column_idx +
                                  (N_Q + N_omega):start_column_idx +
                                  (N_Q + (2 * N_omega))])
      trajectory = QuaternionDMPTrajectory(
          Q_init=QT, omega_init=omegaT, omegad_init=omegadT, time_init=timeT)
      trajectory.computeQdAndQdd()
    else:
      QdT = np.transpose(
          trajectory_file_content[:, start_column_idx + N_Q:start_column_idx +
                                  (2 * N_Q)])
      QddT = np.transpose(
          trajectory_file_content[:, start_column_idx +
                                  (2 * N_Q):start_column_idx + (3 * N_Q)])
      trajectory = QuaternionDMPTrajectory(
          Q_init=QT, Qd_init=QdT, Qdd_init=QddT, time_init=timeT)
    trajectories_list[i] = trajectory
  return trajectories_list
