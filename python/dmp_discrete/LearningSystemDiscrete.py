#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import numpy as np
from scipy import signal
import os
import sys
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities/'))
from LearningSystem import *
from TransformSystemDiscrete import *
from FuncApproximatorDiscrete import *
from utilities import *


class LearningSystemDiscrete(LearningSystem, object):
  """Class for learning systems of discrete DMPs.

       Implemented free of (or abstracted away from)
       the type of state (DMPState/QuaternionDMPState/etc.).
       The function getTargetForcingTermTraj() of transform_sys is the one
       who shall take care of the particular state type being used
       in its implementation underneath.
  """

  def __init__(self, transformation_system_discrete, name=''):
    super(LearningSystemDiscrete, self).__init__(transformation_system_discrete,
                                                 name)

  def isValid(self):
    assert (super(LearningSystemDiscrete, self).isValid())
    return True

  def learnApproximator(self, list_dmptrajectory_demo_local,
                        robot_task_servo_rate):
    assert (self.isValid())
    assert (robot_task_servo_rate > 0.0)

    N_traj = len(list_dmptrajectory_demo_local)
    assert (N_traj > 0)

    list_Ft = [None] * N_traj
    list_cX = [None] * N_traj
    list_cV = [None] * N_traj
    list_tau = [None] * N_traj
    list_A_learn = [None] * N_traj
    list_G = [None] * N_traj
    list_PSI = [None] * N_traj
    for i in range(N_traj):
      dmptrajectory_demo_local = list_dmptrajectory_demo_local[i]
      Ft_inst, cX_inst, cV_inst, tau_inst, tau_relative_inst, A_learn_inst, G_inst = self.transform_sys.getTargetForcingTermTraj(
          dmptrajectory_demo_local, robot_task_servo_rate)
      list_Ft[i] = Ft_inst
      list_cX[i] = cX_inst
      list_cV[i] = cV_inst
      list_tau[i] = tau_inst
      list_A_learn[i] = A_learn_inst
      list_G[i] = G_inst
      PSI_inst = self.transform_sys.func_approx.getBasisFunctionTensor(cX_inst)
      list_PSI[i] = PSI_inst
    mean_tau = np.mean(list_tau)
    mean_A_learn = np.mean(list_A_learn, axis=0)
    Ft = np.hstack(list_Ft)
    cX = np.hstack(list_cX)
    cV = np.hstack(list_cV)
    G = np.hstack(list_G)
    PSI = np.hstack(list_PSI)

    if (self.transform_sys.canonical_sys.order == 2):
      MULT = cV
    elif (self.transform_sys.canonical_sys.order == 1):
      MULT = cX
    sx2 = np.sum(
        np.matmul(
            np.ones((self.transform_sys.func_approx.model_size, 1)),
            np.square(MULT)) * PSI,
        axis=1)
    list_w = [None] * self.transform_sys.dmp_num_dimensions
    for i in range(self.transform_sys.dmp_num_dimensions):
      sxtd = np.sum(
          np.matmul(
              np.ones((self.transform_sys.func_approx.model_size, 1)),
              (MULT * Ft[[i], :])) * PSI,
          axis=1)
      w_dim = sxtd * 1.0 / (sx2 + 1.e-10)
      list_w[i] = w_dim.T
    W = np.vstack(list_w)
    assert (np.isnan(W).any() == False), 'Learned W contains NaN!'
    self.transform_sys.func_approx.weights = W
    self.transform_sys.A_learn = mean_A_learn
    Fp = np.matmul(W, PSI) * np.matmul(
        np.ones((self.transform_sys.dmp_num_dimensions, 1)),
        (MULT * 1.0 / np.sum(PSI, axis=0).reshape(
            (1, PSI.shape[1]))))  # predicted forcing term

    N_filter_order = 2  # Butterworth filter order
    fc = 10.0  # cutoff frequency (in Hz)
    fs = robot_task_servo_rate  # sampling frequency (in Hz)
    Wn = fc / (fs / 2)
    [b, a] = signal.butter(N_filter_order, Wn)
    Ft_filtered = signal.filtfilt(b, a, Ft, axis=1)
    nmse_fit = computeNMSE(Fp.T, Ft_filtered.T)
    print('NMSE of forcing term fitting = ' + str(nmse_fit))

    return W, mean_A_learn, mean_tau, Ft, Fp, G, cX, cV, PSI
