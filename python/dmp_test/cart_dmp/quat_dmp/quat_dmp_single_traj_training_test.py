#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Created on Mon Dec 31 21:00:00 2018

@author: gsutanto
"""

import numpy as np
import os
import sys
import copy
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_base/'))
sys.path.append(
    os.path.join(os.path.dirname(__file__), '../../../dmp_discrete/'))
sys.path.append(
    os.path.join(os.path.dirname(__file__), '../../../cart_dmp/quat_dmp'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from TauSystem import *
from QuaternionDMPUnrollInitParams import *
from CanonicalSystemDiscrete import *
from QuaternionDMP import *
from utilities import *

plt.close('all')


def quat_dmp_single_traj_training_test(
    dmp_home_dir_path='../../../../',
    canonical_order=2,
    time_reproduce_max=0.0,
    tau_reproduce=0.0,
    unroll_qtraj_save_dir_path='',
    unroll_qtraj_save_filename='',
    is_smoothing_training_traj_before_learning=False,
    percentage_padding=None,
    percentage_smoothing_points=None,
    smoothing_mode=None,
    smoothing_cutoff_frequency=None,
    is_plotting=False):
  task_servo_rate = 300.0
  dt = 1.0 / task_servo_rate
  model_size = 50
  tau = MIN_TAU

  assert ((canonical_order >= 1) and (canonical_order <= 2))
  assert (time_reproduce_max >= 0.0)
  assert (tau_reproduce >= 0.0)

  tau_sys = TauSystem(dt, tau)
  canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
  quat_dmp = QuaternionDMP(model_size, canonical_sys_discr)

  #    sub_quat_dmp_training_file_path = "/data/dmp_coupling/learn_tactile_feedback/scraping_w_tool/human_baseline/prim03/07.txt"
  # this one is a more challenging case, because Quaternions -Q and Q both represent the same orientation
  # (this is to see if the low-pass filtering on such Quaternion trajectory is done right;
  #  if NOT low-pass filtering will just make things
  #  (especially the QuaternionDMP fitting) so much worse and WRONG):
  sub_quat_dmp_training_file_path = '/data/cart_dmp/quat_dmp_unscrewing/prim02/03.txt'

  set_qtraj_input = quat_dmp.extractSetTrajectories(
      dmp_home_dir_path + sub_quat_dmp_training_file_path,
      start_column_idx=10,
      time_column_idx=0)

  [critical_states_learn, W, mean_A_learn, mean_tau, Ft, Fp, QgT, cX, cV, PSI
  ] = quat_dmp.learnFromSetTrajectories(
      set_qtraj_input,
      task_servo_rate,
      is_smoothing_training_traj_before_learning=is_smoothing_training_traj_before_learning,
      percentage_padding=percentage_padding,
      percentage_smoothing_points=percentage_smoothing_points,
      smoothing_mode=smoothing_mode,
      smoothing_cutoff_frequency=smoothing_cutoff_frequency)

  ## Reproduce
  if (time_reproduce_max <= 0.0):
    time_reproduce_max = mean_tau
  if (tau_reproduce <= 0.0):
    tau_reproduce = mean_tau
  tau = tau_reproduce
  qdmp_unroll = quat_dmp.unroll(critical_states_learn, tau, time_reproduce_max,
                                dt)
  unroll_qtraj_time = qdmp_unroll.time.T - dt
  unroll_qtraj_Q = qdmp_unroll.X.T
  unroll_qtraj = np.hstack([unroll_qtraj_time, unroll_qtraj_Q])

  if (os.path.isdir(unroll_qtraj_save_dir_path)):
    np.savetxt(unroll_qtraj_save_dir_path + '/' + unroll_qtraj_save_filename,
               unroll_qtraj)

  if (is_plotting):
    quat_dmp.plotDemosVsUnroll(set_qtraj_input, qdmp_unroll)

  return unroll_qtraj, W, mean_A_learn, mean_tau, Ft, Fp, QgT, cX, cV, PSI


if __name__ == '__main__':
  unroll_qtraj, W, mean_A_learn, mean_tau, Ft, Fp, QgT, cX, cV, PSI = quat_dmp_single_traj_training_test(
      is_smoothing_training_traj_before_learning=True,
      percentage_padding=1.5,
      percentage_smoothing_points=3.0,
      smoothing_mode=3,
      smoothing_cutoff_frequency=10.0,
      is_plotting=True)
