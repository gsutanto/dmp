#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Created on Sat Dec 29 21:00:00 2018

@author: gsutanto
"""

import numpy as np
import os
import sys
import copy
from QuaternionDMPState import *


class QuaternionDMPTrajectory(QuaternionDMPState, object):
  "Class for QuaternionDMP trajectories."

  def isValid(self):
    assert (super(QuaternionDMPTrajectory, self).isValid())
    assert (self.X.shape[1] == self.time.shape[1])
    return True

  def getDMPStateAtIndex(self, i):
    return self.getQuaternionDMPStateAtIndex(i)

  def getQuaternionDMPStateAtIndex(self, i):
    assert (self.isValid()), (
        "Pre-condition(s) checking is failed: this QuaternionDMPTrajectory is "
        "invalid!")
    assert ((i >= 0) and (i < self.time.shape[1])
           ), "Index i=" + str(i) + " is out-of-range (TrajectoryLength=" + str(
               self.time.shape[1]) + ")!"
    return QuaternionDMPState(
        self.X[:, i].reshape(4, 1), self.Xd[:, i].reshape(4, 1),
        self.Xdd[:, i].reshape(4, 1),
        self.omega[:, i].reshape(self.dmp_num_dimensions, 1),
        self.omegad[:, i].reshape(self.dmp_num_dimensions,
                                  1), self.time[:, i].reshape(1, 1))

  def setDMPStateAtIndex(self, i, quatdmpstate):
    return self.setQuaternionDMPStateAtIndex(i, quatdmpstate)

  def setQuaternionDMPStateAtIndex(self, i, quatdmpstate):
    assert (self.isValid()), (
        "Pre-condition(s) checking is failed: this QuaternionDMPTrajectory is "
        "invalid!")
    assert ((i >= 0) and (i < self.time.shape[1])
           ), "Index i=" + str(i) + " is out-of-range (TrajectoryLength=" + str(
               self.time.shape[1]) + ")!"
    assert (quatdmpstate.isValid())
    assert (self.dmp_num_dimensions == quatdmpstate.dmp_num_dimensions)
    self.X[:, i] = quatdmpstate.getX().reshape(4,)
    self.Xd[:, i] = quatdmpstate.getXd().reshape(4,)
    self.Xdd[:, i] = quatdmpstate.getXdd().reshape(4,)
    self.omega[:, i] = quatdmpstate.getOmega().reshape(
        quatdmpstate.dmp_num_dimensions,)
    self.omegad[:, i] = quatdmpstate.getOmegad().reshape(
        quatdmpstate.dmp_num_dimensions,)
    self.time[:, i] = quatdmpstate.getTime().reshape(1,)

    assert (self.isValid()), (
        "Post-condition(s) checking is failed: this QuaternionDMPTrajectory "
        "became invalid!")
    return None

  def accessQuaternionDMPQAtIndex(
      self, i):  # for quick (pointer-like) access, NO copying
    assert (self.isValid()), (
        "Pre-condition(s) checking is failed: this QuaternionDMPTrajectory is "
        "invalid!")
    assert ((i >= 0) and (i < self.time.shape[1])
           ), "Index i=" + str(i) + " is out-of-range (TrajectoryLength=" + str(
               self.time.shape[1]) + ")!"
    return self.X[:, i].reshape(4, 1)

  def accessQuaternionDMPQdAtIndex(
      self, i):  # for quick (pointer-like) access, NO copying
    assert (self.isValid()), (
        "Pre-condition(s) checking is failed: this QuaternionDMPTrajectory is "
        "invalid!")
    assert ((i >= 0) and (i < self.time.shape[1])
           ), "Index i=" + str(i) + " is out-of-range (TrajectoryLength=" + str(
               self.time.shape[1]) + ")!"
    return self.Xd[:, i].reshape(4, 1)

  def accessQuaternionDMPQddAtIndex(
      self, i):  # for quick (pointer-like) access, NO copying
    assert (self.isValid()), (
        "Pre-condition(s) checking is failed: this QuaternionDMPTrajectory is "
        "invalid!")
    assert ((i >= 0) and (i < self.time.shape[1])
           ), "Index i=" + str(i) + " is out-of-range (TrajectoryLength=" + str(
               self.time.shape[1]) + ")!"
    return self.Xdd[:, i].reshape(4, 1)

  def accessQuaternionDMPOmegaAtIndex(
      self, i):  # for quick (pointer-like) access, NO copying
    assert (self.isValid()), (
        "Pre-condition(s) checking is failed: this QuaternionDMPTrajectory is "
        "invalid!")
    assert ((i >= 0) and (i < self.time.shape[1])
           ), "Index i=" + str(i) + " is out-of-range (TrajectoryLength=" + str(
               self.time.shape[1]) + ")!"
    return self.omega[:, i].reshape(self.dmp_num_dimensions, 1)

  def accessQuaternionDMPOmegadAtIndex(
      self, i):  # for quick (pointer-like) access, NO copying
    assert (self.isValid()), (
        "Pre-condition(s) checking is failed: this QuaternionDMPTrajectory is "
        "invalid!")
    assert ((i >= 0) and (i < self.time.shape[1])
           ), "Index i=" + str(i) + " is out-of-range (TrajectoryLength=" + str(
               self.time.shape[1]) + ")!"
    return self.omegad[:, i].reshape(self.dmp_num_dimensions, 1)

  def accessQuaternionDMPTimeAtIndex(
      self, i):  # for quick (pointer-like) access, NO copying
    assert (self.isValid()), (
        "Pre-condition(s) checking is failed: this QuaternionDMPTrajectory is "
        "invalid!")
    assert ((i >= 0) and (i < self.time.shape[1])
           ), "Index i=" + str(i) + " is out-of-range (TrajectoryLength=" + str(
               self.time.shape[1]) + ")!"
    return self.time[:, i].reshape(1, 1)


def convertQuaternionDMPStatesListIntoQuaternionDMPTrajectory(
    quatdmpstates_list):
  quatdmpstates_list_size = len(quatdmpstates_list)
  Q_list = [None] * quatdmpstates_list_size
  Qd_list = [None] * quatdmpstates_list_size
  Qdd_list = [None] * quatdmpstates_list_size
  omega_list = [None] * quatdmpstates_list_size
  omegad_list = [None] * quatdmpstates_list_size
  time_list = [None] * quatdmpstates_list_size
  for i in range(quatdmpstates_list_size):
    Q_list[i] = quatdmpstates_list[i].X
    Qd_list[i] = quatdmpstates_list[i].Xd
    Qdd_list[i] = quatdmpstates_list[i].Xdd
    omega_list[i] = quatdmpstates_list[i].omega
    omegad_list[i] = quatdmpstates_list[i].omegad
    time_list[i] = quatdmpstates_list[i].time
  return QuaternionDMPTrajectory(
      np.concatenate(Q_list, axis=1), np.concatenate(Qd_list, axis=1),
      np.concatenate(Qdd_list, axis=1), np.concatenate(omega_list, axis=1),
      np.concatenate(omegad_list, axis=1), np.concatenate(time_list, axis=1))
