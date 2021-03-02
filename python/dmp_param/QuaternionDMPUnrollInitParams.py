#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Created on Mon Dec 31 18:00:00 2018

@author: gsutanto
"""

import numpy as np
import os
import sys
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), "../dmp_state/"))
from QuaternionDMPState import *
from QuaternionDMPTrajectory import *
from TauSystem import *


class QuaternionDMPUnrollInitParams:
  "Class for QuaternionDMP Unroll Initialization Parameters."

  def __init__(self,
               critical_quatdmptraj_init,
               tau_init=None,
               robot_task_servo_rate=None,
               is_zeroing_out_velocity_and_acceleration=True):
    traj_size = critical_quatdmptraj_init.getLength()
    assert (traj_size >= 2)
    start_quatdmpstate = critical_quatdmptraj_init.getQuaternionDMPStateAtIndex(
        0)
    goal_quatdmpstate = critical_quatdmptraj_init.getQuaternionDMPStateAtIndex(
        traj_size - 1)
    if (tau_init == None):
      if (robot_task_servo_rate == None):
        self.tau = goal_quatdmpstate.getTime()[
            0, 0] - start_quatdmpstate.getTime()[0, 0]
      else:
        self.tau = (1.0 * (traj_size - 1)) / robot_task_servo_rate
    else:
      self.tau = tau_init
    assert (self.tau >= MIN_TAU), "QuaternionDMPUnrollInitParams.tau=" + str(
        self.tau) + " < MIN_TAU"
    critical_quatdmpstates_list = []
    critical_quatdmpstates_list.append(start_quatdmpstate)
    if (traj_size >= 3):
      before_goal_quatdmpstate = critical_quatdmptraj_init.getQuaternionDMPStateAtIndex(
          traj_size - 2)
      critical_quatdmpstates_list.append(before_goal_quatdmpstate)
    critical_quatdmpstates_list.append(goal_quatdmpstate)
    processed_critical_states_list = []
    for critical_quatdmpstate in critical_quatdmpstates_list:
      if (is_zeroing_out_velocity_and_acceleration):
        processed_critical_states_list.append(
            QuaternionDMPState(critical_quatdmpstate.getQ()))
      else:
        processed_critical_states_list.append(copy.copy(critical_quatdmpstate))
    self.critical_states = convertQuaternionDMPStatesListIntoQuaternionDMPTrajectory(
        processed_critical_states_list)

  def isValid(self):
    assert (self.tau >= MIN_TAU), "QuaternionDMPUnrollInitParams.tau=" + str(
        self.tau) + " < MIN_TAU"
    assert (self.critical_states.getLength() >= 2)
    assert (self.critical_states.isValid())
    return True


def getQuaternionDMPUnrollInitParams(
    start_quatdmpstate,
    goal_quatdmpstate,
    tau_init=None,
    robot_task_servo_rate=None,
    is_zeroing_out_velocity_and_acceleration=True):
  critical_quatdmptraj_init = convertQuaternionDMPStatesListIntoQuaternionDMPTrajectory(
      [start_quatdmpstate, goal_quatdmpstate])
  return QuaternionDMPUnrollInitParams(
      critical_quatdmptraj_init, tau_init, robot_task_servo_rate,
      is_zeroing_out_velocity_and_acceleration)
