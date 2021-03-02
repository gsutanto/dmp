#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import numpy as np
import os
import sys
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), "../dmp_goal_system/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../dmp_param/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../dmp_state/"))
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../dmp_coupling/base/"))
from TauSystem import *
from CanonicalSystem import *
from FunctionApproximator import *
from TransformCoupling import *
from DMPState import *
from GoalSystem import *

MIN_FABS_AMPLITUDE = 1.e-3


class TransformationSystem:
  "Base class for transformation systems of DMPs."

  def __init__(self,
               dmp_num_dimensions_init,
               canonical_system,
               function_approximator,
               start_dmpstate=None,
               current_dmpstate=None,
               current_velocity_dmpstate=None,
               goal_system=None,
               transform_couplers_list=[],
               name=""):
    self.name = name
    self.dmp_num_dimensions = dmp_num_dimensions_init
    self.canonical_sys = canonical_system
    self.tau_sys = self.canonical_sys.tau_sys
    self.func_approx = function_approximator
    self.transform_couplers_list = transform_couplers_list
    self.is_using_coupling_term_at_dimension = [True] * self.dmp_num_dimensions
    self.is_started = False
    if (start_dmpstate != None):
      self.start_state = start_dmpstate
    else:
      self.start_state = DMPState(np.zeros((self.dmp_num_dimensions, 1)))
    if (current_dmpstate != None):
      self.current_state = current_dmpstate
    else:
      self.current_state = DMPState(np.zeros((self.dmp_num_dimensions, 1)))
    if (current_velocity_dmpstate != None):
      self.current_velocity_state = current_velocity_dmpstate
    else:
      self.current_velocity_state = DMPState(
          np.zeros((self.dmp_num_dimensions, 1)))
    if (goal_system != None):
      self.goal_sys = goal_system
    else:
      self.goal_sys = GoalSystem(self.dmp_num_dimensions, self.tau_sys)

  def isValid(self):
    assert (self.dmp_num_dimensions > 0), "self.dmp_num_dimensions=" + str(
        self.dmp_num_dimensions) + " <= 0 (invalid!)"
    assert (len(
        self.is_using_coupling_term_at_dimension) == self.dmp_num_dimensions
           ), "len(self.is_using_coupling_term_at_dimension)=" + str(
               len(self.is_using_coupling_term_at_dimension)
           ) + " is mis-matched with self.dmp_num_dimensions=" + str(
               self.dmp_num_dimensions) + "!"
    assert (self.tau_sys != None)
    assert (self.canonical_sys != None)
    assert (self.func_approx != None)
    assert (self.goal_sys != None)
    assert (self.tau_sys.isValid())
    assert (self.canonical_sys.isValid())
    assert (self.func_approx.isValid())
    assert (self.goal_sys.isValid())
    assert (self.tau_sys == self.canonical_sys.tau_sys)
    assert (self.tau_sys == self.goal_sys.tau_sys)
    assert (self.start_state != None)
    assert (self.current_state != None)
    assert (self.current_velocity_state != None)
    assert (self.start_state.isValid())
    assert (self.current_state.isValid())
    assert (self.current_velocity_state.isValid())
    assert (self.func_approx.dmp_num_dimensions == self.dmp_num_dimensions
           ), "self.func_approx.dmp_num_dimensions=" + str(
               self.func_approx.dmp_num_dimensions
           ) + " is mis-matched with self.dmp_num_dimensions=" + str(
               self.dmp_num_dimensions) + "!"
    assert (self.goal_sys.dmp_num_dimensions == self.dmp_num_dimensions
           ), "self.goal_sys.dmp_num_dimensions=" + str(
               self.goal_sys.dmp_num_dimensions
           ) + " is mis-matched with self.dmp_num_dimensions=" + str(
               self.dmp_num_dimensions) + "!"
    assert (self.start_state.dmp_num_dimensions == self.dmp_num_dimensions
           ), "self.start_state.dmp_num_dimensions=" + str(
               self.start_state.dmp_num_dimensions
           ) + " is mis-matched with self.dmp_num_dimensions=" + str(
               self.dmp_num_dimensions) + "!"
    assert (self.current_state.dmp_num_dimensions == self.dmp_num_dimensions
           ), "self.current_state.dmp_num_dimensions=" + str(
               self.current_state.dmp_num_dimensions
           ) + " is mis-matched with self.dmp_num_dimensions=" + str(
               self.dmp_num_dimensions) + "!"
    assert (self.current_velocity_state.dmp_num_dimensions ==
            self.dmp_num_dimensions
           ), "self.current_velocity_state.dmp_num_dimensions=" + str(
               self.current_velocity_state.dmp_num_dimensions
           ) + " is mis-matched with self.dmp_num_dimensions=" + str(
               self.dmp_num_dimensions) + "!"
    return True

  def resetCouplingTerm(self):
    for transform_coupler_idx in range(len(self.transform_couplers_list)):
      if (self.transform_couplers_list[transform_coupler_idx] is not None):
        ret = self.transform_couplers_list[transform_coupler_idx].reset()
        if (ret == False):
          return False
    return True

  def getCouplingTerm(self):
    accumulated_ct_acc = np.zeros((self.dmp_num_dimensions, 1))
    accumulated_ct_vel = np.zeros((self.dmp_num_dimensions, 1))
    for transform_coupler_idx in range(len(self.transform_couplers_list)):
      if (self.transform_couplers_list[transform_coupler_idx] is not None):
        ct_acc, ct_vel = self.transform_couplers_list[
            transform_coupler_idx].getValue()
        assert (np.isnan(ct_acc).all() == False
               ), "ct_acc[" + str(transform_coupler_idx) + "] is NaN!"
        assert (np.isnan(ct_vel).all() == False
               ), "ct_vel[" + str(transform_coupler_idx) + "] is NaN!"
        accumulated_ct_acc = accumulated_ct_acc + ct_acc
        accumulated_ct_vel = accumulated_ct_vel + ct_vel
    assert (accumulated_ct_acc.shape == (self.dmp_num_dimensions, 1))
    assert (accumulated_ct_vel.shape == (self.dmp_num_dimensions, 1))
    return accumulated_ct_acc, accumulated_ct_vel

  def getStartState(self):
    return copy.copy(self.start_state)

  def setStartState(self, new_start_state):
    assert (self.isValid()), (
        "Pre-condition(s) checking is failed: this TransformationSystem is "
        "invalid!")
    assert (new_start_state.isValid())
    assert (
        self.start_state.dmp_num_dimensions == new_start_state
        .dmp_num_dimensions), "self.start_state.dmp_num_dimensions=" + str(
            self.start_state.dmp_num_dimensions
        ) + " is mis-matched with new_start_state.dmp_num_dimensions=" + str(
            new_start_state.dmp_num_dimensions) + "!"
    self.start_state = copy.copy(new_start_state)
    assert (self.isValid()), (
        "Post-condition(s) checking is failed: this TransformationSystem became"
        " invalid!")
    return None

  def getCurrentState(self):
    return copy.copy(self.current_state)

  def setCurrentState(self, new_current_state):
    assert (self.isValid()), (
        "Pre-condition(s) checking is failed: this TransformationSystem is "
        "invalid!")
    assert (new_current_state.isValid())
    assert (
        self.current_state.dmp_num_dimensions == new_current_state
        .dmp_num_dimensions), "self.current_state.dmp_num_dimensions=" + str(
            self.current_state.dmp_num_dimensions
        ) + " is mis-matched with new_current_state.dmp_num_dimensions=" + str(
            new_current_state.dmp_num_dimensions) + "!"
    self.current_state = copy.copy(new_current_state)
    self.updateCurrentVelocityStateFromCurrentState()
    assert (self.isValid()), (
        "Post-condition(s) checking is failed: this TransformationSystem became"
        " invalid!")
    return None

  def updateCurrentVelocityStateFromCurrentState(self):
    assert (self.isValid()), (
        "Pre-condition(s) checking is failed: this TransformationSystem is "
        "invalid!")

    tau = self.tau_sys.getTauRelative()
    Xd = self.current_state.getXd()
    Xdd = self.current_state.getXdd()
    V = tau * Xd
    Vd = tau * Xdd
    self.current_velocity_state.setX(V)
    self.current_velocity_state.setXd(Vd)

    assert (self.isValid()), (
        "Post-condition(s) checking is failed: this TransformationSystem became"
        " invalid!")
    return None

  def getCurrentGoalState(self):
    return self.goal_sys.getCurrentGoalState()

  def setCurrentGoalState(self, new_current_goal_state):
    return self.goal_sys.setCurrentGoalState(new_current_goal_state)

  def updateCurrentGoalState(self, dt):
    return self.goal_sys.updateCurrentGoalState(dt)

  def getSteadyStateGoalPosition(self):
    return self.goal_sys.getSteadyStateGoalPosition()

  def setSteadyStateGoalPosition(self, new_G):
    return self.goal_sys.setSteadyStateGoalPosition(new_G)

  def setCouplingTermUsagePerDimensions(
      self, is_using_coupling_term_at_dimension_init):
    assert (
        len(is_using_coupling_term_at_dimension_init) == self.dmp_num_dimensions
    ), "len(is_using_coupling_term_at_dimension_init)=" + str(
        len(is_using_coupling_term_at_dimension_init)
    ) + " is mis-matched with self.dmp_num_dimensions=" + str(
        self.dmp_num_dimensions) + "!"
    self.is_using_coupling_term_at_dimension = is_using_coupling_term_at_dimension_init
    return None
