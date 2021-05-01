#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Created on Fri Dec 28 21:00:00 2018

@author: gsutanto
"""

import numpy as np
import numpy.linalg as npla
import numpy.matlib as npma
import os
import sys
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), "../../dmp_base/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../dmp_discrete/"))
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../dmp_goal_system/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../dmp_param/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../dmp_state/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../utilities/"))
from TauSystem import *
from TransformSystemDiscrete import *
from FuncApproximatorDiscrete import *
from CanonicalSystemDiscrete import *
from QuaternionGoalSystem import *
from DMPState import *
from QuaternionDMPState import *
from QuaternionDMPTrajectory import *
import utility_quaternion as util_quat


class TransformSystemQuaternion(TransformSystemDiscrete, object):
  "Class for transformation systems of QuaternionDMPs."

  def __init__(self,
               canonical_system_discrete,
               func_approximator_discrete,
               is_using_scaling_init,
               transform_couplers_list=[],
               ts_alpha=25.0,
               ts_beta=25.0 / 4.0,
               name=""):
    super(TransformSystemQuaternion, self).__init__(
        dmp_num_dimensions_init=3,
        canonical_system_discrete=canonical_system_discrete,
        func_approximator_discrete=func_approximator_discrete,
        is_using_scaling_init=is_using_scaling_init,
        ts_alpha=ts_alpha,
        ts_beta=ts_beta,
        start_dmpstate_discrete=QuaternionDMPState(),
        current_dmpstate_discrete=QuaternionDMPState(),
        current_velocity_dmpstate_discrete=DMPState(X_init=np.zeros((3, 1))),
        goal_system_discrete=QuaternionGoalSystem(
            canonical_system_discrete.tau_sys),
        transform_couplers_list=[],
        name=name)

  def isValid(self):
    assert (super(TransformSystemQuaternion, self).isValid())
    assert (self.dmp_num_dimensions == 3)
    return True

  def start(self, start_quat_state_init, goal_quat_state_init):
    assert (self.isValid()), (
        "Pre-condition(s) checking is failed: this TransformSystemQuaternion is"
        " invalid!")
    assert (start_quat_state_init.isValid())
    assert (goal_quat_state_init.isValid())
    assert (start_quat_state_init.dmp_num_dimensions == self.dmp_num_dimensions)
    assert (goal_quat_state_init.dmp_num_dimensions == self.dmp_num_dimensions)

    self.start_state = start_quat_state_init
    self.setCurrentState(start_quat_state_init)

    QG_init = goal_quat_state_init.getQ()
    if (self.canonical_sys.order == 2):
      # Best option for Schaal's DMP Model using 2nd order canonical system:
      # Using goal evolution system initialized with the start position (state) as goal position (state),
      # which over time evolves toward a steady-state goal position (state).
      # The main purpose is to avoid discontinuous initial acceleration (see the math formula
      # of Schaal's DMP Model for more details).
      # Please also refer the initialization described in paper:
      # B. Nemec and A. Ude, “Action sequencing using dynamic movement
      # primitives,” Robotica, vol. 30, no. 05, pp. 837–846, 2012.
      Q0 = start_quat_state_init.getQ()
      omega0 = start_quat_state_init.getOmega()
      omegad0 = start_quat_state_init.getOmegad()

      tau = self.tau_sys.getTauRelative()
      log_quat_diff_Qg_and_Q = ((((tau * tau * omegad0) * 1.0 / self.alpha) +
                                 (tau * omega0)) / self.beta)
      quat_diff_Qg_and_Q = util_quat.computeQuaternionExpMap(
          log_quat_diff_Qg_and_Q.T).reshape(1, 4).T
      Qg0 = util_quat.computeQuatProduct(quat_diff_Qg_and_Q.T,
                                         Q0.T).reshape(1, 4).T
      current_goal_quat_state_init = QuaternionDMPState(Q_init=Qg0)
      current_goal_quat_state_init.computeQdAndQdd()
    elif (self.canonical_sys.order == 1):
      # Best option for Schaal's DMP Model using 1st order canonical system:
      # goal position is static, no evolution
      current_goal_quat_state_init = goal_quat_state_init

    self.goal_sys.start(current_goal_quat_state_init, QG_init)
    self.resetCouplingTerm()
    self.is_started = True

    assert (self.isValid()), (
        "Post-condition(s) checking is failed: this TransformSystemQuaternion "
        "became invalid!")
    return None

  def getNextState(self, dt):
    assert (self.is_started)
    assert (self.isValid()), (
        "Pre-condition(s) checking is failed: this TransformSystemQuaternion is"
        " invalid!")
    assert (dt > 0.0)

    tau = self.tau_sys.getTauRelative()
    forcing_term, basis_function_vector = self.func_approx.getForcingTerm()
    ct_acc, ct_vel = self.getCouplingTerm()
    for d in range(self.dmp_num_dimensions):
      if (self.is_using_coupling_term_at_dimension[d] == False):
        ct_acc[d, 0] = 0.0
        ct_vel[d, 0] = 0.0

    time = self.current_state.time
    Q0 = self.start_state.getQ()
    Q = self.current_state.getQ()
    omega = self.current_state.getOmega()
    omegad = self.current_state.getOmegad()

    etha = self.current_velocity_state.getX()
    ethad = self.current_velocity_state.getXd()

    QG = self.goal_sys.getSteadyStateGoalPosition()
    Qg = self.goal_sys.getCurrentGoalState().getQ()

    Q = util_quat.integrateQuat(Q.T, omega.T, dt).reshape(1, 4).T
    omega = omega + (omegad * dt)
    etha = tau * omega

    A = util_quat.computeLogQuatDifference(QG.T, Q0.T).reshape(
        1, self.dmp_num_dimensions).T

    for d in range(self.dmp_num_dimensions):
      if (self.is_using_scaling[d]):
        if (np.fabs(self.A_learn[d, 0]) < MIN_FABS_AMPLITUDE):
          A[d, 0] = 1.0
        else:
          A[d, 0] = A[d, 0] * 1.0 / self.A_learn[d, 0]
      else:
        A[d, 0] = 1.0

    log_quat_diff_Qg_and_Q = util_quat.computeLogQuatDifference(
        Qg.T, Q.T).reshape(1, self.dmp_num_dimensions).T
    ethad = ((self.alpha *
              ((self.beta * log_quat_diff_Qg_and_Q) - etha)) +
             (forcing_term * A) + ct_acc) * 1.0 / tau
    assert (np.isnan(ethad).any() == False), "ethad contains NaN!"

    omegad = ethad * 1.0 / tau

    time = time + dt

    self.current_state = QuaternionDMPState(
        Q_init=Q,
        Qd_init=None,
        Qdd_init=None,
        omega_init=omega,
        omegad_init=omegad,
        time_init=time)
    self.current_state.computeQdAndQdd()
    self.current_velocity_state = DMPState(
        X_init=etha, Xd_init=ethad, Xdd_init=None, time_init=time)
    next_state = copy.copy(self.current_state)

    assert (self.isValid()), (
        "Post-condition(s) checking is failed: this TransformSystemQuaternion "
        "became invalid!")
    return next_state, forcing_term, ct_acc, ct_vel, basis_function_vector

  def getGoalTrajAndCanonicalTrajAndTauAndALearnFromDemo(
      self,
      quatdmptrajectory_demo_local,
      robot_task_servo_rate,
      steady_state_quat_goal_position_local=None):
    assert (self.isValid()), (
        "Pre-condition(s) checking is failed: this TransformSystemQuaternion is"
        " invalid!")
    assert (quatdmptrajectory_demo_local.isValid())
    assert (quatdmptrajectory_demo_local.dmp_num_dimensions ==
            self.dmp_num_dimensions)
    assert (robot_task_servo_rate > 0.0)

    traj_length = quatdmptrajectory_demo_local.getLength()
    start_quatdmpstate_demo_local = quatdmptrajectory_demo_local.getQuaternionDMPStateAtIndex(
        0)
    if (steady_state_quat_goal_position_local is None):
      goal_steady_quatdmpstate_demo_local = quatdmptrajectory_demo_local.getQuaternionDMPStateAtIndex(
          traj_length - 1)
    else:
      goal_steady_quatdmpstate_demo_local = QuaternionDMPState(
          steady_state_quat_goal_position_local)
    QG_learn = goal_steady_quatdmpstate_demo_local.X
    Q0_learn = start_quatdmpstate_demo_local.X
    A_learn = util_quat.computeLogQuatDifference(
        QG_learn.T, Q0_learn.T).reshape(1, self.dmp_num_dimensions).T
    dt = (goal_steady_quatdmpstate_demo_local.time[0, 0] -
          start_quatdmpstate_demo_local.time[0, 0]) / (
              traj_length - 1.0)
    if (dt <= 0.0):
      dt = 1.0 / robot_task_servo_rate
    tau = goal_steady_quatdmpstate_demo_local.time[
        0, 0] - start_quatdmpstate_demo_local.time[0, 0]
    if (tau < MIN_TAU):
      tau = (1.0 * (traj_length - 1)) / robot_task_servo_rate
    self.tau_sys.setTauBase(tau)
    self.canonical_sys.start()
    self.start(start_quatdmpstate_demo_local,
               goal_steady_quatdmpstate_demo_local)
    tau_relative = self.tau_sys.getTauRelative()
    X_list = [None] * traj_length
    V_list = [None] * traj_length
    Qg_list = [None] * traj_length
    for i in range(traj_length):
      x = self.canonical_sys.getCanonicalPosition()
      v = self.canonical_sys.getCanonicalVelocity()
      Qg = self.goal_sys.getCurrentGoalState().getQ()
      X_list[i] = x
      V_list[i] = v
      Qg_list[i] = Qg

      self.canonical_sys.updateCanonicalState(dt)
      self.updateCurrentGoalState(dt)
    X = np.hstack(X_list).reshape((1, traj_length))
    V = np.hstack(V_list).reshape((1, traj_length))
    QgT = np.hstack(Qg_list)
    self.is_started = False
    self.canonical_sys.is_started = False
    self.goal_sys.is_started = False

    quat_goal_position_trajectory = QgT
    canonical_position_trajectory = X  # might be used later during learning
    canonical_velocity_trajectory = V  # might be used later during learning
    assert (self.isValid()), (
        "Post-condition(s) checking is failed: this TransformSystemQuaternion "
        "became invalid!")
    return quat_goal_position_trajectory, canonical_position_trajectory, canonical_velocity_trajectory, tau, tau_relative, A_learn

  def getTargetForcingTermTraj(self,
                               quatdmptrajectory_demo_local,
                               robot_task_servo_rate,
                               is_omega_and_omegad_provided=True):
    QgT, cX, cV, tau, tau_relative, A_learn = self.getGoalTrajAndCanonicalTrajAndTauAndALearnFromDemo(
        quatdmptrajectory_demo_local, robot_task_servo_rate)

    traj_length = quatdmptrajectory_demo_local.getLength()
    QT = quatdmptrajectory_demo_local.getQ()
    if (is_omega_and_omegad_provided):
      omegaT = quatdmptrajectory_demo_local.getOmega()
      omegadT = quatdmptrajectory_demo_local.getOmegad()
    else:
      QdT = quatdmptrajectory_demo_local.getQd()
      QddT = quatdmptrajectory_demo_local.getQdd()
      [omegaT_T, omegadT_T
      ] = util_quat.computeOmegaAndOmegaDotTrajectory(QT.T, QdT.T, QddT.T)
      omegaT = omegaT_T.reshape(traj_length, self.dmp_num_dimensions).T
      omegadT = omegadT_T.reshape(traj_length, self.dmp_num_dimensions).T
    log_quat_diff_Qg_demo_and_Q_demo = util_quat.computeLogQuatDifference(
        QgT.T, QT.T).reshape(traj_length, self.dmp_num_dimensions).T
    F_target = ((np.square(tau_relative) * omegadT) -
                (self.alpha *
                 ((self.beta * log_quat_diff_Qg_demo_and_Q_demo) -
                  (tau_relative * omegaT))))

    return F_target, cX, cV, tau, tau_relative, A_learn, QgT

  def getTargetCouplingTermTraj(self,
                                quatdmptrajectory_demo_local,
                                robot_task_servo_rate,
                                steady_state_quat_goal_position_local,
                                is_omega_and_omegad_provided=True):
    QgT, cX, cV, tau, tau_relative, A_learn = self.getGoalTrajAndCanonicalTrajAndTauAndALearnFromDemo(
        quatdmptrajectory_demo_local, robot_task_servo_rate,
        steady_state_quat_goal_position_local)

    traj_length = quatdmptrajectory_demo_local.getLength()
    QT = quatdmptrajectory_demo_local.getQ()
    if (is_omega_and_omegad_provided):
      omegaT = quatdmptrajectory_demo_local.getOmega()
      omegadT = quatdmptrajectory_demo_local.getOmegad()
    else:
      QdT = quatdmptrajectory_demo_local.getQd()
      QddT = quatdmptrajectory_demo_local.getQdd()
      [omegaT_T, omegadT_T
      ] = util_quat.computeOmegaAndOmegaDotTrajectory(QT.T, QdT.T, QddT.T)
      omegaT = omegaT_T.reshape(traj_length, self.dmp_num_dimensions).T
      omegadT = omegadT_T.reshape(traj_length, self.dmp_num_dimensions).T
    F, PSI = self.func_approx.getForcingTermTraj(cX, cV)
    log_quat_diff_Qg_demo_and_Q_demo = util_quat.computeLogQuatDifference(
        QgT.T, QT.T).reshape(traj_length, self.dmp_num_dimensions).T
    C_target = ((np.square(tau_relative) * omegadT) - F -
                (self.alpha *
                 ((self.beta * log_quat_diff_Qg_demo_and_Q_demo) -
                  (tau_relative * omegaT))))

    return C_target, F, PSI, cX, cV, tau, tau_relative, QgT

  def updateCurrentVelocityStateFromCurrentState(self):
    assert (self.isValid()), (
        "Pre-condition(s) checking is failed: this TransformationSystem is "
        "invalid!")

    tau = self.tau_sys.getTauRelative()
    omega = self.current_state.getOmega()
    omegad = self.current_state.getOmegad()
    etha = tau * omega
    ethad = tau * omegad
    self.current_velocity_state.setX(etha)
    self.current_velocity_state.setXd(ethad)

    assert (self.isValid()), (
        "Post-condition(s) checking is failed: this TransformationSystem became"
        " invalid!")
    return None
