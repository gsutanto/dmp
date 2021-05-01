#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Created on Mon Dec 24  21:00:00 2018

@author: gsutanto
"""

import re
import warnings as wa
import numpy as np
import numpy.linalg as npla
import numpy.matlib as npma
import os
import sys
import copy
import pyplot_util
from utilities import *

division_epsilon = 1.0e-100


def normalizeQuaternion(Q_input, warning_threshold=0.98):
  assert (
      (len(Q_input.shape) >= 1) and
      (len(Q_input.shape) <= 2)), "Q_input has invalid number of dimensions!"
  if (len(Q_input.shape) == 1):
    Q_input = Q_input.reshape(1, 4)
  assert (
      Q_input.shape[1] == 4), "Each row of Q_input has to be 4-dimensional!!!"
  tensor_length = Q_input.shape[0]
  Q_input_norm = npla.norm(Q_input, ord=2, axis=1).reshape(tensor_length, 1)
  if ((Q_input_norm < warning_threshold).any()):
    wa.warn("(Q_input_norm < %f).any() == True ; Q_input_norm=\n" %
            warning_threshold + str(Q_input_norm))
  # Normalize (make sure that norm(Quaternion) == 1)
  Q_output = Q_input / npma.repmat(Q_input_norm, 1, 4)
  if (tensor_length == 1):
    Q_output = Q_output[0, :]
  return Q_output


def standardizeNormalizeQuaternion(Q_input):
  assert (
      (len(Q_input.shape) >= 1) and
      (len(Q_input.shape) <= 2)), "Q_input has invalid number of dimensions!"
  if (len(Q_input.shape) == 1):
    Q_input = Q_input.reshape(1, 4)
  assert (
      Q_input.shape[1] == 4), "Each row of Q_input has to be 4-dimensional!!!"

  Q_output = copy.deepcopy(Q_input)

  # Standardize (make sure that unique Quaternion represents
  # unique orientation)
  Q_idx_tobe_std = np.where(Q_output[:, 0] < 0.0)[0]
  if (len(Q_idx_tobe_std) > 0):
    #        print('Standardizing some Quaternions for uniqueness ...');
    Q_output[Q_idx_tobe_std, :] = -Q_output[Q_idx_tobe_std, :]

  Q_output = normalizeQuaternion(Q_output)
  return Q_output


def computeQuaternionLogMap(Q_input, div_epsilon=division_epsilon):
  assert (
      (len(Q_input.shape) >= 1) and
      (len(Q_input.shape) <= 2)), "Q_input has invalid number of dimensions!"
  if (len(Q_input.shape) == 1):
    Q_input = Q_input.reshape(1, 4)
  assert (
      Q_input.shape[1] == 4), "Each row of Q_input has to be 4-dimensional!!!"
  assert (np.iscomplex(Q_input).any() == False)

  tensor_length = Q_input.shape[0]

  # normalize the input Quaternion first:
  Q_prep = normalizeQuaternion(np.real(Q_input)).reshape(tensor_length, 4)

  u = Q_prep[:, 0].reshape(tensor_length, 1)
  q = Q_prep[:, 1:4]

  arccos_u = np.arccos(u)
  sin_arccos_u = np.sin(arccos_u)

  #    arccos_u_div_sin_arccos_u = (arccos_u + div_epsilon)/(sin_arccos_u + div_epsilon)
  #
  #    log_Q_output = npma.repmat(arccos_u_div_sin_arccos_u, 1, 3) * q

  multiplier_sign = np.ones((tensor_length, 1))
  log_multiplier = np.log(np.zeros((tensor_length, 1)) + division_epsilon)

  Q_idx_w_positive_sin_arccos_u = np.where(sin_arccos_u[:, 0] > 0)[0]
  Q_idx_w_negative_sin_arccos_u = np.where(sin_arccos_u[:, 0] < 0)[0]
  Q_idx_w_nonzero_sin_arccos_u = np.union1d(Q_idx_w_positive_sin_arccos_u,
                                            Q_idx_w_negative_sin_arccos_u)

  log_Q_output = copy.deepcopy(q)
  if (Q_idx_w_nonzero_sin_arccos_u.size > 0):
    log_Q_output[Q_idx_w_nonzero_sin_arccos_u, :] = np.zeros(
        (len(Q_idx_w_nonzero_sin_arccos_u), 3))
    log_multiplier[Q_idx_w_nonzero_sin_arccos_u,
                   0] = np.log(arccos_u[Q_idx_w_nonzero_sin_arccos_u, 0])

  if (Q_idx_w_positive_sin_arccos_u.size > 0):
    log_multiplier[
        Q_idx_w_positive_sin_arccos_u,
        0] = log_multiplier[Q_idx_w_positive_sin_arccos_u, 0] - np.log(
            sin_arccos_u[Q_idx_w_positive_sin_arccos_u, 0])

  if (Q_idx_w_negative_sin_arccos_u.size > 0):
    multiplier_sign[Q_idx_w_negative_sin_arccos_u,
                    0] = -multiplier_sign[Q_idx_w_negative_sin_arccos_u, 0]
    log_multiplier[Q_idx_w_negative_sin_arccos_u, 0] = log_multiplier[
        Q_idx_w_negative_sin_arccos_u,
        0] - np.log(-sin_arccos_u[Q_idx_w_negative_sin_arccos_u, 0])

  for i in range(3):
    q_ith_col_idx_gt_zero = np.where(q[:, i] > 0)[0]
    q_ith_col_idx_gt_zero_intersect_Q_idx_w_nonzero_sin_arccos_u = np.intersect1d(
        q_ith_col_idx_gt_zero, Q_idx_w_nonzero_sin_arccos_u)
    if (q_ith_col_idx_gt_zero_intersect_Q_idx_w_nonzero_sin_arccos_u.size > 0):
      log_Q_output[
          q_ith_col_idx_gt_zero_intersect_Q_idx_w_nonzero_sin_arccos_u, i] = (
              multiplier_sign[
                  q_ith_col_idx_gt_zero_intersect_Q_idx_w_nonzero_sin_arccos_u,
                  0] *
              np.exp(log_multiplier[
                  q_ith_col_idx_gt_zero_intersect_Q_idx_w_nonzero_sin_arccos_u,
                  0] + np.log(q[
                      q_ith_col_idx_gt_zero_intersect_Q_idx_w_nonzero_sin_arccos_u,
                      i])))

    q_ith_col_idx_lt_zero = np.where(q[:, i] < 0)[0]
    q_ith_col_idx_lt_zero_intersect_Q_idx_w_nonzero_sin_arccos_u = np.intersect1d(
        q_ith_col_idx_lt_zero, Q_idx_w_nonzero_sin_arccos_u)
    if (q_ith_col_idx_lt_zero_intersect_Q_idx_w_nonzero_sin_arccos_u.size > 0):
      log_Q_output[
          q_ith_col_idx_lt_zero_intersect_Q_idx_w_nonzero_sin_arccos_u,
          i] = (-multiplier_sign[
              q_ith_col_idx_lt_zero_intersect_Q_idx_w_nonzero_sin_arccos_u,
              0] * np.exp(log_multiplier[
                  q_ith_col_idx_lt_zero_intersect_Q_idx_w_nonzero_sin_arccos_u,
                  0] + np.log(-q[
                      q_ith_col_idx_lt_zero_intersect_Q_idx_w_nonzero_sin_arccos_u,
                      i])))

  assert (np.isnan(log_Q_output).any() == False), "log_Q_output contains NaN!"
  if (tensor_length == 1):
    log_Q_output = log_Q_output[0, :]
  assert (np.iscomplex(log_Q_output).any() == False)
  return (2.0 * np.real(log_Q_output))


def computeQuaternionExpMap(log_Q_input, div_epsilon=division_epsilon):
  assert ((len(log_Q_input.shape) >= 1) and (len(log_Q_input.shape) <= 2)
         ), "log_Q_input has invalid number of dimensions!"
  if (len(log_Q_input.shape) == 1):
    log_Q_input = log_Q_input.reshape(1, 3)
  assert (log_Q_input.shape[1] == 3
         ), "Each row of log_Q_input has to be 3-dimensional!!!"
  assert (np.iscomplex(log_Q_input).any() == False)

  tensor_length = log_Q_input.shape[0]

  r = np.real(log_Q_input) / 2.0
  norm_r = npla.norm(r, ord=2, axis=1).reshape(tensor_length, 1)
  cos_norm_r = np.cos(norm_r)
  sin_norm_r = np.sin(norm_r)

  #    sin_norm_r_div_norm_r = (sin_norm_r + div_epsilon)/(norm_r + div_epsilon)
  #
  #    Q_output = np.hstack([cos_norm_r, (npma.repmat(sin_norm_r_div_norm_r, 1, 3) * r)])

  Q_output = np.zeros((tensor_length, 4))
  Q_output[:, 0] = np.ones(tensor_length)

  log_Q_input_idx_nonzero_norm_r = np.where(norm_r[:, 0] != 0)[0]

  log_Q_input_idx_sin_norm_r_gt_zero = np.where(sin_norm_r[:, 0] > 0)[0]
  log_Q_input_idx_sin_norm_r_gt_zero_intersect_log_Q_input_idx_nonzero_norm_r = np.intersect1d(
      log_Q_input_idx_sin_norm_r_gt_zero, log_Q_input_idx_nonzero_norm_r)
  if (log_Q_input_idx_sin_norm_r_gt_zero_intersect_log_Q_input_idx_nonzero_norm_r
      .size > 0):
    Q_output[
        log_Q_input_idx_sin_norm_r_gt_zero_intersect_log_Q_input_idx_nonzero_norm_r,
        0] = cos_norm_r[
            log_Q_input_idx_sin_norm_r_gt_zero_intersect_log_Q_input_idx_nonzero_norm_r,
            0]
    Q_output[
        log_Q_input_idx_sin_norm_r_gt_zero_intersect_log_Q_input_idx_nonzero_norm_r,
        1:4] = (
            npma.repmat(
                np.exp(
                    np.log(sin_norm_r[
                        log_Q_input_idx_sin_norm_r_gt_zero_intersect_log_Q_input_idx_nonzero_norm_r,
                        0]) -
                    np.log(norm_r[
                        log_Q_input_idx_sin_norm_r_gt_zero_intersect_log_Q_input_idx_nonzero_norm_r,
                        0]))
                .reshape(
                    log_Q_input_idx_sin_norm_r_gt_zero_intersect_log_Q_input_idx_nonzero_norm_r
                    .shape[0], 1), 1, 3) *
            r[log_Q_input_idx_sin_norm_r_gt_zero_intersect_log_Q_input_idx_nonzero_norm_r, :]
        )

  log_Q_input_idx_sin_norm_r_lt_zero = np.where(sin_norm_r[:, 0] < 0)[0]
  log_Q_input_idx_sin_norm_r_lt_zero_intersect_log_Q_input_idx_nonzero_norm_r = np.intersect1d(
      log_Q_input_idx_sin_norm_r_lt_zero, log_Q_input_idx_nonzero_norm_r)
  if (log_Q_input_idx_sin_norm_r_lt_zero_intersect_log_Q_input_idx_nonzero_norm_r
      .size > 0):
    Q_output[
        log_Q_input_idx_sin_norm_r_lt_zero_intersect_log_Q_input_idx_nonzero_norm_r,
        0] = cos_norm_r[
            log_Q_input_idx_sin_norm_r_lt_zero_intersect_log_Q_input_idx_nonzero_norm_r,
            0]
    Q_output[
        log_Q_input_idx_sin_norm_r_lt_zero_intersect_log_Q_input_idx_nonzero_norm_r,
        1:4] = (
            npma.repmat(
                np.exp(
                    np.log(-sin_norm_r[
                        log_Q_input_idx_sin_norm_r_lt_zero_intersect_log_Q_input_idx_nonzero_norm_r,
                        0]) -
                    np.log(norm_r[
                        log_Q_input_idx_sin_norm_r_lt_zero_intersect_log_Q_input_idx_nonzero_norm_r,
                        0]))
                .reshape(
                    log_Q_input_idx_sin_norm_r_lt_zero_intersect_log_Q_input_idx_nonzero_norm_r
                    .shape[0], 1), 1, 3) *
            (-r[log_Q_input_idx_sin_norm_r_lt_zero_intersect_log_Q_input_idx_nonzero_norm_r, :]
            ))

  assert (np.isnan(Q_output).any() == False), "Q_output contains NaN!"
  # don't forget to normalize the resulting Quaternion:
  Q_output = normalizeQuaternion(Q_output)
  assert (np.iscomplex(Q_output).any() == False)
  return np.real(Q_output)


def computeQuatConjugate(Q_input):
  assert (
      (len(Q_input.shape) >= 1) and
      (len(Q_input.shape) <= 2)), "Q_input has invalid number of dimensions!"
  if (len(Q_input.shape) == 1):
    Q_input = Q_input.reshape(1, 4)
  assert (
      Q_input.shape[1] == 4), "Each row of Q_input has to be 4-dimensional!!!"

  Q_output = copy.deepcopy(Q_input)
  Q_output[:, 1:4] = -Q_output[:, 1:4]

  # don't forget to normalize the resulting Quaternion:
  Q_output = normalizeQuaternion(Q_output)
  return Q_output


def computeQuatProduct(Qp, Qq):
  assert ((len(Qp.shape) >= 1) and
          (len(Qp.shape) <= 2)), "Qp has invalid number of dimensions!"
  assert ((len(Qq.shape) >= 1) and
          (len(Qq.shape) <= 2)), "Qq has invalid number of dimensions!"
  if (len(Qp.shape) == 1):
    Qp = Qp.reshape(1, 4)
  if (len(Qq.shape) == 1):
    Qq = Qq.reshape(1, 4)
  assert (Qp.shape[0] == Qq.shape[0]), "Qp and Qq length are NOT equal!!!"
  assert (np.iscomplex(Qp).any() == False)
  Qp = np.real(Qp)
  assert (np.iscomplex(Qq).any() == False)
  Qq = np.real(Qq)
  tensor_length = Qp.shape[0]

  p0 = Qp[:, 0]
  p1 = Qp[:, 1]
  p2 = Qp[:, 2]
  p3 = Qp[:, 3]

  P = np.zeros((tensor_length, 4, 4))

  P[:, 0, 0] = p0
  P[:, 1, 0] = p1
  P[:, 2, 0] = p2
  P[:, 3, 0] = p3

  P[:, 0, 1] = -p1
  P[:, 1, 1] = p0
  P[:, 2, 1] = p3
  P[:, 3, 1] = -p2

  P[:, 0, 2] = -p2
  P[:, 1, 2] = -p3
  P[:, 2, 2] = p0
  P[:, 3, 2] = p1

  P[:, 0, 3] = -p3
  P[:, 1, 3] = p2
  P[:, 2, 3] = -p1
  P[:, 3, 3] = p0

  Qr = np.matmul(P, Qq.reshape(tensor_length, 4, 1)).reshape(tensor_length, 4)
  if (tensor_length == 1):
    Qr = Qr[0, :]
  assert (np.iscomplex(Qr).any() == False)
  return np.real(Qr)


def computeLogQuatDifference(Qp, Qq, is_standardizing_quat_diff=True):
  if (not is_standardizing_quat_diff
     ):  # if NOT standardizing before applying Log Mapping
    omegar = computeQuaternionLogMap(
        computeQuatProduct(normalizeQuaternion(Qp), computeQuatConjugate(Qq)))
  else:  # if (is_standardizing_quat_diff): # if standardizing before applying Log Mapping
    omegar = computeQuaternionLogMap(
        standardizeNormalizeQuaternion(
            computeQuatProduct(
                normalizeQuaternion(Qp), computeQuatConjugate(Qq))))
  return omegar


def computeOmegaAndOmegaDotTrajectory(QT, QdT, QddT):
  """Extracting/converting omega and omegad (trajectories)

       from trajectories of Q, Qd, and Qdd.
  """
  assert ((len(QT.shape) >= 1) and
          (len(QT.shape) <= 2)), "QT has invalid number of dimensions!"
  assert ((len(QdT.shape) >= 1) and
          (len(QdT.shape) <= 2)), "QdT has invalid number of dimensions!"
  assert ((len(QddT.shape) >= 1) and
          (len(QddT.shape) <= 2)), "QddT has invalid number of dimensions!"
  if (len(QT.shape) == 1):
    QT = QT.reshape(1, 4)
  if (len(QdT.shape) == 1):
    QdT = QdT.reshape(1, 4)
  if (len(QddT.shape) == 1):
    QddT = QddT.reshape(1, 4)
  assert ((QT.shape[0] == QdT.shape[0]) and
          (QdT.shape[0]
           == QddT.shape[0])), "QT, QdT, and QddT length are NOT equal!!!"
  tensor_length = QT.shape[0]

  QT_conj = computeQuatConjugate(QT).reshape(tensor_length, 4)
  omegaQT = 2.0 * computeQuatProduct(QdT, QT_conj).reshape(tensor_length, 4)
  omegadQT = 2.0 * computeQuatProduct(
      (QddT - computeQuatProduct(QdT, computeQuatProduct(QT_conj, QdT))),
      QT_conj).reshape(tensor_length, 4)

  # some anomaly-checking:
  if (npla.norm(omegaQT[:, 0], ord=2) > 0):
    print("WARNING: npla.norm(omegaQT[:,0], ord=2)  = %f > 0" %
          npla.norm(omegaQT[:, 0], ord=2))
    print("         np.max(np.fabs(omegaQT[:,0]))   = %f" %
          np.max(np.fabs(omegaQT[:, 0])))
  if (npla.norm(omegadQT[:, 0], ord=2) > 0):
    print("WARNING: npla.norm(omegadQT[:,0], ord=2) = %f > 0" %
          npla.norm(omegadQT[:, 0], ord=2))
    print("         np.max(np.fabs(omegadQT[:,0]))  = %f" %
          np.max(np.fabs(omegadQT[:, 0])))

  omegaT = omegaQT[:, 1:4]
  omegadT = omegadQT[:, 1:4]
  if (tensor_length == 1):
    omegaT = omegaT[0, :]
    omegadT = omegadT[0, :]
  return omegaT, omegadT


def computeOmegaTrajectory(QT, dt):
  """Given Quaternion trajectory and dt,

       compute the angular velocity (omega) trajectory.
  """
  assert (len(QT.shape) == 2), "QT has invalid number of dimensions!"
  assert (QT.shape[0] >= 2), "QT must have at least two Quaternion data points!"
  assert (dt > 0.0), "dt (sampling time) is invalid!"
  tensor_length = QT.shape[0]

  QtT = copy.deepcopy(QT)
  QtT[-1, :] = copy.deepcopy(QT[-2, :])
  Qt_plus_1T = copy.deepcopy(QT)
  Qt_plus_1T[:-1, :] = copy.deepcopy(QT[1:, :])
  omegaT = (1.0 / dt) * computeLogQuatDifference(Qt_plus_1T, QtT)

  return omegaT


def computeQDotTrajectory(QT, omegaT):
  QdT = 0.5 * computeQuatProduct(
      np.hstack([np.zeros((QT.shape[0], 1)), omegaT]), QT)
  return QdT


def computeQDoubleDotTrajectory(QT, QdT, omegaT, omegadT):
  QddT = 0.5 * (
      computeQuatProduct(np.hstack([np.zeros((QT.shape[0], 1)), omegadT]), QT) +
      computeQuatProduct(np.hstack([np.zeros((QdT.shape[0], 1)), omegaT]), QdT))
  return QddT


def computeQDotAndQDoubleDotTrajectory(QT, omegaT, omegadT):
  """Extracting/converting Qd and Qdd (trajectories)

       from trajectories of Q, omega, and omegad.
  """
  tensor_length = QT.shape[0]

  QdT = computeQDotTrajectory(QT, omegaT)
  QddT = computeQDoubleDotTrajectory(QT, QdT.reshape(tensor_length, 4), omegaT,
                                     omegadT)
  return QdT, QddT


def integrateQuat(Qt, omega_t, dt, tau=1.0):
  assert (dt > 0.0), "dt (sampling time) is invalid!"
  assert (tau > 0.0), "tau (time constant) is invalid!"

  theta_v = omega_t * (dt / tau)
  Q_incr = computeQuaternionExpMap(theta_v)
  Qt_plus_1 = normalizeQuaternion(
      computeQuatProduct(Q_incr, normalizeQuaternion(Qt)))
  return Qt_plus_1


def inverseIntegrateQuat(Qt_plus_1, omega_t, dt, tau=1.0):
  assert (dt > 0.0), "dt (sampling time) is invalid!"

  theta_v = omega_t * (dt / tau)
  Q_incr = computeQuaternionExpMap(theta_v)
  Q_decr = computeQuatConjugate(Q_incr)
  Qt = normalizeQuaternion(
      computeQuatProduct(Q_decr, normalizeQuaternion(Qt_plus_1)))
  return Qt


def isQuatArrayHasMajorityNegativeRealParts(Qs):
  assert (Qs.shape[1] == 4), "Qs is NOT a valid array of Quaternions!"
  tensor_length = Qs.shape[0]

  count_Quat_w_negative_real_parts = (Qs[:, 0] < 0.0).sum()
  result = (count_Quat_w_negative_real_parts > (tensor_length / 2.0))
  return result


def computeAverageQuaternions(Qs):
  tensor_length = Qs.shape[0]
  Qs = normalizeQuaternion(Qs).reshape(tensor_length, 4)

  QsTQs = np.matmul(Qs.T, Qs)
  [d, V] = npla.eig(QsTQs)
  max_eig_val_idx = np.argmax(d)
  if (isQuatArrayHasMajorityNegativeRealParts(Qs)):
    mean_Q = -standardizeNormalizeQuaternion(V[:, max_eig_val_idx])
  else:
    mean_Q = standardizeNormalizeQuaternion(V[:, max_eig_val_idx])
  return mean_Q


def preprocessQuaternionSignal(QtT, omega_tT, dt):
  Qt_plus_1T = integrateQuat(QtT, omega_tT, dt)
  QtT_prediction = copy.deepcopy(QtT)
  QtT_prediction[1:, :] = Qt_plus_1T[0:-1, :]
  sign_mismatch_per_Q_dim = ((QtT_prediction / (QtT + division_epsilon)) < 0)
  sign_mismatch_summary = np.logical_and(
      np.logical_and(sign_mismatch_per_Q_dim[:, 0], sign_mismatch_per_Q_dim[:,
                                                                            1]),
      np.logical_and(sign_mismatch_per_Q_dim[:, 2], sign_mismatch_per_Q_dim[:,
                                                                            3]))
  if (sign_mismatch_summary.any() == False):
    return QtT
  else:
    sign_mismatch_idx = list(np.where(sign_mismatch_summary == True)[0])
    if ((len(sign_mismatch_idx) % 2) == 1):
      sign_mismatch_idx.append(QtT.shape[0])
    sign_mismatch_idx_pairs = list(chunks(sign_mismatch_idx, 2))
    sign_corrector = np.ones(QtT.shape)
    for sign_mismatch_idx_pair in sign_mismatch_idx_pairs:
      sign_corrector[
          range(sign_mismatch_idx_pair[0], sign_mismatch_idx_pair[1]), :] *= -1
    QtT_sign_corrected = sign_corrector * QtT
    return QtT_sign_corrected
