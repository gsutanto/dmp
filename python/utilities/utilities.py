#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Created on Thu Mar 30  9:30:00 2017

@author: gsutanto
"""

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
import pyplot_util as pypl_util


def chunks(l, n):
  """Yield successive n-sized chunks from l."""
  for i in range(0, len(l), n):
    yield l[i:i + n]


def computeMSEVarGTNMSE(predictions, ground_truth, axis=0):
  mse = np.mean(
      np.square(predictions - ground_truth),
      axis=axis)  # Mean-Squared Error (MSE)
  var_ground_truth = np.var(
      ground_truth, axis=axis)  # Variance of the Ground-Truth
  nmse = np.divide(mse,
                   var_ground_truth)  # Normalized Mean-Squared Error (NMSE)
  return mse, var_ground_truth, nmse


def computeNMSE(predictions, ground_truth, axis=0):
  [_, _, nmse] = computeMSEVarGTNMSE(predictions, ground_truth, axis)
  return nmse


def computeWNMSE(predictions, ground_truth, weight, axis=0):
  N_data = ground_truth.shape[0]
  N_dims = ground_truth.shape[1]
  wmse = np.mean(
      np.multiply(
          np.tile(weight, (1, N_dims)), np.square(predictions - ground_truth)),
      axis=axis)  # Weighted Mean-Squared Error (WMSE)
  mean_gt = np.mean(ground_truth, axis=axis)
  zero_mean_gt = ground_truth - np.tile(mean_gt, (N_data, 1))
  wvar_gt = (1.0 / (N_data - 1)) * np.sum(
      np.multiply(np.tile(weight, (1, N_dims)), np.square(zero_mean_gt)),
      axis=axis)  # Weighted Variance of the Ground-Truth
  wnmse = np.divide(wmse,
                    wvar_gt)  # Normalized Weighted Mean-Squared Error (NWMSE)
  return wnmse


def computeSumSquaredL2Norm(matrix, axis=None):
  return np.sum(np.square(matrix), axis=axis)


def compareTwoNumericFiles(file_1_path,
                           file_2_path,
                           scalar_max_abs_diff_threshold=1.001e-5,
                           scalar_max_rel_abs_diff_threshold=1.001e-5,
                           is_relaxed_comparison=False):
  file_1 = np.loadtxt(file_1_path)
  file_2 = np.loadtxt(file_2_path)
  return compareTwoMatrices(file_1, file_2, scalar_max_abs_diff_threshold,
                            scalar_max_rel_abs_diff_threshold, file_1_path,
                            file_2_path, is_relaxed_comparison)


def compareTwoMatrices(matrix1,
                       matrix2,
                       scalar_max_abs_diff_threshold=1.001e-5,
                       scalar_max_rel_abs_diff_threshold=1.001e-5,
                       name1="",
                       name2="",
                       is_relaxed_comparison=False):
  assert (matrix1.shape == matrix2.shape
         ), "File dimension mis-match! %s vs %s" % (str(
             matrix1.shape), str(matrix2.shape))

  file_diff = matrix1 - matrix2
  abs_diff = np.abs(file_diff)
  rowvec_max_abs_diff = np.max(abs_diff, axis=0)
  rowvec_max_idx_abs_diff = np.argmax(abs_diff, axis=0)
  scalar_max_abs_diff = np.max(rowvec_max_abs_diff)
  scalar_max_abs_diff_col = np.argmax(rowvec_max_abs_diff)
  scalar_max_abs_diff_row = rowvec_max_idx_abs_diff[scalar_max_abs_diff_col]

  if (is_relaxed_comparison == False):
    if (scalar_max_abs_diff > scalar_max_abs_diff_threshold):
      print("Comparing:")
      print(name1)
      print("and")
      print(name2)
      assert (False), (
          "Two files are NOT similar: scalar_max_abs_diff=" +
          str(scalar_max_abs_diff) + " is beyond threshold=" +
          str(scalar_max_abs_diff_threshold) + " at [row,col]=[" +
          str(1 + scalar_max_abs_diff_row) + "," +
          str(1 + scalar_max_abs_diff_col) + "], i.e. " +
          str(matrix1[scalar_max_abs_diff_row, scalar_max_abs_diff_col]) +
          " vs " +
          str(matrix2[scalar_max_abs_diff_row, scalar_max_abs_diff_col]) + " !")
  else:  # if (is_relaxed_comparison == True):
    abs_min_matrix = np.minimum(np.abs(matrix1), np.abs(matrix2))
    rel_abs_diff = abs_diff / (abs_min_matrix + 1.0e-38)
    violation_check = np.logical_and(
        (abs_diff > scalar_max_abs_diff_threshold),
        (rel_abs_diff > scalar_max_rel_abs_diff_threshold))

    if (violation_check.any()):
      print("Comparing:")
      print(name1)
      print("and")
      print(name2)
      violation_addresses = np.where(violation_check == True)
      assert (len(violation_addresses) == 2)
      first_violation_row = violation_addresses[0][0]
      first_violation_col = violation_addresses[1][0]
      assert (False), (
          "Two files are NOT similar: " + "abs_diff=" +
          str(abs_diff[first_violation_row, first_violation_col]) +
          " is beyond threshold=" + str(scalar_max_abs_diff_threshold) +
          " AND rel_abs_diff=" +
          str(rel_abs_diff[first_violation_row, first_violation_col]) +
          " is beyond threshold=" + str(scalar_max_rel_abs_diff_threshold) +
          " at [row,col]=[" + str(1 + first_violation_row) + "," +
          str(1 + first_violation_col) + "], i.e. " +
          str(matrix1[first_violation_row, first_violation_col]) + " vs " +
          str(matrix2[first_violation_row, first_violation_col]) + " !")
  return None


def naturalSort(l):
  convert = lambda text: int(text) if text.isdigit() else text.lower()
  alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
  return sorted(l, key=alphanum_key)


def countNumericSubdirs(dir_path):
  num_subdirs_count = 1
  subdir_path = dir_path + "/" + str(num_subdirs_count) + "/"
  while (os.path.isdir(subdir_path)):
    num_subdirs_count += 1
    subdir_path = dir_path + "/" + str(num_subdirs_count) + "/"
  num_subdirs_count -= 1
  return num_subdirs_count


def saveObj(obj, file_path):
  with open(file_path, "wb") as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def loadObj(file_path):
  with open(file_path, "rb") as f:
    return pickle.load(f)


def createDirIfNotExist(dir_path):
  if (not os.path.isdir(dir_path)):
    os.makedirs(dir_path)


def recreateDir(dir_path):
  if (os.path.isdir(dir_path)):
    shutil.rmtree(dir_path)
  os.makedirs(dir_path)


def diffnc(X, dt):
  """
    [X] = diffc(X,dt) does non causal differentiation with time interval
    dt between data points. The returned vector (matrix) is of the same length
    as the original one
    """
  [traj_length, D] = X.shape
  XX = np.zeros((traj_length + 2, D))
  for d in range(D):
    XX[:, d] = np.convolve(X[:, d], np.array([1, 0, -1]) / 2.0 / dt)

  X = XX[1:traj_length + 1, :]
  X[0, :] = X[1, :]
  X[traj_length - 1, :] = X[traj_length - 2, :]

  return X


def stretchTrajectory(input_trajectory, new_traj_length):
  if (len(input_trajectory.shape) == 1):
    input_trajectory = input_trajectory.reshape(1, input_trajectory.shape[0])

  D = input_trajectory.shape[0]
  traj_length = input_trajectory.shape[1]

  stretched_trajectory = np.zeros((D, new_traj_length))

  for d in range(D):
    xi = np.linspace(1.0, traj_length * 1.0, num=traj_length)
    vi = input_trajectory[d, :]
    xq = np.linspace(1.0, traj_length * 1.0, num=new_traj_length)
    vq = interp1d(xi, vi, kind="cubic")(xq)

    stretched_trajectory[d, :] = vq

  if (D == 1):
    stretched_trajectory = stretched_trajectory.reshape(new_traj_length,)

  return stretched_trajectory


def getCatkinWSPath():
  home_path = os.environ["HOME"]
  ros_pkg_paths = os.environ["ROS_PACKAGE_PATH"].split(":")
  catkin_ws_path = None
  for ros_pkg_path in ros_pkg_paths:
    if ((ros_pkg_path[:len(home_path)] == home_path) and
        (ros_pkg_path[-14:] == "/workspace/src")):
      catkin_ws_path = ros_pkg_path[:-4]
  assert (catkin_ws_path is not None)
  print("Catkin Workspace path is %s" % catkin_ws_path)
  return catkin_ws_path


def getAllCLMCDataFilePathsInDirectory(directory_path):
  return [
      directory_path + "/" + f
      for f in os.listdir(directory_path)
      if re.match(r"d+\d{5,5}$", f)
  ]  # match the regex pattern of SL data files


def waitUntilTotalCLMCDataFilesReaches(directory_path,
                                       desired_total_clmc_data_files):
  dfilepaths = getAllCLMCDataFilePathsInDirectory(directory_path)
  while (len(dfilepaths) < desired_total_clmc_data_files):
    dfilepaths = getAllCLMCDataFilePathsInDirectory(directory_path)
  return dfilepaths


def deleteAllCLMCDataFilesInDirectory(directory_path):
  dfilepaths = getAllCLMCDataFilePathsInDirectory(directory_path)
  for dfilepath in dfilepaths:
    os.remove(dfilepath)
  return None


def plotManyTrajsVsOneTraj(set_many_trajs,
                           one_traj,
                           title_suffix="",
                           fig_num_offset=0,
                           components_to_be_plotted=["X", "Xd", "Xdd"],
                           many_traj_label="many",
                           one_traj_label="one"):
  # plotting many trajectories against one trajectory (for visual comparison)
  N_many_trajs = len(set_many_trajs)

  for comp_idx in range(len(components_to_be_plotted)):
    comp = components_to_be_plotted[comp_idx]
    all_compT_list = [None] * (1 + N_many_trajs)
    if (comp == "X"):
      for n_demo in range(N_many_trajs):
        all_compT_list[n_demo] = set_many_trajs[n_demo].X.T
      all_compT_list[1 + n_demo] = one_traj.X.T
    elif (comp == "Xd"):
      for n_demo in range(N_many_trajs):
        all_compT_list[n_demo] = set_many_trajs[n_demo].Xd.T
      all_compT_list[1 + n_demo] = one_traj.Xd.T
    elif (comp == "Xdd"):
      for n_demo in range(N_many_trajs):
        all_compT_list[n_demo] = set_many_trajs[n_demo].Xdd.T
      all_compT_list[1 + n_demo] = one_traj.Xdd.T
    elif (comp == "Q"):
      for n_demo in range(N_many_trajs):
        all_compT_list[n_demo] = set_many_trajs[n_demo].X.T
      all_compT_list[1 + n_demo] = one_traj.X.T
    elif (comp == "omega"):
      for n_demo in range(N_many_trajs):
        all_compT_list[n_demo] = set_many_trajs[n_demo].omega.T
      all_compT_list[1 + n_demo] = one_traj.omega.T
    elif (comp == "omegad"):
      for n_demo in range(N_many_trajs):
        all_compT_list[n_demo] = set_many_trajs[n_demo].omegad.T
      all_compT_list[1 + n_demo] = one_traj.omegad.T
    else:
      assert False, "Trajectory component named %s is un-defined!" % comp
    pypl_util.subplot_ND(
        NDtraj_list=all_compT_list,
        title=comp + title_suffix,
        Y_label_list=[
            comp + "%d" % comp_dim
            for comp_dim in range(all_compT_list[1 + n_demo].shape[1])
        ],
        fig_num=fig_num_offset + comp_idx,
        label_list=[
            many_traj_label + " #%d" % n_many_traj
            for n_many_traj in range(N_many_trajs)
        ] + [one_traj_label],
        color_style_list=[["b", ":"]] * N_many_trajs + [["g", "-"]],
        is_auto_line_coloring_and_styling=False)

  return None


def computeAndDisplayTrajectoryMSEVarGTNMSE(set_demo_trajs,
                                            unroll_traj,
                                            print_prefix="",
                                            is_orientation_trajectory=False):
  # assumes trajectory structure like DMPTrajectory or QuaternionDMPTrajectory
  unroll_traj_position = unroll_traj.X
  if (not is_orientation_trajectory):
    unroll_traj_velocity = unroll_traj.Xd
    unroll_traj_acceleration = unroll_traj.Xdd
  else:
    unroll_traj_velocity = unroll_traj.omega
    unroll_traj_acceleration = unroll_traj.omegad
  stretched_unroll_traj_position_list = list()
  stretched_unroll_traj_velocity_list = list()
  stretched_unroll_traj_acceleration_list = list()
  demo_traj_position_list = list()
  demo_traj_velocity_list = list()
  demo_traj_acceleration_list = list()
  for demo_traj in set_demo_trajs:
    demo_traj_position = demo_traj.X
    if (not is_orientation_trajectory):
      demo_traj_velocity = demo_traj.Xd
      demo_traj_acceleration = demo_traj.Xdd
    else:
      demo_traj_velocity = demo_traj.omega
      demo_traj_acceleration = demo_traj.omegad
    demo_traj_position_list.append(demo_traj_position)
    demo_traj_velocity_list.append(demo_traj_velocity)
    demo_traj_acceleration_list.append(demo_traj_acceleration)

    demo_traj_length = demo_traj_position.shape[1]

    if (unroll_traj_position.shape[1] != demo_traj_length):
      stretched_unroll_traj_position = stretchTrajectory(
          unroll_traj_position, demo_traj_length)
      stretched_unroll_traj_velocity = stretchTrajectory(
          unroll_traj_velocity, demo_traj_length)
      stretched_unroll_traj_acceleration = stretchTrajectory(
          unroll_traj_acceleration, demo_traj_length)
    else:
      stretched_unroll_traj_position = unroll_traj_position
      stretched_unroll_traj_velocity = unroll_traj_velocity
      stretched_unroll_traj_acceleration = unroll_traj_acceleration
    stretched_unroll_traj_position_list.append(stretched_unroll_traj_position)
    stretched_unroll_traj_velocity_list.append(stretched_unroll_traj_velocity)
    stretched_unroll_traj_acceleration_list.append(
        stretched_unroll_traj_acceleration)
  stacked_stretched_unroll_traj_position = np.hstack(
      stretched_unroll_traj_position_list)
  stacked_stretched_unroll_traj_velocity = np.hstack(
      stretched_unroll_traj_velocity_list)
  stacked_stretched_unroll_traj_acceleration = np.hstack(
      stretched_unroll_traj_acceleration_list)
  stacked_demo_traj_position = np.hstack(demo_traj_position_list)
  stacked_demo_traj_velocity = np.hstack(demo_traj_velocity_list)
  stacked_demo_traj_acceleration = np.hstack(demo_traj_acceleration_list)
  [MSE_position, varGT_position, NMSE_position] = computeMSEVarGTNMSE(
      predictions=stacked_stretched_unroll_traj_position,
      ground_truth=stacked_demo_traj_position,
      axis=1)
  [MSE_velocity, varGT_velocity, NMSE_velocity] = computeMSEVarGTNMSE(
      predictions=stacked_stretched_unroll_traj_velocity,
      ground_truth=stacked_demo_traj_velocity,
      axis=1)
  [MSE_acceleration, varGT_acceleration,
   NMSE_acceleration] = computeMSEVarGTNMSE(
       predictions=stacked_stretched_unroll_traj_acceleration,
       ground_truth=stacked_demo_traj_acceleration,
       axis=1)
  if (not is_orientation_trajectory):
    print(print_prefix + "X      MSE  = " + str(MSE_position))
    print(print_prefix + "Xd     MSE  = " + str(MSE_velocity))
    print(print_prefix + "Xdd    MSE  = " + str(MSE_acceleration))
    print("")
    print(print_prefix + "X      varGT= " + str(varGT_position))
    print(print_prefix + "Xd     varGT= " + str(varGT_velocity))
    print(print_prefix + "Xdd    varGT= " + str(varGT_acceleration))
    print("")
    print(print_prefix + "X      NMSE = " + str(NMSE_position))
    print(print_prefix + "Xd     NMSE = " + str(NMSE_velocity))
    print(print_prefix + "Xdd    NMSE = " + str(NMSE_acceleration))
  else:
    print(print_prefix + "Q      MSE  = " + str(MSE_position))
    print(print_prefix + "omega  MSE  = " + str(MSE_velocity))
    print(print_prefix + "omegad MSE  = " + str(MSE_acceleration))
    print("")
    print(print_prefix + "Q      varGT= " + str(varGT_position))
    print(print_prefix + "omega  varGT= " + str(varGT_velocity))
    print(print_prefix + "omegad varGT= " + str(varGT_acceleration))
    print("")
    print(print_prefix + "Q      NMSE = " + str(NMSE_position))
    print(print_prefix + "omega  NMSE = " + str(NMSE_velocity))
    print(print_prefix + "omegad NMSE = " + str(NMSE_acceleration))
  return [
      MSE_position, MSE_velocity, MSE_acceleration, varGT_position,
      varGT_velocity, varGT_acceleration, NMSE_position, NMSE_velocity,
      NMSE_acceleration
  ]
