#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>

#include "dmp/cart_dmp/cart_coord_dmp/CartesianCoordDMP.h"
#include "dmp/cart_dmp/cart_coord_dmp/CartesianCoordTransformer.h"
#include "dmp/dmp_discrete/CanonicalSystemDiscrete.h"
#include "dmp/dmp_discrete/DMPDiscrete.h"
#include "dmp/dmp_state/DMPState.h"
#include "dmp/paths.h"

using namespace dmp;

void print_usage() {
  printf(
      "Usage: dmp_cart_coord_dmp_multi_traj_training_demo [-f "
      "formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] "
      "[-m learning_method(0=_SCHAAL_LWR_METHOD_ OR "
      "1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r time_reproduce_max(>=0)] "
      "[-t tau_reproduce(>=0)] [-e rt_err_file_path]\n");
}

int main(int argc, char** argv) {
  bool is_real_time = false;

  double task_servo_rate = 1000.0;  // MasterArm robot's task servo rate
  double dt = 1.0 / task_servo_rate;
  uint model_size = 25;
  double tau =
      0.5;  // this value will be changed (possibly many times) during learning,
            // because each trajectory have different tau value

  double tau_learn_dataset1;
  double tau_learn_dataset2;
  Trajectory critical_states_learn_dataset1(2);
  Trajectory critical_states_learn_dataset2(2);
  double time;

  uint formula_type = _SCHAAL_DMP_;
  uint canonical_order = 2;  // default is using 2nd order canonical system
  uint learning_method = _SCHAAL_LWR_METHOD_;
  double time_reproduce_max = 0.5;
  double tau_reproduce = 0.5;
  char dmp_plot_dir_path[1000];
  get_plot_path(dmp_plot_dir_path,
                "/cart_dmp/cart_coord_dmp/multi_traj_training/");
  char rt_err_file_path[1000];
  get_rt_errors_path(rt_err_file_path, "/rt_errors/rt_err.txt");

  static struct option long_options[] = {
      {"formula_type", required_argument, 0, 'f'},
      {"canonical_order", required_argument, 0, 'c'},
      {"learning_method", required_argument, 0, 'm'},
      {"time_reproduce_max", required_argument, 0, 'r'},
      {"tau_reproduce", required_argument, 0, 't'},
      {"rt_err_file_path", required_argument, 0, 'e'},
      {0, 0, 0, 0}};

  int opt = 0;
  int long_index = 0;
  while ((opt = getopt_long(argc, argv, "f:c:m:r:t:e:", long_options,
                            &long_index)) != -1) {
    switch (opt) {
      case 'f':
        formula_type = atoi(optarg);
        break;
      case 'c':
        canonical_order = atoi(optarg);
        break;
      case 'm':
        learning_method = atoi(optarg);
        break;
      case 'r':
        time_reproduce_max = atof(optarg);
        break;
      case 't':
        tau_reproduce = atof(optarg);
        break;
      case 'e':
        strcpy(rt_err_file_path, optarg);
        break;
      default:
        print_usage();
        exit(EXIT_FAILURE);
    }
  }

  if ((formula_type < _SCHAAL_DMP_) || (formula_type > _HOFFMANN_DMP_) ||
      (canonical_order < 1) || (canonical_order > 2) ||
      (learning_method < _SCHAAL_LWR_METHOD_) ||
      (learning_method > _JPETERS_GLOBAL_LEAST_SQUARES_METHOD_) ||
      (time_reproduce_max < 0.0) || (tau_reproduce < 0.0) ||
      (strcmp(rt_err_file_path, "") == 0)) {
    print_usage();
    exit(EXIT_FAILURE);
  }

  if (time_reproduce_max < 0.5) {
    time_reproduce_max = 0.5;
  }

  // Initialize real-time assertor
  RealTimeAssertor rt_assertor(rt_err_file_path);
  rt_assertor.clear_rt_err_file();

  // Initialize tau system
  TauSystem tau_sys(tau, &rt_assertor);

  // Initialize canonical system
  CanonicalSystemDiscrete canonical_sys_discr(&tau_sys, &rt_assertor,
                                              canonical_order);

  // Initialize Cartesian Coordinate DMP
  CartesianCoordDMP cart_dmp(&canonical_sys_discr, model_size, formula_type,
                             learning_method, &rt_assertor,
                             _GSUTANTO_LOCAL_COORD_FRAME_, dmp_plot_dir_path);

  DMPState current_state(3, &rt_assertor);

  // change tau here during movement reproduction
  tau = tau_reproduce;

  ///////////////////////////// Dataset 1 (Start) /////////////////////////////
  // Learn the Cartesian trajectory
  if (rt_assert_main(cart_dmp.learn(
          get_data_path(
              "/cart_dmp/cart_coord_dmp/multi_traj_training/dataset1/")
              .c_str(),
          task_servo_rate, &tau_learn_dataset1,
          &critical_states_learn_dataset1)) == false) {
    return (-1);
  }

  // Reproduce Cartesian DMP
  // define unrolling parameters:
  DMPUnrollInitParams dmp_unroll_init_params_dataset1(
      tau, critical_states_learn_dataset1, &rt_assertor);
  // Start DMP
  if (rt_assert_main(cart_dmp.start(dmp_unroll_init_params_dataset1)) ==
      false) {
    return (-1);
  }

  // Run Cartesian DMP and print state values
  if (rt_assert_main(cart_dmp.startCollectingTrajectoryDataSet()) == false) {
    return (-1);
  }
  for (uint i = 0; i < (round(time_reproduce_max * task_servo_rate) + 1); ++i) {
    time = 1.0 * (i * dt);

    // Get the next state of the Cartesian DMP
    if (rt_assert_main(cart_dmp.getNextState(dt, true, current_state)) ==
        false) {
      return (-1);
    }

    // Log state trajectory:
    printf("%.05f %.05f %.05f %.05f\n", time, current_state.getX()[0],
           current_state.getX()[1], current_state.getX()[2]);
  }
  cart_dmp.stopCollectingTrajectoryDataSet();
  if (rt_assert_main(cart_dmp.saveTrajectoryDataSet(
          get_plot_path(
              "/cart_dmp/cart_coord_dmp/multi_traj_training/dataset1/")
              .c_str())) == false) {
    return (-1);
  }
  if (rt_assert_main(cart_dmp.saveParams(
          get_plot_path(
              "/cart_dmp/cart_coord_dmp/multi_traj_training/dataset1/")
              .c_str())) == false) {
    return (-1);
  }

  // Also print out the parameters:
  MatrixNxM w(3, model_size);
  VectorN A_learn(3);
  if (rt_assert_main(cart_dmp.getParams(w, A_learn)) == false) {
    return (-1);
  }
  Vector3 mean_start_global_position = cart_dmp.getMeanStartGlobalPosition();
  Vector3 mean_goal_global_position = cart_dmp.getMeanGoalGlobalPosition();
  Vector3 mean_start_local_position = cart_dmp.getMeanStartLocalPosition();
  Vector3 mean_goal_local_position = cart_dmp.getMeanGoalLocalPosition();
  Matrix4x4 T_local_to_global_H =
      cart_dmp.getHomogeneousTransformMatrixLocalToGlobal();
  Matrix4x4 T_global_to_local_H =
      cart_dmp.getHomogeneousTransformMatrixGlobalToLocal();
  for (uint im = 0; im < model_size; im++) {
    printf("%.05f %.05f %.05f %.05f\n", w(0, im), w(1, im), w(2, im), 0.0);
  }
  printf("%.05f %.05f %.05f %.05f\n", A_learn(0), A_learn(1), A_learn(2),
         tau_learn_dataset1);
  printf("%.05f %.05f %.05f %.05f\n", mean_start_global_position(0),
         mean_start_global_position(1), mean_start_global_position(2), 0.0);
  printf("%.05f %.05f %.05f %.05f\n", mean_goal_global_position(0),
         mean_goal_global_position(1), mean_goal_global_position(2), 0.0);
  printf("%.05f %.05f %.05f %.05f\n", mean_start_local_position(0),
         mean_start_local_position(1), mean_start_local_position(2), 0.0);
  printf("%.05f %.05f %.05f %.05f\n", mean_goal_local_position(0),
         mean_goal_local_position(1), mean_goal_local_position(2), 0.0);
  for (uint ir = 0; ir < 4; ir++) {
    printf("%.05f %.05f %.05f %.05f\n", T_local_to_global_H(ir, 0),
           T_local_to_global_H(ir, 1), T_local_to_global_H(ir, 2),
           T_local_to_global_H(ir, 3));
  }
  for (uint ir = 0; ir < 4; ir++) {
    printf("%.05f %.05f %.05f %.05f\n", T_global_to_local_H(ir, 0),
           T_global_to_local_H(ir, 1), T_global_to_local_H(ir, 2),
           T_global_to_local_H(ir, 3));
  }
  ///////////////////////////// Dataset 1 (End) /////////////////////////////

  ///////////////////////////// Dataset 2 (Start) /////////////////////////////
  // Learn the Cartesian trajectory
  if (rt_assert_main(cart_dmp.learn(
          get_data_path(
              "/cart_dmp/cart_coord_dmp/multi_traj_training/dataset2/")
              .c_str(),
          task_servo_rate, &tau_learn_dataset2,
          &critical_states_learn_dataset2)) == false) {
    return (-1);
  }

  // Reproduce Cartesian DMP
  // define unrolling parameters:
  DMPUnrollInitParams dmp_unroll_init_params_dataset2(
      tau, critical_states_learn_dataset2, &rt_assertor);
  // Start DMP
  if (rt_assert_main(cart_dmp.start(dmp_unroll_init_params_dataset2)) ==
      false) {
    return (-1);
  }

  // Run Cartesian DMP and print state values
  if (rt_assert_main(cart_dmp.startCollectingTrajectoryDataSet()) == false) {
    return (-1);
  }
  for (uint i = 0; i < (round(time_reproduce_max * task_servo_rate) + 1); ++i) {
    time = 1.0 * (i * dt);

    // Get the next state of the Cartesian DMP
    if (rt_assert_main(cart_dmp.getNextState(dt, true, current_state)) ==
        false) {
      return (-1);
    }

    // Log state trajectory:
    printf("%.05f %.05f %.05f %.05f\n", time, current_state.getX()[0],
           current_state.getX()[1], current_state.getX()[2]);
  }
  cart_dmp.stopCollectingTrajectoryDataSet();
  if (rt_assert_main(cart_dmp.saveTrajectoryDataSet(
          get_plot_path(
              "/cart_dmp/cart_coord_dmp/multi_traj_training/dataset2/")
              .c_str())) == false) {
    return (-1);
  }
  if (rt_assert_main(cart_dmp.saveParams(
          get_plot_path(
              "/cart_dmp/cart_coord_dmp/multi_traj_training/dataset2/")
              .c_str())) == false) {
    return (-1);
  }

  // Also print out the parameters:
  if (rt_assert_main(cart_dmp.getParams(w, A_learn)) == false) {
    return (-1);
  }
  mean_start_global_position = cart_dmp.getMeanStartGlobalPosition();
  mean_goal_global_position = cart_dmp.getMeanGoalGlobalPosition();
  mean_start_local_position = cart_dmp.getMeanStartLocalPosition();
  mean_goal_local_position = cart_dmp.getMeanGoalLocalPosition();
  T_local_to_global_H = cart_dmp.getHomogeneousTransformMatrixLocalToGlobal();
  T_global_to_local_H = cart_dmp.getHomogeneousTransformMatrixGlobalToLocal();
  for (uint im = 0; im < model_size; im++) {
    printf("%.05f %.05f %.05f %.05f\n", w(0, im), w(1, im), w(2, im), 0.0);
  }
  printf("%.05f %.05f %.05f %.05f\n", A_learn(0), A_learn(1), A_learn(2),
         tau_learn_dataset2);
  printf("%.05f %.05f %.05f %.05f\n", mean_start_global_position(0),
         mean_start_global_position(1), mean_start_global_position(2), 0.0);
  printf("%.05f %.05f %.05f %.05f\n", mean_goal_global_position(0),
         mean_goal_global_position(1), mean_goal_global_position(2), 0.0);
  printf("%.05f %.05f %.05f %.05f\n", mean_start_local_position(0),
         mean_start_local_position(1), mean_start_local_position(2), 0.0);
  printf("%.05f %.05f %.05f %.05f\n", mean_goal_local_position(0),
         mean_goal_local_position(1), mean_goal_local_position(2), 0.0);
  for (uint ir = 0; ir < 4; ir++) {
    printf("%.05f %.05f %.05f %.05f\n", T_local_to_global_H(ir, 0),
           T_local_to_global_H(ir, 1), T_local_to_global_H(ir, 2),
           T_local_to_global_H(ir, 3));
  }
  for (uint ir = 0; ir < 4; ir++) {
    printf("%.05f %.05f %.05f %.05f\n", T_global_to_local_H(ir, 0),
           T_global_to_local_H(ir, 1), T_global_to_local_H(ir, 2),
           T_global_to_local_H(ir, 3));
  }
  ///////////////////////////// Dataset 2 (End) /////////////////////////////

  return 0;
}
