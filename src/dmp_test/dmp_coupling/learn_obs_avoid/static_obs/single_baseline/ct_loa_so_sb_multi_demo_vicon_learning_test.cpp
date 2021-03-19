#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>

#include "dmp/cart_dmp/cart_coord_dmp/CartesianCoordDMP.h"
#include "dmp/cart_dmp/cart_coord_dmp/CartesianCoordTransformer.h"
#include "dmp/dmp_coupling/learn_obs_avoid/TransformCouplingLearnObsAvoid.h"
#include "dmp/dmp_discrete/CanonicalSystemDiscrete.h"
#include "dmp/dmp_state/DMPState.h"
#include "dmp/paths.h"
#include "dmp/utility/RealTimeAssertor.h"
#include "dmp/utility/utility.h"

using namespace dmp;

void print_usage() {
  printf(
      "Usage: dmp_ct_loa_so_sb_multi_demo_vicon_learning_demo [-f "
      "formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] "
      "[-m learning_method(0=_SCHAAL_LWR_METHOD_ OR "
      "1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r regularization_const] [-o "
      "loa_plot_dir_path] [-e rt_err_file_path]\n");
}

int main(int argc, char** argv) {
  bool is_real_time = false;

  double task_servo_rate = 300.0;  // ARM robot's task servo rate
  double dt = 1.0 / task_servo_rate;
  uint model_size = 25;

  uint formula_type = _SCHAAL_DMP_;
  // uint        formula_type                = _HOFFMANN_DMP_;
  uint canonical_order = 2;  // default is using 2nd order canonical system
  uint learning_method = _SCHAAL_LWR_METHOD_;
  char loa_data_dir_path[1000];
  std::string loa_data_dir_path_str = 
    get_data_path("/dmp_coupling/learn_obs_avoid/static_obs/data_multi_demo_vicon_static/");
  snprintf(loa_data_dir_path, sizeof(loa_data_dir_path),
           "%s", loa_data_dir_path_str.c_str());
  char loa_plot_dir_path[1000];
  std::string loa_plot_dir_path_str = 
    get_plot_path("/dmp_coupling/learn_obs_avoid/feature_trajectory/static_obs/"
                  "single_baseline/multi_demo_vicon/");
  snprintf(loa_plot_dir_path, sizeof(loa_plot_dir_path),
           "%s", loa_plot_dir_path_str.c_str());
  char dmp_plot_dir_path[1000];
  std::string dmp_plot_dir_path_str = 
    get_plot_path("/dmp_coupling/learn_obs_avoid/feature_trajectory/static_obs/"
                  "single_baseline/multi_demo_vicon/unroll_tests/");
  snprintf(dmp_plot_dir_path, sizeof(dmp_plot_dir_path),
           "%s", dmp_plot_dir_path_str.c_str());
  char var_input_file_path[1000] = "";
  char var_output_file_path[1000] = "";
  char rt_err_file_path[1000];
  std::string rt_err_file_path_str = 
    get_rt_errors_path("/rt_err.txt");
  snprintf(rt_err_file_path, sizeof(rt_err_file_path),
           "%s", rt_err_file_path_str.c_str());

  double tau_reproduce = 2.0;
  double regularization_const = 1e-5;
  uint max_num_trajs_per_setting = 2;
  std::vector<uint> selected_obs_avoid_setting_numbers(2);
  selected_obs_avoid_setting_numbers[0] = 27;
  selected_obs_avoid_setting_numbers[1] = 119;
  // selected_obs_avoid_setting_numbers[2]               = 194;

  Matrix4x4 ctraj_hmg_transform_global_to_local_matrix;

  static struct option long_options[] = {
      {"formula_type", required_argument, 0, 'f'},
      {"canonical_order", required_argument, 0, 'c'},
      {"learning_method", required_argument, 0, 'm'},
      {"regularization_const", required_argument, 0, 'r'},
      {"loa_plot_dir_path", required_argument, 0, 'o'},
      {"rt_err_file_path", required_argument, 0, 'e'},
      {0, 0, 0, 0}};

  int opt = 0;
  int long_index = 0;
  while ((opt = getopt_long(argc, argv, "f:c:m:r:o:e:", long_options,
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
        regularization_const = atof(optarg);
        break;
      case 'o':
        snprintf(loa_plot_dir_path, sizeof(loa_plot_dir_path), "%s", optarg);
        break;
      case 'e':
        snprintf(rt_err_file_path, sizeof(rt_err_file_path), "%s", optarg);
        break;
      default:
        print_usage();
        exit(EXIT_FAILURE);
    }
  }
  snprintf(dmp_plot_dir_path, sizeof(dmp_plot_dir_path),
           "%s/unroll_tests/", loa_plot_dir_path);

  if ((formula_type < _SCHAAL_DMP_) || (formula_type > _HOFFMANN_DMP_) ||
      (canonical_order < 1) || (canonical_order > 2) ||
      (learning_method < _SCHAAL_LWR_METHOD_) ||
      (learning_method > _JPETERS_GLOBAL_LEAST_SQUARES_METHOD_) ||
      (strcmp(loa_plot_dir_path, "") == 0) ||
      (strcmp(rt_err_file_path, "") == 0)) {
    print_usage();
    exit(EXIT_FAILURE);
  }

  // Initialize real-time assertor
  RealTimeAssertor rt_assertor(rt_err_file_path);
  rt_assertor.clear_rt_err_file();

  DMPState ee_cart_state_global(3, &rt_assertor);

  ObstacleStates point_obstacles_cart_state_global(2);

  DataIO data_io_main(&rt_assertor);
  if (rt_assert_main(data_io_main.writeMatrixToFile(
          loa_plot_dir_path, "tau_reproduce.txt", tau_reproduce)) == false) {
    return (-1);
  }

  // Initialize tau system
  TauSystem tau_sys(MIN_TAU, &rt_assertor);

  // Initialize canonical system
  CanonicalSystemDiscrete canonical_sys_discr(&tau_sys, &rt_assertor,
                                              canonical_order);

  /*TCLearnObsAvoidFeatureParameter tc_loa_feat_param(&rt_assertor,
                                                    &canonical_sys_discr,
                                                    _VICON_MARKERS_DATA_FILE_FORMAT_,
                                                    _AF_H14_NO_USE_,
                                                    10, 10,
                                                    (1.0/M_PI), (21.0/M_PI),
                                                    0.05, 0.75,
                                                    10,
                                                    0.05, 0.75,
                                                    _INVERSE_QUADRATIC_,
                                                    _PF_DYN2_,
                                                    5, 5,
                                                    0.5, 2.5,
                                                    0.01, 0.5,
                                                    _KGF_,
                                                    4, 5, 6,
                                                    0.0, M_PI/2.0,
                                                    0.01, 0.5,
                                                    0.0, 1.0);*/

  TCLearnObsAvoidFeatureParameter tc_loa_feat_param(
      &rt_assertor, &canonical_sys_discr, _VICON_MARKERS_DATA_FILE_FORMAT_,
      _AF_H14_NO_USE_, 10, 10, (1.0 / M_PI), (21.0 / M_PI), 0.05, 0.75, 10,
      0.05, 0.75, _INVERSE_QUADRATIC_, _PF_DYN2_NO_USE_, 5, 5, 0.5, 2.5, 0.01,
      0.5, _KGF_, 5, 5, 5, 0.0, M_PI, 0.01, 0.5, 0.0, 1.0);

  TransformCouplingLearnObsAvoid transform_coupling_learn_obs_avoid(
      &tc_loa_feat_param, &tau_sys, &ee_cart_state_global,
      &point_obstacles_cart_state_global,
      &ctraj_hmg_transform_global_to_local_matrix, &rt_assertor, true, true,
      loa_plot_dir_path);

  std::vector<TransformCoupling*> transform_couplers(1);
  // Using obstacle avoidance coupling term:
  transform_couplers[0] = &transform_coupling_learn_obs_avoid;

  // Initialize Cartesian Coordinate DMP
  CartesianCoordDMP cart_dmp(
      &canonical_sys_discr, model_size, formula_type, learning_method,
      &rt_assertor, _GSUTANTO_LOCAL_COORD_FRAME_, "", &transform_couplers);

  // Turn-off all scaling for now:
  std::vector<bool> is_using_scaling_init = std::vector<bool>(3, false);
  if (rt_assert_main(cart_dmp.setScalingUsage(is_using_scaling_init)) ==
      false) {
    return (-1);
  }

  // Learn both the coupling term and the BASELINE DMP:
  if (rt_assert_main(transform_coupling_learn_obs_avoid.learnCouplingTerm(
          loa_data_dir_path, task_servo_rate, &cart_dmp, regularization_const,
          true, max_num_trajs_per_setting,
          &selected_obs_avoid_setting_numbers)) == false) {
    return (-1);
  }

  return 0;
}
