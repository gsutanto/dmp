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
      "Usage: dmp_ct_loa_so_sb_multi_demo_vicon_PMNN_unrolling_demo [-f "
      "formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] "
      "[-m learning_method(0=_SCHAAL_LWR_METHOD_ OR "
      "1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r regularization_const] [-o "
      "loa_plot_dir_path] [-w ct_loa_weights_subpath] [-e rt_err_file_path]\n");
}

int main(int argc, char** argv) {
  bool is_real_time = false;

  double task_servo_rate = 300.0;  // ARM robot's task servo rate
  double dt = 1.0 / task_servo_rate;
  uint model_size = 25;
  uint PMNN_input_size = 17;
  uint PMNN_output_size = 3;

  uint formula_type = _SCHAAL_DMP_;
  uint canonical_order = 2;  // default is using 2nd order canonical system
  uint learning_method = _SCHAAL_LWR_METHOD_;

  std::vector<uint> topology(5);
  topology[0] = PMNN_input_size;
  topology[1] = 100;
  topology[2] = 75;
  topology[3] = model_size;
  topology[4] = 1;

  std::vector<uint> activation_functions(5);
  activation_functions[0] = _IDENTITY_;
  activation_functions[1] = _RELU_;
  activation_functions[2] = _TANH_;
  activation_functions[3] = _IDENTITY_;
  activation_functions[4] = _IDENTITY_;

  char loa_data_dir_path[1000];
  get_data_path(loa_data_dir_path, "/dmp_coupling/learn_obs_avoid/");
  char loa_data_prim_dir_path[1000];
  snprintf(loa_data_prim_dir_path, sizeof(loa_data_prim_dir_path),
           "%s/static_obs/learned_prims_params/position/prim1/",
           loa_data_dir_path);
  char loa_data_vicon_dir_path[1000];
  snprintf(loa_data_vicon_dir_path, sizeof(loa_data_vicon_dir_path),
           "%s/static_obs/data_multi_demo_vicon_static/", loa_data_dir_path);
  char loa_plot_dir_path[1000];
  get_plot_path(loa_plot_dir_path,
                "/dmp_coupling/learn_obs_avoid/feature_trajectory/static_obs/"
                "single_baseline/multi_demo_vicon/");
  char dmp_plot_dir_path[1000];
  get_plot_path(dmp_plot_dir_path,
                "/dmp_coupling/learn_obs_avoid/feature_trajectory/static_obs/"
                "single_baseline/multi_demo_vicon/unroll_tests/");
  char ct_loa_weights_subpath[1000];
  snprintf(ct_loa_weights_subpath, sizeof(ct_loa_weights_subpath),
           "learned_weights/1/learn_obs_avoid_weights_matrix_ARD.txt");
  char var_input_file_path[1000] = "";
  char var_output_file_path[1000] = "";
  char rt_err_file_path[1000];
  get_rt_errors_path(rt_err_file_path, "/rt_err.txt");

  double tau_reproduce = 2.0;
  double regularization_const = 1e-5;
  uint max_num_trajs_per_setting = 2;
  std::vector<uint> selected_obs_avoid_setting_numbers(2);
  selected_obs_avoid_setting_numbers[0] = 27;
  selected_obs_avoid_setting_numbers[1] = 119;
  // selected_obs_avoid_setting_numbers[2]               = 194;

  Matrix4x4 ctraj_hmg_transform_global_to_local_matrix;
  Vector3 goal_cart_position_global = ZeroVector3;

  static struct option long_options[] = {
      {"formula_type", required_argument, 0, 'f'},
      {"canonical_order", required_argument, 0, 'c'},
      {"learning_method", required_argument, 0, 'm'},
      {"regularization_const", required_argument, 0, 'r'},
      {"loa_plot_dir_path", required_argument, 0, 'o'},
      {"ct_loa_weights_subpath", required_argument, 0, 'w'},
      {"rt_err_file_path", required_argument, 0, 'e'},
      {0, 0, 0, 0}};

  int opt = 0;
  int long_index = 0;
  while ((opt = getopt_long(argc, argv, "f:c:m:r:o:w:e:", long_options,
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
      case 'w':
        snprintf(ct_loa_weights_subpath, sizeof(ct_loa_weights_subpath), "%s",
                 optarg);
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
      (strcmp(ct_loa_weights_subpath, "") == 0) ||
      (strcmp(rt_err_file_path, "") == 0)) {
    print_usage();
    exit(EXIT_FAILURE);
  }

  // Initialize real-time assertor
  RealTimeAssertor rt_assertor(rt_err_file_path);
  rt_assertor.clear_rt_err_file();

  DMPState ee_cart_state_global(3, &rt_assertor);

  ObstacleStates point_obstacles_cart_state_global;
  point_obstacles_cart_state_global.reserve(
      MIN_POINT_OBSTACLE_VECTOR_PREALLOCATION);

  // Initialize tau system
  TauSystem tau_sys(MIN_TAU, &rt_assertor);

  // Initialize canonical system
  CanonicalSystemDiscrete canonical_sys_discr(&tau_sys, &rt_assertor,
                                              canonical_order);

  uint NN_feature_method = _NN_;
  VectorNN_N X_vector(PMNN_input_size);
  VectorM func_approx_basis_functions(model_size);
  VectorNN_N normalized_phase_PSI_mult_phase_V(model_size);
  VectorNN_N Ct_vector(PMNN_output_size);
  std::shared_ptr<PMNN> pmnn;
  char pmnn_model_path[1000] = "";
  // Initialize Phase-Modulated Neural Network (PMNN):
  pmnn = std::make_shared<PMNN>(PMNN_output_size, topology,
                                activation_functions, &rt_assertor);

  snprintf(pmnn_model_path, sizeof(pmnn_model_path), "%sprim1/",
           get_data_path("/dmp_coupling/learn_obs_avoid/static_obs/neural_nets/"
                         "pmnn/cpp_models/")
               .c_str());
  if (rt_assert_main(pmnn->loadParams(pmnn_model_path, 0)) == false) {
    printf("Failed loading PMNN params for primitive #1\n");
    return (-1);
  }

  // Load trained Neural Network parameters for obstacle avoidance coupling
  // term:
  char NN_param_dir_path[1000];
  snprintf(NN_param_dir_path, sizeof(NN_param_dir_path),
           "%s/trained_NN_params/", loa_data_vicon_dir_path);
  TCLearnObsAvoidFeatureParameter tc_loa_feat_param(
      &rt_assertor, &canonical_sys_discr, _VICON_MARKERS_DATA_FILE_FORMAT_,
      _AF_H14_NO_USE_, 10, 10, (1.0 / M_PI), (21.0 / M_PI), 0.05, 0.75, 10,
      0.05, 0.75, _INVERSE_QUADRATIC_, _PF_DYN2_NO_USE_, 5, 5, 0.5, 2.5, 0.01,
      0.5, _KGF_NO_USE_, 5, 5, 5, 0.0, M_PI, 0.01, 0.5, 0.0, 1.0,
      NN_feature_method, 17, 20, 10, 3,  // 63, 10, 5, 2,
      NN_param_dir_path, &X_vector, &normalized_phase_PSI_mult_phase_V,
      &Ct_vector);
  tc_loa_feat_param.pmnn = pmnn.get();

  std::vector<TransformCoupling*> transform_couplers(1);

  // Initialize Cartesian Coordinate DMP
  CartesianCoordDMP cart_dmp(
      &canonical_sys_discr, model_size, formula_type, learning_method,
      &rt_assertor, _GSUTANTO_LOCAL_COORD_FRAME_, "", &transform_couplers);

  TransformCouplingLearnObsAvoid transform_coupling_learn_obs_avoid(
      &tc_loa_feat_param, &tau_sys, &ee_cart_state_global,
      &point_obstacles_cart_state_global,
      &ctraj_hmg_transform_global_to_local_matrix, &rt_assertor, true, true,
      loa_plot_dir_path, &goal_cart_position_global,
      (FuncApproximatorDiscrete*)cart_dmp.getFunctionApproximatorPointer());

  // Turn-off all scaling for now:
  std::vector<bool> is_using_scaling_init = std::vector<bool>(3, false);
  if (rt_assert_main(cart_dmp.setScalingUsage(is_using_scaling_init)) ==
      false) {
    return (-1);
  }

  // Load the obstacle avoidance coupling term weights:
  if (NN_feature_method == _NN_NO_USE_) {
    if (rt_assert_main(transform_coupling_learn_obs_avoid.loadWeights(
            loa_plot_dir_path, ct_loa_weights_subpath)) == false) {
      return (-1);
    }
  }

  // We will load all of the obstacle avoidance demonstrated trajectories
  // across different obstacle settings from the specified directories into
  // the following data structure:
  DemonstrationGroupSetPtr input_demo_group_set_obs_avoid_global;
  if (rt_assert_main(allocateMemoryIfNonRealTime(
          is_real_time, input_demo_group_set_obs_avoid_global)) == false) {
    return false;
  }

  // Container for the set of point obstacles (processed from Vicon data), each
  // corresponds to a set of demonstrations:
  PointsVectorPtr vector_point_obstacles_cart_position_global;
  if (rt_assert_main(allocateMemoryIfNonRealTime(
          is_real_time, vector_point_obstacles_cart_position_global)) ==
      false) {
    return false;
  }

  ObstacleStatesVectorPtr point_obss_cart_state_traj_global;
  if (rt_assert_main(allocateMemoryIfNonRealTime(
          is_real_time, point_obss_cart_state_traj_global)) == false) {
    return false;
  }

  uint N_demo_settings;

  VecVecDMPUnrollInitParamsPtr demo_group_set_dmp_unroll_init_params;
  if (rt_assert_main(allocateMemoryIfNonRealTime(
          is_real_time, demo_group_set_dmp_unroll_init_params)) == false) {
    return false;
  }

  if (rt_assert_main(transform_coupling_learn_obs_avoid
                         .extractObstacleAvoidanceTrainingDataVicon(
                             loa_data_vicon_dir_path,
                             *input_demo_group_set_obs_avoid_global,
                             *vector_point_obstacles_cart_position_global,
                             N_demo_settings, max_num_trajs_per_setting,
                             &selected_obs_avoid_setting_numbers,
                             demo_group_set_dmp_unroll_init_params.get())) ==
      false) {
    return (-1);
  }

  // Load baseline DMP (forcing term weights and other parameter):
  if (rt_assert_main(cart_dmp.loadParams(loa_data_prim_dir_path, "w", "A_learn",
                                         "start_global", "goal_global",
                                         "tau")) == false) {
    return (-1);
  }

  DMPState current_state(3, &rt_assertor);

  /****************** without obstacle (START) ******************/
  // NOT using obstacle avoidance coupling term:
  transform_couplers[0] = NULL;

  tau_reproduce = (*demo_group_set_dmp_unroll_init_params)[0][0].tau;

  // define unrolling parameters:
  DMPUnrollInitParams dmp_unroll_init_parameters_wo_obs(
      tau_reproduce, DMPState(cart_dmp.getMeanStartPosition(), &rt_assertor),
      DMPState(cart_dmp.getMeanGoalPosition(), &rt_assertor), &rt_assertor);
  if (rt_assert_main(cart_dmp.start(dmp_unroll_init_parameters_wo_obs)) ==
      false) {
    return (-1);
  }

  // Run DMP and print state values:
  //    if (rt_assert_main(cart_dmp.startCollectingTrajectoryDataSet()) ==
  //    false)
  //    {
  //        return (-1);
  //    }
  for (uint i = 1; i <= (round(tau_reproduce * task_servo_rate) + 1); ++i) {
    double time = 1.0 * (i * dt);

    // Get the next state of the Cartesian DMP
    if (rt_assert_main(cart_dmp.getNextState(dt, true, current_state)) ==
        false) {
      return (-1);
    }

    // Log state trajectory:
    printf("%.05f %.05f %.05f\n", current_state.getX()[0],
           current_state.getX()[1], current_state.getX()[2]);
  }
  //    cart_dmp.stopCollectingTrajectoryDataSet();
  //    snprintf(var_output_file_path, sizeof(var_output_file_path),
  //             "%s/baseline/", dmp_plot_dir_path);
  //    if (rt_assert_main(cart_dmp.saveTrajectoryDataSet(var_output_file_path))
  //    == false)
  //    {
  //        return (-1);
  //    }
  /****************** without obstacle (END) ******************/

  if (rt_assert_main(
          (demo_group_set_dmp_unroll_init_params->size() == N_demo_settings) &&
          (vector_point_obstacles_cart_position_global->size() ==
           N_demo_settings)) == false) {
    return (-1);
  }

  /****************** with obstacle (START) ******************/
  // Using obstacle avoidance coupling term:
  transform_couplers[0] = &transform_coupling_learn_obs_avoid;
  for (uint n = 0; n < N_demo_settings; n++) {
    for (uint j = 0; j < (*demo_group_set_dmp_unroll_init_params)[n].size();
         j++) {
      //            std::cout << "Unrolling learned obstacle avoidance on Demo
      //            Setting #" << (n+1) << "/" << N_demo_settings << ", demo #"
      //            << (j+1) << "/" <<
      //            (*demo_group_set_dmp_unroll_init_params)[n].size() << " ..."
      //            << std::endl;

      tau_reproduce = (*demo_group_set_dmp_unroll_init_params)[n][j].tau;

      // define unrolling parameters:
      DMPUnrollInitParams dmp_unroll_init_parameters_w_obs(
          tau_reproduce,
          DMPState(cart_dmp.getMeanStartPosition(), &rt_assertor),
          DMPState(cart_dmp.getMeanGoalPosition(), &rt_assertor), &rt_assertor);
      if (rt_assert_main(cart_dmp.start(dmp_unroll_init_parameters_w_obs)) ==
          false) {
        return (-1);
      }
      current_state = cart_dmp.getCurrentState();
      ctraj_hmg_transform_global_to_local_matrix =
          cart_dmp.getHomogeneousTransformMatrixGlobalToLocal();
      goal_cart_position_global = cart_dmp.getMeanGoalPosition();

      if (rt_assert_main(
              transform_coupling_learn_obs_avoid
                  .computePointObstaclesCartStateGlobalTrajectoryFromStaticViconData(
                      (*vector_point_obstacles_cart_position_global)[n],
                      (*point_obss_cart_state_traj_global))) == false) {
        return false;
      }

      // Run DMP and print state values:
      //            if
      //            (rt_assert_main(cart_dmp.startCollectingTrajectoryDataSet())
      //            == false)
      //            {
      //                return (-1);
      //            }
      for (uint i = 1; i <= (round(tau_reproduce * task_servo_rate) + 1); ++i) {
        double time = 1.0 * (i * dt);

        // IMPORTANT PART here:
        // Here ee_cart_state_global is updated using
        // the current (or real-time) DMP unrolled trajectory. These variable
        // will in turn be used for computing the features of obstacle avoidance
        // coupling term. Thus the resulting obstacle avoidance coupling term is
        // the "realistic" one, i.e. i.e. conditioned with features computed
        // with current information.
        ee_cart_state_global = current_state;
        point_obstacles_cart_state_global =
            (*point_obss_cart_state_traj_global)[0];
        goal_cart_position_global = cart_dmp.getMeanGoalPosition();

        // Get the next state of the Cartesian DMP
        if (rt_assert_main(cart_dmp.getNextState(dt, false, current_state)) ==
            false) {
          return (-1);
        }

        if (rt_assert_main(canonical_sys_discr.updateCanonicalState(dt)) ==
            false) {
          printf("Failed updating canonical state!!!\n");
          return (-1);
        }

        // Log state trajectory:
        printf("%.05f %.05f %.05f\n", current_state.getX()[0],
               current_state.getX()[1], current_state.getX()[2]);
      }
      //            cart_dmp.stopCollectingTrajectoryDataSet();
      //            snprintf(var_output_file_path, sizeof(var_output_file_path),
      //            "%s/%u/%u/", dmp_plot_dir_path, (n+1), (j+1)); if
      //            (rt_assert_main(cart_dmp.saveTrajectoryDataSet(var_output_file_path))
      //            == false)
      //            {
      //                return (-1);
      //            }
    }
  }
  /****************** with obstacle (END) ******************/

  return 0;
}
