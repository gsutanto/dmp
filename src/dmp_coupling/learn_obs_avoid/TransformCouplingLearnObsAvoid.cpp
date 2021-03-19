#include "dmp/dmp_coupling/learn_obs_avoid/TransformCouplingLearnObsAvoid.h"

namespace dmp {

/**
 * NON-REAL-TIME!!!
 */
TransformCouplingLearnObsAvoid::TransformCouplingLearnObsAvoid()
    : TransformCoupling(),
      cart_coord_transformer(CartesianCoordTransformer()),
      loa_data_io(LearnObsAvoidDataIO()),
      is_logging_learning_data(false),
      is_collecting_trajectory_data(false),
      is_constraining_Rp_yd_relationship(false),
      tau_sys(NULL),
      endeff_cartesian_state_global(NULL),
      point_obstacles_cartesian_state_global(NULL),
      ctraj_hmg_transform_global_to_local_matrix(NULL),
      goal_cartesian_position_global(NULL),
      goal_cartesian_position_local(ZeroVector3),
      func_approx_discrete(NULL) {
  snprintf(data_directory_path, sizeof(data_directory_path), "");
}

/**
 * NON-REAL-TIME!!!\n
 *
 * @param real_time_assertor Real-Time Assertor for troubleshooting and
 * debugging
 * @param opt_is_logging_learning_data [optional] Option to save/log learning
 * data (i.e. weights matrix, Ct_target, and Ct_fit) into a file (set to 'true'
 * to save)
 * @param opt_data_directory_path [optional] Full directory path where to
 * save/log all the data
 */
TransformCouplingLearnObsAvoid::TransformCouplingLearnObsAvoid(
    TCLearnObsAvoidFeatureParameter* loa_feature_parameter,
    TauSystem* tau_system, DMPState* endeff_cart_state_global,
    ObstacleStates* point_obstacles_cart_state_global,
    Matrix4x4* cart_traj_homogeneous_transform_global_to_local_matrix,
    RealTimeAssertor* real_time_assertor,
    bool opt_is_constraining_Rp_yd_relationship,
    bool opt_is_logging_learning_data, const char* opt_data_directory_path,
    Vector3* goal_position_global,
    FuncApproximatorDiscrete* function_approximator_discrete)
    : TransformCoupling(3, real_time_assertor),
      loa_feat_param(loa_feature_parameter),
      cart_coord_transformer(CartesianCoordTransformer(real_time_assertor)),
      loa_data_io(LearnObsAvoidDataIO(real_time_assertor)),
      is_logging_learning_data(opt_is_logging_learning_data),
      is_collecting_trajectory_data(false),
      is_constraining_Rp_yd_relationship(
          opt_is_constraining_Rp_yd_relationship),
      tau_sys(tau_system),
      endeff_cartesian_state_global(endeff_cart_state_global),
      endeff_cartesian_state_local(new DMPState(3, real_time_assertor)),
      point_obstacles_cartesian_state_global(point_obstacles_cart_state_global),
      ctraj_hmg_transform_global_to_local_matrix(
          cart_traj_homogeneous_transform_global_to_local_matrix),
      goal_cartesian_position_global(goal_position_global),
      goal_cartesian_position_local(ZeroVector3),
      func_approx_discrete(function_approximator_discrete) {
  bool is_real_time = false;

  snprintf(data_directory_path, sizeof(data_directory_path),
           "%s", opt_data_directory_path);

  loa_data_io.setFeatureVectorSize(loa_feat_param->feature_vector_size);

  rt_assert(allocateMemoryIfNonRealTime(is_real_time,
                                        point_obstacles_cartesian_state_local));
  point_obstacles_cartesian_state_local->reserve(MAX_TRAJ_SIZE);

  rt_assert(allocateMemoryIfNonRealTime(
      is_real_time, point_obstacles_cartesian_distances_from_endeff,
      MAX_TRAJ_SIZE, 1));

  if (is_logging_learning_data) {
    if (loa_feat_param->is_using_NN == false) {
      if (loa_feat_param->is_using_AF_H14_feature) {
        rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "AF_H14_N_beta_phi1_phi2_grid.txt",
            loa_feat_param->AF_H14_N_beta_phi1_phi2_grid));
        rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "AF_H14_N_k_phi1_phi2_grid.txt",
            loa_feat_param->AF_H14_N_k_phi1_phi2_grid));
        rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "AF_H14_N_k_phi3_grid.txt",
            loa_feat_param->AF_H14_N_k_phi3_grid));
        rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "AF_H14_beta_phi1_phi2_low.txt",
            loa_feat_param->AF_H14_beta_phi1_phi2_low));
        rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "AF_H14_beta_phi1_phi2_high.txt",
            loa_feat_param->AF_H14_beta_phi1_phi2_high));
        rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "AF_H14_k_phi1_phi2_low.txt",
            loa_feat_param->AF_H14_k_phi1_phi2_low));
        rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "AF_H14_k_phi1_phi2_high.txt",
            loa_feat_param->AF_H14_k_phi1_phi2_high));
        rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "AF_H14_k_phi3_low.txt",
            loa_feat_param->AF_H14_k_phi3_low));
        rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "AF_H14_k_phi3_high.txt",
            loa_feat_param->AF_H14_k_phi3_high));
        rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "AF_H14_beta_phi1_phi2_grid.txt",
            *(loa_feat_param->AF_H14_beta_phi1_phi2_grid)));
        rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "AF_H14_k_phi1_phi2_grid.txt",
            *(loa_feat_param->AF_H14_k_phi1_phi2_grid)));
        rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "AF_H14_beta_phi1_phi2_vector.txt",
            *(loa_feat_param->AF_H14_beta_phi1_phi2_vector)));
        rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "AF_H14_k_phi1_phi2_vector.txt",
            *(loa_feat_param->AF_H14_k_phi1_phi2_vector)));
        if (loa_feat_param->AF_H14_N_k_phi3_grid > 0) {
          rt_assert(loa_data_io.writeMatrixToFile(
              data_directory_path, "AF_H14_k_phi3_vector.txt",
              *(loa_feat_param->AF_H14_k_phi3_vector)));
        }
        rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "AF_H14_num_indep_feat_obs_points.txt",
            loa_feat_param->AF_H14_num_indep_feat_obs_points));
      }
      rt_assert(loa_data_io.writeMatrixToFile(
          data_directory_path, "parameter_table.txt",
          *(loa_feat_param->parameter_table)));
    }
  }
}

/**
 * Checks whether this transformation coupling term for obstacle avoidance by
 * demonstration is valid or not.
 *
 * @return Valid (true) or invalid (false)
 */
bool TransformCouplingLearnObsAvoid::isValid() {
  if (rt_assertor == NULL) {
    return false;
  }
  if (rt_assert(TransformCoupling::isValid()) == false) {
    return false;
  }
  if (rt_assert(
          (rt_assert(loa_feat_param != NULL)) && (rt_assert(tau_sys != NULL)) &&
          (rt_assert(endeff_cartesian_state_global != NULL)) &&
          (rt_assert(point_obstacles_cartesian_state_global != NULL)) &&
          (rt_assert(point_obstacles_cartesian_state_local != NULL)) &&
          (rt_assert(ctraj_hmg_transform_global_to_local_matrix != NULL)) &&
          (rt_assert(point_obstacles_cartesian_distances_from_endeff !=
                     NULL))) == false) {
    return false;
  }
  if (rt_assert((rt_assert(cart_coord_transformer.isValid())) &&
                (rt_assert(loa_data_io.isValid())) &&
                (rt_assert(loa_feat_param->isValid())) &&
                (rt_assert(tau_sys->isValid()))) == false) {
    return false;
  }
  if (rt_assert(loa_data_io.getFeatureVectorSize() ==
                loa_feat_param->feature_vector_size) == false) {
    return false;
  }
  if (loa_feat_param->pmnn != NULL) {
    if (rt_assert(func_approx_discrete != NULL) == false) {
      return false;
    }
    if (rt_assert(func_approx_discrete->isValid()) == false) {
      return false;
    }
    if (rt_assert(func_approx_discrete->getCanonicalSystemPointer()
                      ->getTauSystemPointer() == tau_sys) == false) {
      return false;
    }
  }
  return true;
}

/**
 * NON-REAL-TIME!!!\n
 * Learn coupling term for obstacle avoidance from demonstrated trajectories \n
 * (at baseline (during the absence of an obstacle) and during the presence of
 * an obstacle).
 *
 * @param training_data_directory Directory where all training data is stored\n
 *        This directory must contain the following:\n
 *        (1) directory "baseline" (contains demonstrated trajectories
 * information when there is NO obstacle),\n (2) numerical directories (start
 * from "1", each contains demonstrated trajectories information \n when there
 * is an obstacle at a specific position/setting,\n which is different between a
 * numbered directory and the other)
 * @param robot_task_servo_rate Robot's task servo rate
 * @param cart_dmp_baseline_ptr Pointer to the BASELINE CartesianCoordDMP
 * @param regularization_const [optional] Regularization constant for the ridge
 * regression
 * @param is_learning_baseline [optional] Also learning the baseline DMP forcing
 * term?
 * @param max_num_trajs_per_setting [optional] Maximum number of trajectories
 * that could be loaded into a trajectory set \n of an obstacle avoidance
 * demonstration setting \n (setting = obstacle position) (default is 500)
 * @param selected_obs_avoid_setting_numbers [optional] If we are just
 * interested in learning a subset of the obstacle avoidance demonstration
 * settings,\n then specify the numbers (IDs) of the setting here (as a
 * vector).\n If not specified, then all settings will be considered.
 * @param max_critical_point_distance_baseline_vs_obs_avoid_demo [optional]
 * Maximum tolerable distance (in meter) \n between start position in baseline
 * and obstacle avoidance demonstrations,\n as well as between goal position in
 * baseline and obstacle avoidance demonstrations.
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::learnCouplingTerm(
    const char* training_data_directory, double robot_task_servo_rate,
    CartesianCoordDMP* cart_dmp_baseline_ptr, double regularization_const,
    bool is_learning_baseline, uint max_num_trajs_per_setting,
    std::vector<uint>* selected_obs_avoid_setting_numbers,
    double max_critical_point_distance_baseline_vs_obs_avoid_demo) {
  bool is_real_time = false;

  // pre-conditions checking
  if (rt_assert(TransformCouplingLearnObsAvoid::isValid()) == false) {
    return false;
  }
  // input checking
  if (rt_assert((rt_assert(robot_task_servo_rate > 0.0)) &&
                (rt_assert(robot_task_servo_rate <= MAX_TASK_SERVO_RATE))) ==
      false) {
    return false;
  }
  if (rt_assert(cart_dmp_baseline_ptr != NULL) == false) {
    return false;
  }
  if (rt_assert(cart_dmp_baseline_ptr->isValid()) == false) {
    return false;
  }
  if (rt_assert(regularization_const > 0.0) == false) {
    return false;
  }

  // We will load all of the obstacle avoidance demonstrated trajectories
  // across different obstacle settings from the specified directories into
  // the following data structure:
  DemonstrationGroupSetPtr input_demo_group_set_obs_avoid_global;
  if (rt_assert(allocateMemoryIfNonRealTime(
          is_real_time, input_demo_group_set_obs_avoid_global)) == false) {
    return false;
  }

  // Container for the set of center point coordinates of the spherical
  // obstacles, in Cartesian global coordinate system, each corresponds to a set
  // of demonstrations:
  PointsPtr obs_sph_centers_cart_position_global;
  if (rt_assert(allocateMemoryIfNonRealTime(
          is_real_time, obs_sph_centers_cart_position_global)) == false) {
    return false;
  }

  // Container for the set of spherical obstacles radius, each corresponds to a
  // set of demonstrations:
  DoubleVectorPtr obs_sph_radius;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, obs_sph_radius)) ==
      false) {
    return false;
  }

  // Container for the set of point obstacles (processed from Vicon data), each
  // corresponds to a set of demonstrations:
  PointsVectorPtr vector_point_obstacles_cart_position_global;
  if (rt_assert(allocateMemoryIfNonRealTime(
          is_real_time, vector_point_obstacles_cart_position_global)) ==
      false) {
    return false;
  }

  ObstacleStatesVectorPtr point_obss_cart_state_traj_global;
  if (rt_assert(allocateMemoryIfNonRealTime(
          is_real_time, point_obss_cart_state_traj_global)) == false) {
    return false;
  }

  uint N_demo_settings;

  if (loa_feat_param->data_file_format ==
      _SPHERE_OBSTACLE_CENTER_RADIUS_FILE_FORMAT_) {
    if (rt_assert(loa_data_io.extractObstacleAvoidanceTrainingDataSphObs(
            training_data_directory, *input_demo_group_set_obs_avoid_global,
            *obs_sph_centers_cart_position_global, *obs_sph_radius,
            N_demo_settings, max_num_trajs_per_setting,
            selected_obs_avoid_setting_numbers)) == false) {
      return false;
    }
  } else if (loa_feat_param->data_file_format ==
             _VICON_MARKERS_DATA_FILE_FORMAT_) {
    if (rt_assert(loa_data_io.extractObstacleAvoidanceTrainingDataVicon(
            training_data_directory, *input_demo_group_set_obs_avoid_global,
            *vector_point_obstacles_cart_position_global, N_demo_settings,
            max_num_trajs_per_setting, selected_obs_avoid_setting_numbers)) ==
        false) {
      return false;
    }
  } else {  // if loa_feat_param->data_file_format is NOT recognized ...
    return false;
  }

  char var_in_data_dir_path[1000];

  uint N_feature_matrix_rows = 0;
  // count total number of demonstrated trajectories per each obstacle setting:
  for (uint n = 0; n < N_demo_settings; n++) {
    for (uint i = 0; i < (*input_demo_group_set_obs_avoid_global)[n].size();
         i++) {
      N_feature_matrix_rows +=
          (*input_demo_group_set_obs_avoid_global)[n][i].size();
    }
  }

  if (is_learning_baseline) {
    char baseline_out_data_dir_path[1000];
    snprintf(baseline_out_data_dir_path, sizeof(baseline_out_data_dir_path),
             "%s/baseline/", data_directory_path);

    // Load the set of Cartesian trajectories without obstacle (baseline):
    snprintf(var_in_data_dir_path, sizeof(var_in_data_dir_path),
             "%s/baseline/endeff_trajs/", training_data_directory);
    // Learn the BASELINE Cartesian trajectory (without obstacle):
    if (rt_assert(cart_dmp_baseline_ptr->learn(
            var_in_data_dir_path, robot_task_servo_rate)) == false) {
      return false;
    }

    if (is_logging_learning_data) {
      if (rt_assert(cart_dmp_baseline_ptr->saveParams(
              baseline_out_data_dir_path, "f_baseline_weights_matrix.txt",
              "f_baseline_A_learn_matrix.txt",
              "baseline_mean_start_position.txt",
              "baseline_mean_goal_position.txt", "baseline_mean_tau.txt")) ==
          false) {
        return false;
      }
    }
  }

  // Buffers to avoid memory re-allocations multiple times:
  VectorloaF3Ptr feature_vector_buffer;
  MatrixloaTxF3Ptr sub_feature_matrix;
  Matrix3xTPtr sub_target_coupling_term;
  Matrixloa3TxF1Ptr sub_feature_matrix_Rp_yd_constrained;
  Vectorloa3TPtr sub_target_coupling_term_Rp_yd_constrained;

  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, feature_vector_buffer,
                                            loa_feat_param->feature_vector_size,
                                            1)) == false) {
    return false;
  }

  if (rt_assert(allocateMemoryIfNonRealTime(
          is_real_time, sub_feature_matrix, MAX_TRAJ_SIZE,
          loa_feat_param->feature_vector_size)) == false) {
    return false;
  }

  if (rt_assert(allocateMemoryIfNonRealTime(
          is_real_time, sub_target_coupling_term, 3, MAX_TRAJ_SIZE)) == false) {
    return false;
  }

  if (is_constraining_Rp_yd_relationship == true) {
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, sub_feature_matrix_Rp_yd_constrained,
            (3 * MAX_TRAJ_SIZE), (loa_feat_param->feature_vector_size / 3))) ==
        false) {
      return false;
    }

    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, sub_target_coupling_term_Rp_yd_constrained,
            (3 * MAX_TRAJ_SIZE), 1)) == false) {
      return false;
    }
  }

  MatrixXxXPtr X;
  MatrixXxXPtr Ct_target;
  MatrixXxXPtr Ct_fit;
  MatrixXxXPtr XTX;
  MatrixXxXPtr R;  // Regularization (Diagonal) Matrix
  MatrixXxXPtr XTX_plus_R;
  MatrixXxXPtr XTX_plus_R_inv;

  if (is_constraining_Rp_yd_relationship == false) {
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, X, N_feature_matrix_rows,
            loa_feat_param->feature_vector_size)) == false) {
      return false;
    }
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, Ct_target, 3, N_feature_matrix_rows)) == false) {
      return false;
    }
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, Ct_fit, 3, N_feature_matrix_rows)) == false) {
      return false;
    }
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, XTX, loa_feat_param->feature_vector_size,
            loa_feat_param->feature_vector_size)) == false) {
      return false;
    }
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, R, loa_feat_param->feature_vector_size,
            loa_feat_param->feature_vector_size)) == false) {
      return false;
    }
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, XTX_plus_R, loa_feat_param->feature_vector_size,
            loa_feat_param->feature_vector_size)) == false) {
      return false;
    }
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, XTX_plus_R_inv, loa_feat_param->feature_vector_size,
            loa_feat_param->feature_vector_size)) == false) {
      return false;
    }
  } else {  // if (is_constraining_Rp_yd_relationship == true)
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, X, 3 * N_feature_matrix_rows,
            loa_feat_param->feature_vector_size / 3)) == false) {
      return false;
    }
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, Ct_target, 1, 3 * N_feature_matrix_rows)) == false) {
      return false;
    }
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, Ct_fit, 1, 3 * N_feature_matrix_rows)) == false) {
      return false;
    }
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, XTX, loa_feat_param->feature_vector_size / 3,
            loa_feat_param->feature_vector_size / 3)) == false) {
      return false;
    }
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, R, loa_feat_param->feature_vector_size / 3,
            loa_feat_param->feature_vector_size / 3)) == false) {
      return false;
    }
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, XTX_plus_R, loa_feat_param->feature_vector_size / 3,
            loa_feat_param->feature_vector_size / 3)) == false) {
      return false;
    }
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, XTX_plus_R_inv,
            loa_feat_param->feature_vector_size / 3,
            loa_feat_param->feature_vector_size / 3)) == false) {
      return false;
    }
  }

  // Set the critical states (start position and goal position):
  Trajectory critical_states(2);
  critical_states[0] =
      DMPState(cart_dmp_baseline_ptr->getMeanStartPosition(), rt_assertor);
  critical_states[1] =
      DMPState(cart_dmp_baseline_ptr->getMeanGoalPosition(), rt_assertor);

  uint last_feature_matrix_row = 0;

  // Iterate over different demonstration settings of the obstacle position:
  for (uint n = 1; n <= N_demo_settings; n++) {
    if (loa_feat_param->data_file_format == _VICON_MARKERS_DATA_FILE_FORMAT_) {
      if (rt_assert(
              computePointObstaclesCartStateGlobalTrajectoryFromStaticViconData(
                  (*vector_point_obstacles_cart_position_global)[n - 1],
                  (*point_obss_cart_state_traj_global))) == false) {
        return false;
      }
    }

    for (uint i = 1;
         i <= (*input_demo_group_set_obs_avoid_global)[n - 1].size(); i++) {
      if (selected_obs_avoid_setting_numbers == NULL)  // use all settings
      {
        std::cout << "Building Feature Matrix from Demo Setting #" << n << "/"
                  << N_demo_settings << ", Demo Number #" << i << "/"
                  << (*input_demo_group_set_obs_avoid_global)[n - 1].size()
                  << " ..." << std::endl;
      } else {
        std::cout << "Building Feature Matrix from Demo Setting #"
                  << (*selected_obs_avoid_setting_numbers)[n - 1] << " (" << n
                  << "/" << N_demo_settings << "), Demo Number #" << i << "/"
                  << (*input_demo_group_set_obs_avoid_global)[n - 1].size()
                  << " ..." << std::endl;
      }

      uint traj_length =
          (*input_demo_group_set_obs_avoid_global)[n - 1][i - 1].size();

      sub_feature_matrix->resize(traj_length,
                                 loa_feat_param->feature_vector_size);
      sub_target_coupling_term->resize(3, traj_length);

      Matrixloa3TxF1* sub_feature_matrix_Rp_yd_constrained_ptr = NULL;
      Vectorloa3T* sub_target_coupling_term_Rp_yd_constrained_ptr = NULL;

      if (is_constraining_Rp_yd_relationship == true) {
        sub_feature_matrix_Rp_yd_constrained->resize(
            (3 * traj_length), (loa_feat_param->feature_vector_size / 3));
        sub_target_coupling_term_Rp_yd_constrained->resize((3 * traj_length),
                                                           1);

        sub_feature_matrix_Rp_yd_constrained_ptr =
            sub_feature_matrix_Rp_yd_constrained.get();
        sub_target_coupling_term_Rp_yd_constrained_ptr =
            sub_target_coupling_term_Rp_yd_constrained.get();
      }

      if (loa_feat_param->data_file_format ==
          _SPHERE_OBSTACLE_CENTER_RADIUS_FILE_FORMAT_) {
        point_obss_cart_state_traj_global->resize(traj_length);

        if (rt_assert(
                computePointObstaclesCartStateGlobalTrajectoryFromStaticSphereObstacle(
                    (*input_demo_group_set_obs_avoid_global)[n - 1][i - 1],
                    DMPState((*obs_sph_centers_cart_position_global)[n - 1],
                             ZeroVector3, ZeroVector3, rt_assertor),
                    (*obs_sph_radius)[n - 1],
                    (*point_obss_cart_state_traj_global))) == false) {
          return false;
        }
      }

      if (rt_assert(computeSubFeatureMatrixAndSubTargetCouplingTerm(
              (*input_demo_group_set_obs_avoid_global)[n - 1][i - 1],
              (*point_obss_cart_state_traj_global), robot_task_servo_rate,
              critical_states, (*cart_dmp_baseline_ptr),
              (*feature_vector_buffer), (*sub_feature_matrix),
              (*sub_target_coupling_term),
              sub_feature_matrix_Rp_yd_constrained_ptr,
              sub_target_coupling_term_Rp_yd_constrained_ptr,
              max_critical_point_distance_baseline_vs_obs_avoid_demo)) ==
          false) {
        return false;
      }

      if (is_constraining_Rp_yd_relationship == false) {
        X->block(last_feature_matrix_row, 0, traj_length,
                 loa_feat_param->feature_vector_size) = *sub_feature_matrix;

        // Store the computed target Ct (Coupling Term):
        Ct_target->block(0, last_feature_matrix_row, 3, traj_length) =
            (*sub_target_coupling_term);
      } else {  // if (is_constraining_Rp_yd_relationship == true)
        X->block(3 * last_feature_matrix_row, 0, 3 * traj_length,
                 loa_feat_param->feature_vector_size / 3) =
            *sub_feature_matrix_Rp_yd_constrained_ptr;

        // Store the computed target Ct (Coupling Term):
        Ct_target->block(0, 3 * last_feature_matrix_row, 1, 3 * traj_length) =
            sub_target_coupling_term_Rp_yd_constrained_ptr->transpose();
      }

      last_feature_matrix_row += traj_length;
    }
  }
  if (rt_assert(last_feature_matrix_row == N_feature_matrix_rows) == false) {
    return false;
  }
  std::cout << "Done" << std::endl;

  // Computed feature weights of the Ct (Coupling Term):
  *XTX = X->transpose() * (*X);
  // regularization_const now become an input argument to this
  // learnCouplingTerm() function, here are some of the previous values ever
  // used:
  //    regularization_const        = 5.0;    // Best for Schaal's DMP 2nd order
  //    CS (on static obs, multi-demo-settings) regularization_const        =
  //    1e-2;   // OK for Hoffmann's DMP 2nd order CS, NOT OK for Schaal's DMP
  //    2nd order CS (on static obs, multi-demo-settings) regularization_const
  //    = 1e-3;
  *R = regularization_const * EyeMatrixXxX(XTX->rows(), XTX->cols());
  *XTX_plus_R = (*XTX) + (*R);

  std::cout << "Performing Moore-Penrose Inverse ..." << std::endl;
  if (rt_assert(pseudoInverse(*XTX_plus_R, *XTX_plus_R_inv)) == false) {
    return false;
  }
  std::cout << "Done" << std::endl;

  if (is_constraining_Rp_yd_relationship == false) {
    *(loa_feat_param->weights) = (*Ct_target) * (*X) * (*XTX_plus_R_inv);

    // Compute fitted Ct (Coupling Term):
    *Ct_fit = *(loa_feat_param->weights) * X->transpose();
  } else {  // if (is_constraining_Rp_yd_relationship == true)
    *(loa_feat_param->weights_Rp_yd_constrained) =
        (*Ct_target) * (*X) * (*XTX_plus_R_inv);

    // Compute fitted Ct (Coupling Term):
    *Ct_fit = *(loa_feat_param->weights_Rp_yd_constrained) * X->transpose();
  }

  std::cout << "Logging Learning Obstacle Avoidance Data..." << std::endl;
  if (is_logging_learning_data) {
    if (is_constraining_Rp_yd_relationship == false) {
      if (rt_assert(loa_data_io.writeMatrixToFile(
              data_directory_path, "learn_obs_avoid_weights_matrix.txt",
              *(loa_feat_param->weights))) == false) {
        return false;
      }
      if (rt_assert(loa_data_io.writeMatrixToFile(
              data_directory_path,
              "learn_obs_avoid_weights_transpose_matrix.txt",
              loa_feat_param->weights->transpose())) == false) {
        return false;
      }
    } else {  // if (is_constraining_Rp_yd_relationship == true)
      if (rt_assert(loa_data_io.writeMatrixToFile(
              data_directory_path, "learn_obs_avoid_weights_matrix.txt",
              *(loa_feat_param->weights_Rp_yd_constrained))) == false) {
        return false;
      }
      if (rt_assert(loa_data_io.writeMatrixToFile(
              data_directory_path,
              "learn_obs_avoid_weights_transpose_matrix.txt",
              loa_feat_param->weights_Rp_yd_constrained->transpose())) ==
          false) {
        return false;
      }
    }

    if (rt_assert(loa_data_io.writeMatrixToFile(data_directory_path, "X.txt",
                                                *X)) == false) {
      return false;
    }

    if (rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "Ct_target.txt", Ct_target->transpose())) ==
        false) {
      return false;
    }

    if (rt_assert(loa_data_io.writeMatrixToFile(
            data_directory_path, "Ct_fit.txt", Ct_fit->transpose())) == false) {
      return false;
    }
  }
  std::cout << "Done" << std::endl;

  return true;
}

/**
 * NON-REAL-TIME!!!\n
 * If you want to maintain real-time performance,
 * do/call this function from a separate (non-real-time) thread!!!\n
 * Compute the sub-feature-matrix and the sub-target DMP coupling term (as a
 * sub-block of a matrix) from a single demonstrated obstacle avoidance
 * trajectory.\n This will later be stacked together with those from other
 * demonstrations, to become a huge feature matrix (usually called X matrix) and
 * \n a huge target DMP coupling term (target regression variable, usually a
 * vector).
 *
 * @param demo_obs_avoid_traj_global A demonstrated obstacle avoidance
 * trajectory for a particular obstacle setting, in Cartesian global coordinate
 * system
 * @param point_obstacles_cart_state_traj_global Trajectory or vector of vectors
 * of \n point obstacles' Cartesian state \n in the global coordinate system
 * @param robot_task_servo_rate Robot's task servo rate
 * @param cart_coord_dmp_critical_states Critical states for unrolling the
 * Cartesian Coordinate DMP
 * @param cart_coord_dmp Cartesian Coordinate DMP that will be used to extract
 * the sub-feature-matrix and the target coupling term
 * @param feature_vector_buffer Buffer for the feature vector, to be transferred
 * to the sub-feature-matrix (so that we do not have to allocate memory in this
 * function)
 * @param sub_feature_matrix The resulting sub-feature-matrix (to be returned)
 * @param sub_target_coupling_term The resulting target obstacle avoidance
 * coupling term (as a sub-block of a matrix) (to be returned)
 * @param sub_feature_matrix_Rp_yd_constrained The resulting sub-feature-matrix
 * when Rp*yd relationship is constrained (to be returned)
 * @param sub_target_coupling_term_Rp_yd_constrained The resulting target
 * obstacle avoidance coupling term when Rp*yd relationship is constrained (to
 * be returned)
 * @param max_critical_point_distance_baseline_vs_obs_avoid_demo [optional]
 * Maximum tolerable distance (in meter) \n between start position in baseline
 * and obstacle avoidance demonstrations,\n as well as between goal position in
 * baseline and obstacle avoidance demonstrations.
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::
    computeSubFeatureMatrixAndSubTargetCouplingTerm(
        const Trajectory& demo_obs_avoid_traj_global,
        const ObstacleStatesVector& point_obstacles_cart_state_traj_global,
        const double& robot_task_servo_rate,
        const Trajectory& cart_coord_dmp_critical_states,
        CartesianCoordDMP& cart_coord_dmp, VectorloaF3& feature_vector_buffer,
        MatrixloaTxF3& sub_feature_matrix, Matrix3xT& sub_target_coupling_term,
        Matrixloa3TxF1* sub_feature_matrix_Rp_yd_constrained,
        Vectorloa3T* sub_target_coupling_term_Rp_yd_constrained,
        double max_critical_point_distance_baseline_vs_obs_avoid_demo) {
  // input checking
  if (rt_assert((point_obstacles_cart_state_traj_global.size() ==
                 demo_obs_avoid_traj_global.size()) ||
                (point_obstacles_cart_state_traj_global.size() == 1)) ==
      false) {
    return false;
  }
  if (rt_assert((robot_task_servo_rate > 0.0) &&
                (robot_task_servo_rate <= MAX_TASK_SERVO_RATE)) == false) {
    return false;
  }
  if (rt_assert(cart_coord_dmp_critical_states.size() >= 2) == false) {
    return false;
  }
  if (is_constraining_Rp_yd_relationship == true) {
    if (rt_assert((sub_feature_matrix_Rp_yd_constrained != NULL) &&
                  (sub_target_coupling_term_Rp_yd_constrained != NULL)) ==
        false) {
      return false;
    }
  }

  uint traj_length = demo_obs_avoid_traj_global.size();

  if (feature_vector_buffer.rows() != loa_feat_param->feature_vector_size) {
    feature_vector_buffer.resize(loa_feat_param->feature_vector_size, 1);
  }

  if (is_constraining_Rp_yd_relationship == true) {
    if ((sub_feature_matrix_Rp_yd_constrained->rows() != (3 * traj_length)) ||
        (sub_feature_matrix_Rp_yd_constrained->cols() !=
         (loa_feat_param->feature_vector_size / 3))) {
      sub_feature_matrix_Rp_yd_constrained->resize(
          (3 * traj_length), (loa_feat_param->feature_vector_size / 3));
    }

    if ((sub_target_coupling_term_Rp_yd_constrained->rows() !=
         (3 * traj_length)) ||
        (sub_target_coupling_term_Rp_yd_constrained->cols() != 1)) {
      sub_target_coupling_term_Rp_yd_constrained->resize((3 * traj_length), 1);
    }
  } else {  // if (is_constraining_Rp_yd_relationship == false)
    if ((sub_feature_matrix.rows() != traj_length) ||
        (sub_feature_matrix.cols() != loa_feat_param->feature_vector_size)) {
      sub_feature_matrix.resize(traj_length,
                                loa_feat_param->feature_vector_size);
    }

    if ((sub_target_coupling_term.rows() != 3) ||
        (sub_target_coupling_term.cols() != traj_length)) {
      sub_target_coupling_term.resize(3, traj_length);
    }
  }

  double dt;
  Vector3 start_position_baseline_dmp =
      cart_coord_dmp_critical_states[0].getX();
  Vector3 goal_position_baseline_dmp =
      cart_coord_dmp_critical_states[cart_coord_dmp_critical_states.size() - 1]
          .getX();
  Vector3 start_position_obs_avoid_demo = demo_obs_avoid_traj_global[0].getX();
  Vector3 goal_position_obs_avoid_demo =
      demo_obs_avoid_traj_global[demo_obs_avoid_traj_global.size() - 1].getX();
  Vector3 diff_start_position =
      start_position_obs_avoid_demo - start_position_baseline_dmp;
  Vector3 diff_goal_position =
      goal_position_obs_avoid_demo - goal_position_baseline_dmp;

  // Error checking on the demonstration:
  // (distance between start position of baseline DMP and obstacle avoidance
  // demonstration,
  //  as well as distance between goal position of baseline DMP and obstacle
  //  avoidance demonstration, both should be lower than max_tolerable_distance,
  //  otherwise the demonstrated obstacle avoidance trajectory is flawed):
  if (rt_assert((diff_start_position.norm() <=
                 max_critical_point_distance_baseline_vs_obs_avoid_demo) &&
                (diff_goal_position.norm() <=
                 max_critical_point_distance_baseline_vs_obs_avoid_demo)) ==
      false) {
    return false;
  }

  // Start the with-obstacle/obstacle-coupled DMP (here we are using the
  // critical states, i.e. start and goal positions, of the demonstrated
  // baseline trajectory): Compute DMP unrolling parameters:
  DMPUnrollInitParams demo_obs_avoid_traj_parameters(demo_obs_avoid_traj_global,
                                                     rt_assertor);
  DMPUnrollInitParams dmp_unroll_init_parameters(
      demo_obs_avoid_traj_parameters.tau, cart_coord_dmp_critical_states,
      rt_assertor);
  if (rt_assert(cart_coord_dmp.start(dmp_unroll_init_parameters)) == false) {
    return false;
  }
  Matrix4x4 cart_hmg_transform_global_to_local_matrix =
      cart_coord_dmp.getHomogeneousTransformMatrixGlobalToLocal();

  VectorN current_ct_target(dmp_num_dimensions);
  current_ct_target = ZeroVectorN(dmp_num_dimensions);

  DMPState current_state_demo_obs_avoid_global(3, rt_assertor);
  DMPState current_state_demo_obs_avoid_local(3, rt_assertor);

  ObstacleStates point_obstacles_cart_state_global;
  ObstacleStates point_obstacles_cart_state_local;

  // Iterate over DMP state sequence
  for (uint j = 0; j <= traj_length; j++) {
    // Based on the reconstruction experiment of the synthetic obstacle
    // avoidance demonstration data, the input states for computing feature
    // matrix X need to be delayed one time-step in order to get a "perfect"
    // reconstruction, while the target coupling term Ct_target is NOT delayed:
    if (j == 0) {
      current_state_demo_obs_avoid_global = demo_obs_avoid_traj_global[j];
      if (point_obstacles_cart_state_traj_global.size() ==
          demo_obs_avoid_traj_global.size()) {
        point_obstacles_cart_state_global =
            point_obstacles_cart_state_traj_global[j];
      } else if (point_obstacles_cart_state_traj_global.size() == 1) {
        point_obstacles_cart_state_global =
            point_obstacles_cart_state_traj_global[0];
      }
    } else {
      current_state_demo_obs_avoid_global = demo_obs_avoid_traj_global[j - 1];
      if (point_obstacles_cart_state_traj_global.size() ==
          demo_obs_avoid_traj_global.size()) {
        point_obstacles_cart_state_global =
            point_obstacles_cart_state_traj_global[j - 1];
      } else if (point_obstacles_cart_state_traj_global.size() == 1) {
        point_obstacles_cart_state_global =
            point_obstacles_cart_state_traj_global[0];
      }
    }
    point_obstacles_cart_state_local.resize(
        point_obstacles_cart_state_global.size());

    // Remember that demo_obs_avoid_traj_global is demonstration in the global
    // coordinate system, thus we need to convert them into its representation
    // in DMP's local coordinate system:
    if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(
            current_state_demo_obs_avoid_global,
            cart_hmg_transform_global_to_local_matrix,
            current_state_demo_obs_avoid_local)) == false) {
      return false;
    }

    if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(
            point_obstacles_cart_state_global,
            cart_hmg_transform_global_to_local_matrix,
            point_obstacles_cart_state_local)) == false) {
      return false;
    }

    if (j == 0) {
      dt = demo_obs_avoid_traj_global[1].getTime() -
           demo_obs_avoid_traj_global[0].getTime();
    } else if (j == traj_length) {
      // Compute time difference between old state and new state
      dt = demo_obs_avoid_traj_global[j - 1].getTime() -
           demo_obs_avoid_traj_global[j - 2].getTime();
    } else {
      // Compute time difference between old state and new state
      dt = demo_obs_avoid_traj_global[j].getTime() -
           demo_obs_avoid_traj_global[j - 1].getTime();
    }
    if (dt == 0.0) {
      dt = 1.0 / robot_task_servo_rate;
    }

    // Based on the reconstruction experiment of the synthetic obstacle
    // avoidance demonstration data, the input states for computing feature
    // matrix X need to be delayed one time-step in order to get a "perfect"
    // reconstruction, while the target coupling term Ct_target is NOT delayed:
    if (j < traj_length) {
      if (rt_assert(TransformCouplingLearnObsAvoid::computeFeatures(
              current_state_demo_obs_avoid_local,
              point_obstacles_cart_state_local, feature_vector_buffer,
              loa_feat_param->cart_coord_loa_feature_matrix.get())) == false) {
        return false;
      }

      if (is_constraining_Rp_yd_relationship == true) {
        sub_feature_matrix_Rp_yd_constrained->block(
            3 * j, 0, 3, (loa_feat_param->feature_vector_size / 3)) =
            loa_feat_param->cart_coord_loa_feature_matrix->transpose();
      } else {  // if (is_constraining_Rp_yd_relationship == false)
        sub_feature_matrix.row(j) = feature_vector_buffer.transpose();
      }
    }

    if (j > 0) {
      // Compute target coupling term value for this state and update canonical
      // and goal states:
      if (rt_assert(cart_coord_dmp.getTargetCouplingTermAndUpdateStates(
              current_state_demo_obs_avoid_local, dt, current_ct_target)) ==
          false) {
        return false;
      }

      if (is_constraining_Rp_yd_relationship == true) {
        sub_target_coupling_term_Rp_yd_constrained->block(
            3 * (j - 1), 0, 3, 1) = current_ct_target.block(0, 0, 3, 1);
      } else {  // if (is_constraining_Rp_yd_relationship == false)
        sub_target_coupling_term.col(j - 1) = current_ct_target;
      }
    }
  }

  return true;
}

/**
 * This is for computing the stacked feature vector Phi, such that: \n
 * Ct = Gamma^T * Phi , where Gamma is the weight vector.\n
 * Inputs are represented in the global coordinate system.
 *
 * @param endeff_cart_state_global The end-effector Cartesian state in the
 * global coordinate system
 * @param point_obstacles_cart_state_global Vector of point obstacles' Cartesian
 * state in the global coordinate system
 * @param cart_hmg_transform_global_to_local_matrix The Cartesian homogeneous
 * transformation matrix from global to local coordinate system
 * @param feature_vector The resulting stacked feature vector (Phi) (to be
 * returned)
 * @param feature_matrix [optional] The original (unstacked) feature matrix (to
 * be returned)
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::computeFeatures(
    const DMPState& endeff_cart_state_global,
    const ObstacleStates& point_obstacles_cart_state_global,
    const Matrix4x4& cart_hmg_transform_global_to_local_matrix,
    VectorloaF3& feature_vector, MatrixloaF1x3* feature_matrix) {
  // pre-conditions checking (needed for real-time guarantee of this
  // function!!!)
  if (rt_assert(point_obstacles_cartesian_state_local->capacity() >=
                point_obstacles_cart_state_global.size()) == false) {
    return false;
  }
  point_obstacles_cartesian_state_local->resize(
      point_obstacles_cart_state_global.size());

  if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(
          endeff_cart_state_global, cart_hmg_transform_global_to_local_matrix,
          *endeff_cartesian_state_local)) == false) {
    return false;
  }

  if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(
          point_obstacles_cart_state_global,
          cart_hmg_transform_global_to_local_matrix,
          *point_obstacles_cartesian_state_local)) == false) {
    return false;
  }

  return (rt_assert(TransformCouplingLearnObsAvoid::computeFeatures(
      *endeff_cartesian_state_local, *point_obstacles_cartesian_state_local,
      feature_vector, feature_matrix)));
}

/**
 * This is for computing the stacked feature vector Phi, such that: \n
 * Ct = Gamma^T * Phi , where Gamma is the weight vector.\n
 * Inputs are represented in the local coordinate system.
 *
 * @param endeff_cart_state_local The end-effector Cartesian state in the local
 * coordinate system
 * @param point_obstacles_cart_state_local Vector of point obstacles' Cartesian
 * state in the local coordinate system
 * @param feature_vector The resulting stacked feature vector (Phi) (to be
 * returned)
 * @param feature_matrix [optional] The original (unstacked) feature matrix (to
 * be returned)
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::computeFeatures(
    const DMPState& endeff_cart_state_local,
    const ObstacleStates& point_obstacles_cart_state_local,
    VectorloaF3& feature_vector, MatrixloaF1x3* feature_matrix) {
  // pre-conditions checking
  if (rt_assert(TransformCouplingLearnObsAvoid::isValid()) == false) {
    return false;
  }

  double tau;
  if (rt_assert(tau_sys->getTauRelative(tau)) == false) {
    return false;
  }

  if (loa_feat_param->is_using_NN == false) {
    if (loa_feat_param->is_using_AF_H14_feature) {
      if (loa_feat_param->AF_H14_num_indep_feat_obs_points > 1) {
        // the following for-loop is designed for extensibility to the case of
        // many evaluation points (e.g. the sphere obstacle's center point could
        // be one of these evaluation points)
        for (uint i = 0; i < loa_feat_param->AF_H14_num_indep_feat_obs_points;
             i++) {
          if (i == 0) {
            // for obstacle center point:
            if (rt_assert(
                    TransformCouplingLearnObsAvoid::
                        computeAF_H14ObsAvoidCtFeatureMatricesPerObsPoint(
                            endeff_cart_state_local,
                            point_obstacles_cart_state_local[i],
                            (*(loa_feat_param
                                   ->AF_H14_loa_feature_matrix_phi1_phi2_per_obs_point)),
                            (loa_feat_param
                                 ->AF_H14_loa_feature_matrix_phi3_per_obs_point
                                 .get()))) == false) {
              return false;
            }

            if (loa_feat_param->AF_H14_N_k_phi3_grid > 0) {
              loa_feat_param->cart_coord_loa_feature_matrix->block(
                  (loa_feat_param->AF_H14_feature_matrix_start_idx +
                   (loa_feat_param->AF_H14_num_indep_feat_obs_points *
                    loa_feat_param->AF_H14_beta_phi1_phi2_vector->rows())),
                  0, loa_feat_param->AF_H14_k_phi3_vector->rows(), 3) =
                  loa_feat_param->AF_H14_loa_feature_matrix_phi3_per_obs_point
                      ->transpose();
            }
          } else {
            if (rt_assert(
                    TransformCouplingLearnObsAvoid::
                        computeAF_H14ObsAvoidCtFeatureMatricesPerObsPoint(
                            endeff_cart_state_local,
                            point_obstacles_cart_state_local[i],
                            (*(loa_feat_param
                                   ->AF_H14_loa_feature_matrix_phi1_phi2_per_obs_point)))) ==
                false) {
              return false;
            }
          }

          loa_feat_param->cart_coord_loa_feature_matrix->block(
              (loa_feat_param->AF_H14_feature_matrix_start_idx +
               (i * loa_feat_param->AF_H14_beta_phi1_phi2_vector->rows())),
              0, loa_feat_param->AF_H14_beta_phi1_phi2_vector->rows(), 3) =
              loa_feat_param->AF_H14_loa_feature_matrix_phi1_phi2_per_obs_point
                  ->transpose();
        }
      } else if (loa_feat_param->AF_H14_num_indep_feat_obs_points == 1) {
        loa_feat_param->cart_coord_loa_feature_matrix->block(
            loa_feat_param->AF_H14_feature_matrix_start_idx, 0,
            loa_feat_param->AF_H14_feature_matrix_rows, 3) =
            ZeroMatrixloaF1x3(loa_feat_param->AF_H14_feature_matrix_rows);

        // the following for-loop is designed for extensibility to the case of
        // many evaluation points (e.g. the sphere obstacle's center point could
        // be one of these evaluation points)
        for (uint i = 0; i < point_obstacles_cart_state_local.size(); i++) {
          if (rt_assert(
                  TransformCouplingLearnObsAvoid::
                      computeAF_H14ObsAvoidCtFeatureMatricesPerObsPoint(
                          endeff_cart_state_local,
                          point_obstacles_cart_state_local[i],
                          (*(loa_feat_param
                                 ->AF_H14_loa_feature_matrix_phi1_phi2_per_obs_point)),
                          (loa_feat_param
                               ->AF_H14_loa_feature_matrix_phi3_per_obs_point
                               .get()))) == false) {
            return false;
          }

          loa_feat_param->cart_coord_loa_feature_matrix->block(
              loa_feat_param->AF_H14_feature_matrix_start_idx, 0,
              loa_feat_param->AF_H14_beta_phi1_phi2_vector->rows(), 3) +=
              loa_feat_param->AF_H14_loa_feature_matrix_phi1_phi2_per_obs_point
                  ->transpose();

          if (loa_feat_param->AF_H14_N_k_phi3_grid > 0) {
            loa_feat_param->cart_coord_loa_feature_matrix->block(
                (loa_feat_param->AF_H14_feature_matrix_start_idx +
                 (loa_feat_param->AF_H14_num_indep_feat_obs_points *
                  loa_feat_param->AF_H14_beta_phi1_phi2_vector->rows())),
                0, loa_feat_param->AF_H14_k_phi3_vector->rows(), 3) +=
                loa_feat_param->AF_H14_loa_feature_matrix_phi3_per_obs_point
                    ->transpose();
          }
        }
        loa_feat_param->cart_coord_loa_feature_matrix->block(
            loa_feat_param->AF_H14_feature_matrix_start_idx, 0,
            loa_feat_param->AF_H14_feature_matrix_rows, 3) *=
            (1.0 / point_obstacles_cart_state_local.size());
      } else  // if (loa_feat_param->AF_H14_num_indep_feat_obs_points <= 0)
      {
        return false;
      }
    }

    if (loa_feat_param->is_using_PF_DYN2_feature) {
      loa_feat_param->cart_coord_loa_feature_matrix->block(
          loa_feat_param->PF_DYN2_feature_matrix_start_idx, 0,
          loa_feat_param->PF_DYN2_feature_matrix_rows, 3) =
          ZeroMatrixloaF1x3(loa_feat_param->PF_DYN2_feature_matrix_rows);

      // the following for-loop is designed for extensibility to the case of
      // many evaluation points (e.g. the sphere obstacle's center point could
      // be one of these evaluation points)
      for (uint i = 0; i < point_obstacles_cart_state_local.size(); i++) {
        if (rt_assert(
                TransformCouplingLearnObsAvoid::
                    computePF_DYN2ObsAvoidCtFeatureMatricesPerObsPoint(
                        endeff_cart_state_local,
                        point_obstacles_cart_state_local[i],
                        (*(loa_feat_param
                               ->PF_DYN2_loa_feature_matrix_per_obs_point)))) ==
            false) {
          return false;
        }

        loa_feat_param->cart_coord_loa_feature_matrix->block(
            loa_feat_param->PF_DYN2_feature_matrix_start_idx, 0,
            loa_feat_param->PF_DYN2_feature_matrix_rows, 3) +=
            loa_feat_param->PF_DYN2_loa_feature_matrix_per_obs_point
                ->transpose();
      }
      loa_feat_param->cart_coord_loa_feature_matrix->block(
          loa_feat_param->PF_DYN2_feature_matrix_start_idx, 0,
          loa_feat_param->PF_DYN2_feature_matrix_rows, 3) *=
          (1.0 / point_obstacles_cart_state_local.size());
    }

    if (loa_feat_param->is_using_KGF_feature) {
      loa_feat_param->cart_coord_loa_feature_matrix->block(
          loa_feat_param->KGF_feature_matrix_start_idx, 0,
          loa_feat_param->KGF_feature_matrix_rows, 3) =
          ZeroMatrixloaF1x3(loa_feat_param->KGF_feature_matrix_rows);

      // the following for-loop is designed for extensibility to the case of
      // many evaluation points (e.g. the sphere obstacle's center point could
      // be one of these evaluation points)
      for (uint i = 0; i < point_obstacles_cart_state_local.size(); i++) {
        if (rt_assert(
                TransformCouplingLearnObsAvoid::
                    computeKGFObsAvoidCtFeatureMatricesPerObsPoint(
                        endeff_cart_state_local,
                        point_obstacles_cart_state_local[i],
                        (*(loa_feat_param
                               ->KGF_loa_feature_matrix_per_obs_point)))) ==
            false) {
          return false;
        }

        loa_feat_param->cart_coord_loa_feature_matrix->block(
            loa_feat_param->KGF_feature_matrix_start_idx, 0,
            loa_feat_param->KGF_feature_matrix_rows, 3) +=
            loa_feat_param->KGF_loa_feature_matrix_per_obs_point->transpose();
      }
      loa_feat_param->cart_coord_loa_feature_matrix->block(
          loa_feat_param->KGF_feature_matrix_start_idx, 0,
          loa_feat_param->KGF_feature_matrix_rows, 3) *=
          (1.0 / point_obstacles_cart_state_local.size());
    }

    // if NOT using bias term to construct the coupling term (i.e. assuming the
    // coupling term is zero-mean, which makes sense, given the nature of all
    // coupling terms in DMP):
    feature_vector = Eigen::Map<MatrixXxX>(
        loa_feat_param->cart_coord_loa_feature_matrix->data(),
        loa_feat_param->feature_vector_size, 1);
  } else {  // if (loa_feat_param->is_using_NN == true)
    if (rt_assert(TransformCouplingLearnObsAvoid::computeNNObsAvoidCtFeature(
            endeff_cart_state_local, point_obstacles_cart_state_local,
            feature_vector)) == false) {
      return false;
    }
  }

  if (rt_assert(containsNaN(feature_vector) == false) == false) {
    return false;
  }

  if (feature_matrix != NULL) {
    *feature_matrix = *(loa_feat_param->cart_coord_loa_feature_matrix);
  }

  if (is_collecting_trajectory_data) {
    // (log only the first point obstacle Cartesian state in local coordinate
    // system,
    //  as a representative of the obstacle)
    if (rt_assert(loa_data_io.collectTrajectoryDataSet(
            feature_vector, endeff_cart_state_local,
            point_obstacles_cart_state_local[0])) == false) {
      return false;
    }
  }

  return true;
}

/**
 * In Rai et al. (Humanoids 2014) paper, this is for computing \n
 * the sub-feature vector (R*yd*phi), i.e. either (R*yd*phi1) or (R_p*yd*phi2),
 * \n which is needed for constructing the stacked feature vector Phi in
 * computeFeatures() function.\n (AF_H14 = Akshara-Franzi Humanoids'14: Learning
 * Obstacle Avoidance Coupling Term paper)
 *
 * @param endeff_cart_state_local The end-effector Cartesian state in local
 * coordinate system
 * @param obs_evalpt_cart_state_local The Cartesian state (in local coordinate
 * system) of an evaluation point inside/on the obstacle geometry
 * @param feature_matrix_phi1_or_phi2_per_point The computed feature matrix
 * (R*yd*phi1) or (R*yd*phi2) for the specified obstacle point (to be returned)
 * @param feature_matrix_phi3_per_point [optional] The computed feature matrix
 * (R*yd*phi3) for the specified obstacle point (to be returned)
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::
    computeAF_H14ObsAvoidCtFeatureMatricesPerObsPoint(
        const DMPState& endeff_cart_state_local,
        const DMPState& obs_evalpt_cart_state_local,
        Matrixloa3xG2& feature_matrix_phi1_or_phi2_per_point,
        Matrixloa3xG* feature_matrix_phi3_per_point) {
  double tau;
  if (rt_assert(tau_sys->getTauRelative(tau)) == false) {
    return false;
  }

  Vector3 endeff_cart_position_local =
      endeff_cart_state_local.getX().block(0, 0, 3, 1);
  Vector3 endeff_cart_velocity_local =
      endeff_cart_state_local.getXd().block(0, 0, 3, 1);
  Vector3 obs_evalpt_cart_position_local =
      obs_evalpt_cart_state_local.getX().block(0, 0, 3, 1);
  Vector3 obs_evalpt_cart_velocity_local =
      obs_evalpt_cart_state_local.getXd().block(0, 0, 3, 1);

  // p_to_ee_vector is a vector originating from
  // the evaluation point (evalpt) p (a point inside/on the obstacle geometry),
  // and going to endeff_cart_position_local:
  Vector3 p_to_ee_vector =
      endeff_cart_position_local - obs_evalpt_cart_position_local;

  // dp is distance between the evaluation point (evalpt) p and
  // endeff_cart_position_local:
  double d_p = p_to_ee_vector.norm();

  // ee_speed is the (scalar) magnitude of endeff_cart_velocity_local:
  double ee_speed = endeff_cart_velocity_local.norm();

  double theta_p;

  double numerical_threshold = 1e-4;

  if ((d_p >= numerical_threshold) && (ee_speed >= numerical_threshold)) {
    // (taking only positive angles)
    theta_p = fabs(acos((-p_to_ee_vector.dot(endeff_cart_velocity_local)) /
                        (d_p * ee_speed)));
  } else {
    theta_p = 0.0;
  }

  Vector3 rotation_axis_p = (-p_to_ee_vector).cross(endeff_cart_velocity_local);
  double rotation_axis_p_norm = rotation_axis_p.norm();
  Matrix3x3 R_p;  // the rotation matrix for evaluation point p

  if (rotation_axis_p_norm >= numerical_threshold) {
    rotation_axis_p =
        rotation_axis_p / rotation_axis_p_norm;  // normalize vector (required
                                                 // by Eigen's AngleAxisd)

    // compute the rotation matrix:
    R_p = Eigen::AngleAxisd(0.5 * M_PI, rotation_axis_p);
  } else {
    rotation_axis_p = ZeroVector3;
    R_p = ZeroMatrix3x3;
  }

  if (feature_matrix_phi1_or_phi2_per_point.cols() !=
      loa_feat_param->AF_H14_N_beta_phi1_phi2_vector) {
    feature_matrix_phi1_or_phi2_per_point.resize(
        3, loa_feat_param->AF_H14_N_beta_phi1_phi2_vector);
  }

  // original:
  // feature_matrix_phi1_or_phi2_per_point   = (R_p *
  // (endeff_cart_velocity_local - obs_evalpt_cart_velocity_local) *
  // theta_p).replicate(1, loa_feat_param->AF_H14_beta_phi1_phi2_vector.rows());

  // Giovanni's 1st modification (added tau multiplier):
  // feature_matrix_phi1_or_phi2_per_point   = (R_p * (tau *
  // (endeff_cart_velocity_local - obs_evalpt_cart_velocity_local)) *
  // theta_p).replicate(1, loa_feat_param->AF_H14_beta_phi1_phi2_vector.rows());

  // Giovanni's 2nd modification (added tau and (M_PI - theta_p) multiplier):
  feature_matrix_phi1_or_phi2_per_point =
      (R_p *
       (tau * (endeff_cart_velocity_local - obs_evalpt_cart_velocity_local)) *
       (M_PI - theta_p) * theta_p)
          .replicate(1, loa_feat_param->AF_H14_beta_phi1_phi2_vector->rows());

  feature_matrix_phi1_or_phi2_per_point =
      feature_matrix_phi1_or_phi2_per_point.array() *
      (-loa_feat_param->AF_H14_beta_phi1_phi2_vector->transpose() * theta_p)
          .array()
          .exp()
          .replicate(3, 1);
  feature_matrix_phi1_or_phi2_per_point =
      feature_matrix_phi1_or_phi2_per_point.array() *
      (-loa_feat_param->AF_H14_k_phi1_phi2_vector->transpose() * (d_p * d_p))
          .array()
          .exp()
          .replicate(3, 1);

  if (rt_assert(containsNaN(feature_matrix_phi1_or_phi2_per_point) == false) ==
      false) {
    return false;
  }

  if (feature_matrix_phi3_per_point != NULL) {
    if (loa_feat_param->AF_H14_N_k_phi3_grid > 0) {
      if (feature_matrix_phi3_per_point->cols() !=
          loa_feat_param->AF_H14_N_k_phi3_grid) {
        feature_matrix_phi3_per_point->resize(
            3, loa_feat_param->AF_H14_N_k_phi3_grid);
      }

      //(*feature_matrix_phi3_per_point)    =
      // ZeroMatrixloa3xG(loa_feat_param->AF_H14_N_k_phi3_grid);

      // original:
      //(*feature_matrix_phi3_per_point)    = (R_p * (endeff_cart_velocity_local
      //- obs_evalpt_cart_velocity_local)).replicate(1,
      // loa_feat_param->AF_H14_k_phi3_vector->rows());

      // Giovanni's 1st modification (added tau multiplier):
      (*feature_matrix_phi3_per_point) =
          (R_p * (tau * (endeff_cart_velocity_local -
                         obs_evalpt_cart_velocity_local)))
              .replicate(1, loa_feat_param->AF_H14_k_phi3_vector->rows());

      (*feature_matrix_phi3_per_point) =
          feature_matrix_phi3_per_point->array() *
          (-loa_feat_param->AF_H14_k_phi3_vector->transpose() * (d_p * d_p))
              .array()
              .exp()
              .replicate(3, 1);
    }

    if (rt_assert(containsNaN(*feature_matrix_phi3_per_point) == false) ==
        false) {
      return false;
    }
  }

  return true;
}

/**
 * In Park et al. (Humanoids 2008) paper, this is for computing \n
 * an improved version of the dynamic obstacle avoidance model presented there
 * (designed by Giovanni Sutanto), \n i.e. the sub-feature vector, which is
 * needed for constructing the stacked feature vector Phi \n in
 * computeFeatures() function.
 *
 * @param endeff_cart_state_local The end-effector Cartesian state in local
 * coordinate system
 * @param obs_evalpt_cart_state_local The Cartesian state (in local coordinate
 * system) of an evaluation point inside/on the obstacle geometry
 * @param feature_matrix_per_point The computed feature matrix for the specified
 * obstacle point (to be returned)
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::
    computePF_DYN2ObsAvoidCtFeatureMatricesPerObsPoint(
        const DMPState& endeff_cart_state_local,
        const DMPState& obs_evalpt_cart_state_local,
        Matrixloa3xG2& feature_matrix_per_point) {
  double tau;
  if (rt_assert(tau_sys->getTauRelative(tau)) == false) {
    return false;
  }

  Vector3 endeff_cart_position_local =
      endeff_cart_state_local.getX().block(0, 0, 3, 1);
  Vector3 endeff_cart_velocity_local =
      endeff_cart_state_local.getXd().block(0, 0, 3, 1);
  Vector3 obs_evalpt_cart_position_local =
      obs_evalpt_cart_state_local.getX().block(0, 0, 3, 1);
  Vector3 obs_evalpt_cart_velocity_local =
      obs_evalpt_cart_state_local.getXd().block(0, 0, 3, 1);

  Vector3 rel_v3 = endeff_cart_velocity_local - obs_evalpt_cart_velocity_local;
  Vector3 ox3 = obs_evalpt_cart_position_local - endeff_cart_position_local;

  double norm_rel_v3 = rel_v3.norm();
  double norm_ox3 = ox3.norm();
  double rel_v3_dot_ox3 = rel_v3.dot(ox3);
  double ox3_dot_ox3 = ox3.dot(ox3);

  // numerical threshold
  double num_thresh = 1e-5;

  if (feature_matrix_per_point.cols() !=
      loa_feat_param->PF_DYN2_N_beta_vector) {
    feature_matrix_per_point.resize(3, loa_feat_param->PF_DYN2_N_beta_vector);
  }

  if ((norm_rel_v3 >= num_thresh) && (norm_ox3 >= num_thresh)) {
    feature_matrix_per_point =
        (((rel_v3_dot_ox3 / (norm_rel_v3 * (norm_ox3 * norm_ox3 * norm_ox3))) *
          ox3.replicate(1, loa_feat_param->PF_DYN2_beta_vector->rows()))
             .array() *
         (loa_feat_param->PF_DYN2_beta_vector->transpose()
              .replicate(3, 1)
              .array()));
    feature_matrix_per_point =
        feature_matrix_per_point.array() -
        (((1.0 / (norm_rel_v3 * norm_ox3)) *
          rel_v3.replicate(1, loa_feat_param->PF_DYN2_beta_vector->rows()))
             .array() *
         (loa_feat_param->PF_DYN2_beta_vector->transpose()
              .replicate(3, 1)
              .array()))
            .array();
    feature_matrix_per_point =
        feature_matrix_per_point.array() +
        ((2.0 * ox3.replicate(1, loa_feat_param->PF_DYN2_k_vector->rows()))
             .array() *
         (loa_feat_param->PF_DYN2_k_vector->transpose()
              .replicate(3, 1)
              .array()))
            .array();
    feature_matrix_per_point =
        feature_matrix_per_point.array() *
        (((loa_feat_param->PF_DYN2_beta_vector->transpose()
               .replicate(3, 1)
               .array()) *
          (rel_v3_dot_ox3 / (norm_rel_v3 * norm_ox3))) -
         ((loa_feat_param->PF_DYN2_k_vector->transpose()
               .replicate(3, 1)
               .array()) *
          ox3_dot_ox3))
            .array()
            .exp()
            .replicate(3, 1)
            .array();
    feature_matrix_per_point =
        -tau * norm_rel_v3 * feature_matrix_per_point.array();
  } else {
    feature_matrix_per_point =
        ZeroMatrixloa3xG2(loa_feat_param->PF_DYN2_N_beta_vector);
  }

  /*// dp is distance between the obstacle evaluation point (evalpt) p and
  endeff in the local coordinate system: double          d_p             =
  norm_ox3;

  // rel_ee_speed is the (scalar) magnitude of relative endeff velocity with
  respect to the obstacle evaluation point (evalpt) p
  // in the local coordinate system:
  double          rel_ee_speed    = norm_rel_v3;

  double          cos_theta_p;
  double          theta_p;

  if ((norm_rel_v3 >= num_thresh) && (norm_ox3 >= num_thresh))
  {
      cos_theta_p                 = rel_v3_dot_ox3/(rel_ee_speed*d_p);
      theta_p                     = fabs(acos(cos_theta_p));
  }
  else
  {
      cos_theta_p                 = 0.0;
      theta_p                     = 0.0;
  }

  Vector3         rotation_axis_p     = ox3.cross(rel_v3);
  double          norm_rotation_axis_p= rotation_axis_p.norm();
  Matrix3x3       R_p;                // the rotation matrix for evaluation
  point p

  if (norm_rotation_axis_p < MIN_ROTATION_AXIS_NORM)
  {
      rotation_axis_p                 = ZeroVector3;
      R_p                             = ZeroMatrix3x3;
  }
  else
  {
      rotation_axis_p                 = rotation_axis_p/norm_rotation_axis_p; //
  normalize vector (required by Eigen's AngleAxisd)

      // compute the rotation matrix:
      R_p                             = Eigen::AngleAxisd(0.5*M_PI,
  rotation_axis_p);
  }

  if (feature_matrix_per_point.cols() != loa_feat_param->PF_DYN2_N_beta_vector)
  {
      feature_matrix_per_point.resize(3, loa_feat_param->PF_DYN2_N_beta_vector);
  }

  feature_matrix_per_point    = (R_p * (tau * rel_v3)).replicate(1,
  loa_feat_param->PF_DYN2_beta_vector->rows()); feature_matrix_per_point    =
  feature_matrix_per_point.array() * (-0.5 * ((((theta_p  *
  OnesVectorloaG2(loa_feat_param->PF_DYN2_beta_vector->rows())) -
  *(loa_feat_param->PF_DYN2_beta_vector)).array().square()).array()   *
  loa_feat_param->PF_DYN2_beta_D_vector->array()).transpose()).array().exp().replicate(3,1).array();
  feature_matrix_per_point    = feature_matrix_per_point.array() * (-0.5 *
  ((((d_p          * OnesVectorloaG2(loa_feat_param->PF_DYN2_k_vector->rows()))
  - *(loa_feat_param->PF_DYN2_k_vector)).array().square()).array()      *
  loa_feat_param->PF_DYN2_k_D_vector->array()).transpose()).array().exp().replicate(3,1).array();
  */

  if (rt_assert(containsNaN(feature_matrix_per_point) == false) == false) {
    return false;
  }

  return true;
}

/**
 * This is for computing the kernelized general features (KGF), which is a
 * product of \n 3 kernels: 1 kernel as a function of theta_p, \n 1 kernel as a
 * function of d_p, and \n 1 kernel as a function of the phase variable s (from
 * canonical system discrete)\n (designed by Giovanni Sutanto), \n i.e. the
 * sub-feature vector, which is needed for constructing the stacked feature
 * vector Phi \n in computeFeatures() function.
 *
 * @param endeff_cart_state_local The end-effector Cartesian state in local
 * coordinate system
 * @param obs_evalpt_cart_state_local The Cartesian state (in local coordinate
 * system) of an evaluation point inside/on the obstacle geometry
 * @param feature_matrix_per_point The computed feature matrix for the specified
 * obstacle point (to be returned)
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::
    computeKGFObsAvoidCtFeatureMatricesPerObsPoint(
        const DMPState& endeff_cart_state_local,
        const DMPState& obs_evalpt_cart_state_local,
        Matrixloa3xG3& feature_matrix_per_point) {
  double tau;
  if (rt_assert(tau_sys->getTauRelative(tau)) == false) {
    return false;
  }

  Vector3 endeff_cart_position_local =
      endeff_cart_state_local.getX().block(0, 0, 3, 1);
  Vector3 endeff_cart_velocity_local =
      endeff_cart_state_local.getXd().block(0, 0, 3, 1);
  Vector3 obs_evalpt_cart_position_local =
      obs_evalpt_cart_state_local.getX().block(0, 0, 3, 1);
  Vector3 obs_evalpt_cart_velocity_local =
      obs_evalpt_cart_state_local.getXd().block(0, 0, 3, 1);

  Vector3 rel_v3 = endeff_cart_velocity_local - obs_evalpt_cart_velocity_local;
  Vector3 ox3 = obs_evalpt_cart_position_local - endeff_cart_position_local;

  double norm_rel_v3 = rel_v3.norm();
  double norm_ox3 = ox3.norm();
  double rel_v3_dot_ox3 = rel_v3.dot(ox3);
  double ox3_dot_ox3 = ox3.dot(ox3);

  // numerical threshold
  double numerical_threshold = 1e-4;

  // dp is distance between the obstacle evaluation point (evalpt) p and endeff
  // in the local coordinate system:
  double d_p = norm_ox3;

  // rel_ee_speed is the (scalar) magnitude of relative endeff velocity with
  // respect to the obstacle evaluation point (evalpt) p in the local coordinate
  // system:
  double rel_ee_speed = norm_rel_v3;

  double cos_theta_p;
  double theta_p;

  // get current phase variable value from canonical system:
  double s = loa_feat_param->canonical_sys_discrete->getX();
  double s_multiplier =
      loa_feat_param->canonical_sys_discrete->getCanonicalMultiplier();

  if ((norm_rel_v3 >= numerical_threshold) &&
      (norm_ox3 >= numerical_threshold)) {
    cos_theta_p = rel_v3_dot_ox3 / (rel_ee_speed * d_p);
    theta_p = fabs(acos(cos_theta_p));
  } else {
    cos_theta_p = 0.0;
    theta_p = 0.0;
  }

  Vector3 rotation_axis_p = ox3.cross(rel_v3);
  double norm_rotation_axis_p = rotation_axis_p.norm();
  Matrix3x3 R_p;  // the rotation matrix for evaluation point p

  if (norm_rotation_axis_p < numerical_threshold) {
    rotation_axis_p = ZeroVector3;
    R_p = ZeroMatrix3x3;
  } else {
    rotation_axis_p =
        rotation_axis_p / norm_rotation_axis_p;  // normalize vector (required
                                                 // by Eigen's AngleAxisd)

    // compute the rotation matrix:
    R_p = Eigen::AngleAxisd(0.5 * M_PI, rotation_axis_p);
  }

  if (feature_matrix_per_point.cols() != loa_feat_param->KGF_N_beta_vector) {
    feature_matrix_per_point.resize(3, loa_feat_param->KGF_N_beta_vector);
  }

  Vector3 tau_R_p_rel_v3 = (R_p * (tau * rel_v3));
  double theta_kernel_normalizer = 0.0;
  double s_kernel_normalizer = 0.0;
  for (uint i = 0; i < loa_feat_param->KGF_beta_rowcoldepth_vector->rows();
       i++) {
    double theta_kernel_value =
        (theta_p - (*(loa_feat_param->KGF_beta_rowcoldepth_vector))(i, 0));
    theta_kernel_value =
        exp(-0.5 * theta_kernel_value * theta_kernel_value *
            (*(loa_feat_param->KGF_beta_rowcoldepth_D_vector))(i, 0));
    // double  d_kernel_value      = (d_p     -
    // (*(loa_feat_param->KGF_k_rowcoldepth_vector))(i,0)); d_kernel_value =
    // exp(-0.5 * d_kernel_value     * d_kernel_value     *
    // (*(loa_feat_param->KGF_k_rowcoldepth_D_vector))(i,0));
    double d_kernel_value =
        exp(-(*(loa_feat_param->KGF_k_rowcoldepth_vector))(i, 0) * (d_p * d_p));
    double s_kernel_value =
        (s - (*(loa_feat_param->KGF_s_rowcoldepth_vector))(i, 0));
    s_kernel_value = exp(-0.5 * s_kernel_value * s_kernel_value *
                         (*(loa_feat_param->KGF_s_rowcoldepth_D_vector))(i, 0));
    feature_matrix_per_point.block(0, i, 3, 1) =
        tau_R_p_rel_v3 * theta_kernel_value * d_kernel_value * s_kernel_value;
  }

  for (uint j = 0; j < loa_feat_param->KGF_beta_col_grid->rows(); j++) {
    double theta_kernel_value =
        (theta_p - (*(loa_feat_param->KGF_beta_col_grid))(j, 0));
    theta_kernel_value = exp(-0.5 * theta_kernel_value * theta_kernel_value *
                             (*(loa_feat_param->KGF_beta_col_D_grid))(j, 0));
    theta_kernel_normalizer += theta_kernel_value;
  }

  for (uint k = 0; k < loa_feat_param->KGF_s_depth_grid->rows(); k++) {
    double s_kernel_value = (s - (*(loa_feat_param->KGF_s_depth_grid))(k, 0));
    s_kernel_value = exp(-0.5 * s_kernel_value * s_kernel_value *
                         (*(loa_feat_param->KGF_s_depth_D_grid))(k, 0));
    s_kernel_normalizer += s_kernel_value;
  }

  feature_matrix_per_point = feature_matrix_per_point * s_multiplier /
                             (theta_kernel_normalizer * s_kernel_normalizer);

  /* An identical, but NON-REAL-TIME (MATLAB-like) implementation is as follows:
  {
      feature_matrix_per_point    = (R_p * (tau * rel_v3)).replicate(1,
  loa_feat_param->KGF_beta_rowcoldepth_vector->rows()); feature_matrix_per_point
  = feature_matrix_per_point.array() * (-0.5 * ((((theta_p *
  OnesVectorloaG3(loa_feat_param->KGF_beta_rowcoldepth_vector->rows())) -
  *(loa_feat_param->KGF_beta_rowcoldepth_vector)).array().square()).array()   *
  loa_feat_param->KGF_beta_rowcoldepth_D_vector->array()).transpose()).array().exp().replicate(3,1).array();
      //feature_matrix_per_point    = feature_matrix_per_point.array() * (-0.5 *
  ((((d_p     *
  OnesVectorloaG3(loa_feat_param->KGF_k_rowcoldepth_vector->rows()))    -
  *(loa_feat_param->KGF_k_rowcoldepth_vector)).array().square()).array()      *
  loa_feat_param->KGF_k_rowcoldepth_D_vector->array()).transpose()).array().exp().replicate(3,1).array();
      feature_matrix_per_point    = feature_matrix_per_point.array() *
  (-loa_feat_param->KGF_k_rowcoldepth_vector->transpose() * (d_p *
  d_p)).array().exp().replicate(3,1); feature_matrix_per_point    =
  feature_matrix_per_point.array() * (-0.5 * ((((s       *
  OnesVectorloaG3(loa_feat_param->KGF_s_rowcoldepth_vector->rows()))    -
  *(loa_feat_param->KGF_s_rowcoldepth_vector)).array().square()).array()      *
  loa_feat_param->KGF_s_rowcoldepth_D_vector->array()).transpose()).array().exp().replicate(3,1).array();

      double  theta_kernel_normalizer = (-0.5 * ((((theta_p *
  OnesVectorloaG3(loa_feat_param->KGF_beta_col_grid->rows())) -
  *(loa_feat_param->KGF_beta_col_grid)).array().square()).array()   *
  loa_feat_param->KGF_beta_col_D_grid->array())).array().exp().sum(); double
  s_kernel_normalizer     = (-0.5 * ((((s       *
  OnesVectorloaG3(loa_feat_param->KGF_s_depth_grid->rows()))    -
  *(loa_feat_param->KGF_s_depth_grid)).array().square()).array()  *
  loa_feat_param->KGF_s_depth_D_grid->array())).array().exp().sum();

      feature_matrix_per_point    = feature_matrix_per_point.array() *
  s_multiplier/(theta_kernel_normalizer * s_kernel_normalizer);
  }*/

  if (rt_assert(containsNaN(feature_matrix_per_point) == false) == false) {
    return false;
  }

  return true;
}

bool TransformCouplingLearnObsAvoid::computeNNObsAvoidCtFeature(
    const DMPState& endeff_cart_state_local,
    const ObstacleStates& point_obstacles_cart_state_local,
    VectorloaF3& feature_vector) {
  // input checking:
  if (rt_assert(feature_vector.rows() ==
                loa_feat_param->NN_feature_vector_rows) == false) {
    return false;
  }

  double tau;
  if (rt_assert(tau_sys->getTauRelative(tau)) == false) {
    return false;
  }

  Vector3 x3 = endeff_cart_state_local.getX().block(0, 0, 3, 1);
  Vector3 v3 = endeff_cart_state_local.getXd().block(0, 0, 3, 1);

  Vector3 tau_v3 = tau * v3;

  Vector3 o3_center = ZeroVector3;

  uint N_obs = point_obstacles_cart_state_local.size();

  uint N_closest_obs = 3;
  uint closest_obs_idx[3];

  point_obstacles_cartesian_distances_from_endeff->resize(N_obs);
  Vector3 o3;
  Vector3 xo3;
  double max_xo3_norm;
  uint max_xo3_norm_idx;
  double min_xo3_norm;
  uint min_xo3_norm_idx;

  for (uint i = 0; i < N_obs; i++) {
    o3 = point_obstacles_cart_state_local[i].getX().block(0, 0, 3, 1);
    xo3 = o3 - x3;
    o3_center += o3;
    (*point_obstacles_cartesian_distances_from_endeff)[i] = xo3.norm();
    if (i == 0) {
      max_xo3_norm = (*point_obstacles_cartesian_distances_from_endeff)[0];
      max_xo3_norm_idx = 0;
      min_xo3_norm = (*point_obstacles_cartesian_distances_from_endeff)[0];
      min_xo3_norm_idx = 0;
    } else {
      if ((*point_obstacles_cartesian_distances_from_endeff)[i] >
          max_xo3_norm) {
        max_xo3_norm = (*point_obstacles_cartesian_distances_from_endeff)[i];
        max_xo3_norm_idx = i;
      }
      if ((*point_obstacles_cartesian_distances_from_endeff)[i] <
          min_xo3_norm) {
        min_xo3_norm = (*point_obstacles_cartesian_distances_from_endeff)[i];
        min_xo3_norm_idx = i;
      }
    }
  }
  o3_center /= N_obs;  // = mean(point_obstacles_cart_position_local) = mean(o3)

  // this is not the most efficient sorting algorithm, but it will work in
  // real-time:
  closest_obs_idx[0] = min_xo3_norm_idx;
  (*point_obstacles_cartesian_distances_from_endeff)[closest_obs_idx[0]] =
      max_xo3_norm + 1.0;
  max_xo3_norm =
      (*point_obstacles_cartesian_distances_from_endeff)[closest_obs_idx[0]];
  max_xo3_norm_idx = closest_obs_idx[0];
  min_xo3_norm = max_xo3_norm;
  min_xo3_norm_idx = max_xo3_norm_idx;

  for (uint j = 1; j < N_closest_obs; j++) {
    for (uint i = 0; i < N_obs; i++) {
      if ((*point_obstacles_cartesian_distances_from_endeff)[i] <
          min_xo3_norm) {
        min_xo3_norm = (*point_obstacles_cartesian_distances_from_endeff)[i];
        min_xo3_norm_idx = i;
      }
    }

    closest_obs_idx[j] = min_xo3_norm_idx;
    (*point_obstacles_cartesian_distances_from_endeff)[closest_obs_idx[j]] =
        max_xo3_norm + 1.0;
    max_xo3_norm =
        (*point_obstacles_cartesian_distances_from_endeff)[closest_obs_idx[j]];
    max_xo3_norm_idx = closest_obs_idx[j];
    min_xo3_norm = max_xo3_norm;
    min_xo3_norm_idx = max_xo3_norm_idx;
  }

  Vector3 o3_centerx = o3_center - x3;

  (*loa_feat_param->NN_loa_feature_vector_buffer).block(0, 0, 3, 1) =
      o3_centerx;
  for (uint i = 0; i < 3; i++) {
    for (uint j = 0; j < N_closest_obs; j++) {
      (*loa_feat_param->NN_loa_feature_vector_buffer)(
          3 + ((i * N_closest_obs) + j), 0) =
          (point_obstacles_cart_state_local[closest_obs_idx[j]].getX())(i, 0) -
          x3(i, 0);
    }
  }
  (*loa_feat_param->NN_loa_feature_vector_buffer).block(12, 0, 3, 1) = tau_v3;
  (*loa_feat_param->NN_loa_feature_vector_buffer)(15, 0) = o3_centerx.norm();

  Vector3 ox3 =
      point_obstacles_cart_state_local[closest_obs_idx[0]].getX().block(0, 0, 3,
                                                                        1) -
      x3;

  double num_thresh = 0.0;

  double cos_theta;
  double clamped_cos_theta;

  if ((v3.norm() > num_thresh) && (ox3.norm() > num_thresh)) {
    cos_theta = (ox3.dot(v3)) / (ox3.norm() * v3.norm());
    clamped_cos_theta = cos_theta;
    if (clamped_cos_theta < 0.0) {
      clamped_cos_theta = 0.0;
    }
  } else {
    cos_theta = -1.0;
    clamped_cos_theta = 0.0;
  }

  (*loa_feat_param->NN_loa_feature_vector_buffer)(16, 0) = cos_theta;

  // get current phase variable value from canonical system:
  /*double          s               =
  loa_feat_param->canonical_sys_discrete->getX(); double          s_multiplier
  = loa_feat_param->canonical_sys_discrete->getCanonicalMultiplier();

  (*loa_feat_param->NN_loa_feature_vector_buffer)(20,0)   = s;
  *loa_feat_param->NN_loa_feature_vector_buffer           *= s_multiplier;

  double          ox3_squared     = ox3.transpose() * ox3;
  for (uint i=0; i<loa_feat_param->NN_N_k_grid; i++)
  {
      (*loa_feat_param->NN_dist_kernels)(i,0) = exp(-0.5 *
  (*loa_feat_param->NN_k_grid)(i,0) * ox3_squared);
  }

  for (uint m=0; m<loa_feat_param->NN_loa_feature_vector_buffer->rows(); m++)
  {
      for (uint i=0; i<loa_feat_param->NN_N_k_grid; i++)
      {
          feature_vector(((m*loa_feat_param->NN_N_k_grid)+i),0) =
  (*loa_feat_param->NN_dist_kernels)(i,0) *
  (*loa_feat_param->NN_loa_feature_vector_buffer)(m,0);
      }
  }*/

  feature_vector.block(0, 0, loa_feat_param->NN_N_input, 1) =
      loa_feat_param->NN_loa_feature_vector_buffer->block(
          0, 0, loa_feat_param->NN_N_input, 1);

  return true;
}

bool TransformCouplingLearnObsAvoid::computeNNInference(
    const VectorloaF3& feature_vector, VectorloaNN_O& NN_output) {
  // input checking:
  if (rt_assert(feature_vector.rows() == loa_feat_param->NN_N_input) == false) {
    return false;
  }

  *loa_feat_param->NN_loa_I_Xp1_vector =
      feature_vector.block(0, 0, loa_feat_param->NN_N_input, 1) -
      (*loa_feat_param->NN_loa_I_x1_step1_xoffset_vector);
  *loa_feat_param->NN_loa_I_Xp1_vector =
      loa_feat_param->NN_loa_I_Xp1_vector->cwiseProduct(
          *loa_feat_param->NN_loa_I_x1_step1_gain_vector);
  *loa_feat_param->NN_loa_I_Xp1_vector +=
      OnesVectorloaNN_I(loa_feat_param->NN_N_input) *
      (loa_feat_param->NN_loa_I_x1_step1_ymin);

  *loa_feat_param->NN_loa_H1_a1_vector =
      (*loa_feat_param->NN_loa_H1xI__IW1_1_weight_matrix) *
      (*loa_feat_param->NN_loa_I_Xp1_vector);
  *loa_feat_param->NN_loa_H1_a1_vector +=
      *loa_feat_param->NN_loa_H1_b1_bias_vector;
  for (uint i = 0; i < loa_feat_param->NN_loa_H1_a1_vector->rows(); i++) {
    if ((*loa_feat_param->NN_loa_H1_a1_vector)(i, 0) < 0.0) {
      (*loa_feat_param->NN_loa_H1_a1_vector)(i, 0) = 0.0;
    }
  }

  *loa_feat_param->NN_loa_H2_a2_vector =
      (*loa_feat_param->NN_loa_H2xH1_LW2_1_weight_matrix) *
      (*loa_feat_param->NN_loa_H1_a1_vector);
  *loa_feat_param->NN_loa_H2_a2_vector +=
      *loa_feat_param->NN_loa_H2_b2_bias_vector;
  for (uint i = 0; i < loa_feat_param->NN_loa_H2_a2_vector->rows(); i++) {
    if ((*loa_feat_param->NN_loa_H2_a2_vector)(i, 0) < 0.0) {
      (*loa_feat_param->NN_loa_H2_a2_vector)(i, 0) = 0.0;
    }
  }

  *loa_feat_param->NN_loa_O_a3_vector =
      (*loa_feat_param->NN_loa_OxH2__LW3_2_weight_matrix) *
      (*loa_feat_param->NN_loa_H2_a2_vector);
  *loa_feat_param->NN_loa_O_a3_vector +=
      *loa_feat_param->NN_loa_O_b3_bias_vector;
  for (uint i = 0; i < loa_feat_param->NN_loa_O_a3_vector->rows(); i++) {
    (*loa_feat_param->NN_loa_O_a3_vector)(i, 0) =
        (2.0 /
         (1.0 + exp(-2.0 * (*loa_feat_param->NN_loa_O_a3_vector)(i, 0)))) -
        1.0;
  }

  *loa_feat_param->NN_loa_O_y_vector =
      (*loa_feat_param->NN_loa_O_a3_vector) -
      (OnesVectorloaNN_O(loa_feat_param->NN_N_output) *
       loa_feat_param->NN_loa_O_y1_step1_ymin);
  *loa_feat_param->NN_loa_O_y_vector =
      loa_feat_param->NN_loa_O_y_vector->cwiseQuotient(
          *loa_feat_param->NN_loa_O_y1_step1_gain_vector);
  *loa_feat_param->NN_loa_O_y_vector +=
      (*loa_feat_param->NN_loa_O_y1_step1_xoffset_vector);

  NN_output = *loa_feat_param->NN_loa_O_y_vector;

  return true;
}

/**
 * Returns both acceleration-level coupling value (C_t_{Acc}) and
 * velocity-level coupling value (C_t_{Vel}) that will be used in the
 * transformation system to adjust motion of a DMP.
 *
 * @param ct_acc Acceleration-level coupling value (C_t_{Acc}) that will be used
 * to adjust the acceleration of DMP transformation system
 * @param ct_vel Velocity-level coupling value (C_t_{Vel}) that will be used to
 * adjust the velocity of DMP transformation system
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::getValue(VectorN& ct_acc,
                                              VectorN& ct_vel) {
  // pre-conditions checking
  if (rt_assert(TransformCouplingLearnObsAvoid::isValid()) == false) {
    return false;
  }

  Vector3 ct_acc_3D = ZeroVector3;

  if (loa_feat_param->is_using_NN == false) {
    if (rt_assert(TransformCouplingLearnObsAvoid::computeFeatures(
            *endeff_cartesian_state_global,
            *point_obstacles_cartesian_state_global,
            *ctraj_hmg_transform_global_to_local_matrix,
            *(loa_feat_param->cart_coord_loa_feature_vector),
            loa_feat_param->cart_coord_loa_feature_matrix.get())) == false) {
      return false;
    }

    if (is_constraining_Rp_yd_relationship == false) {
      ct_acc_3D = *(loa_feat_param->weights) *
                  (*(loa_feat_param->cart_coord_loa_feature_vector));
    } else {  // if (is_constraining_Rp_yd_relationship == true)
      ct_acc_3D = loa_feat_param->cart_coord_loa_feature_matrix->transpose() *
                  (*(loa_feat_param->weights_Rp_yd_constrained));
    }
  } else {  // if (loa_feat_param->is_using_NN == true)
    if (rt_assert(TransformCouplingLearnObsAvoid::computeFeatures(
            *endeff_cartesian_state_global,
            *point_obstacles_cartesian_state_global,
            *ctraj_hmg_transform_global_to_local_matrix,
            *(loa_feat_param->cart_coord_loa_feature_vector))) == false) {
      return false;
    }

    if (loa_feat_param->pmnn != NULL) {
      loa_feat_param->pmnn_input_vector->block(
          0, 0, loa_feat_param->cart_coord_loa_feature_vector->rows(), 1) =
          *(loa_feat_param->cart_coord_loa_feature_vector);
      VectorM normalized_basis_func_vector_mult_phase_multiplier =
          ZeroVectorM(func_approx_discrete->getModelSize());
      if (rt_assert(
              func_approx_discrete
                  ->getNormalizedBasisFunctionVectorMultipliedPhaseMultiplier(
                      normalized_basis_func_vector_mult_phase_multiplier)) ==
          false) {
        return false;
      }
      *(loa_feat_param->pmnn_phase_kernel_modulation) =
          normalized_basis_func_vector_mult_phase_multiplier;
      if (rt_assert(loa_feat_param->pmnn->computePrediction(
              *(loa_feat_param->pmnn_input_vector),
              *(loa_feat_param->pmnn_phase_kernel_modulation),
              *(loa_feat_param->pmnn_output_vector), 0, 2)) == false) {
        return false;
      }
      ct_acc_3D.block(0, 0, 3, 1) = *(loa_feat_param->pmnn_output_vector);
    } else {
      VectorloaNN_O NN_loa_output(3);
      if (rt_assert(TransformCouplingLearnObsAvoid::computeNNInference(
              *(loa_feat_param->cart_coord_loa_feature_vector),
              NN_loa_output)) == false) {
        return false;
      }

      ct_acc_3D.block(0, 0, 3, 1) = NN_loa_output;
      ct_acc_3D(0, 0) = 0.0;

      // post-processing:
      Vector4 goal_cart_position_global_H = ZeroVector4;
      Vector4 goal_cart_position_local_H = ZeroVector4;
      goal_cart_position_global_H.block(0, 0, 3, 1) =
          *goal_cartesian_position_global;
      goal_cart_position_global_H(3, 0) = 1.0;

      if (rt_assert(cart_coord_transformer.computeCTrajAtNewCoordSys(
              &goal_cart_position_global_H, NULL, NULL, NULL,
              *ctraj_hmg_transform_global_to_local_matrix,
              &goal_cart_position_local_H, NULL, NULL, NULL)) == false) {
        return false;
      }
      goal_cartesian_position_local.block(0, 0, 3, 1) =
          goal_cart_position_local_H.block(0, 0, 3, 1);

      Vector3 o3_center = ZeroVector3;
      Vector3 point_obstacle_cart_position_local = ZeroVector3;
      double min_x;
      double max_x;
      for (uint i = 0; i < point_obstacles_cartesian_state_local->size(); i++) {
        point_obstacle_cart_position_local =
            (*point_obstacles_cartesian_state_local)[i].getX();
        if (i == 0) {
          min_x = point_obstacle_cart_position_local(0, 0);
          max_x = point_obstacle_cart_position_local(0, 0);
        } else {
          if (point_obstacle_cart_position_local(0, 0) < min_x) {
            min_x = point_obstacle_cart_position_local(0, 0);
          }
          if (point_obstacle_cart_position_local(0, 0) > max_x) {
            max_x = point_obstacle_cart_position_local(0, 0);
          }
        }
        o3_center += point_obstacle_cart_position_local;
      }
      o3_center /= point_obstacles_cartesian_state_local->size();

      Vector3 x3 = endeff_cartesian_state_local->getX();

      if (x3(0, 0) > max_x) {
        ct_acc_3D =
            ct_acc_3D * exp(-10.0 * ((x3(0, 0) - max_x) * (x3(0, 0) - max_x)));
      }
      if (goal_cartesian_position_local(0, 0) < min_x) {
        ct_acc_3D = ZeroVector3;
      }

      // double          s_multiplier    =
      // loa_feat_param->canonical_sys_discrete->getCanonicalMultiplier();
      // ct_acc_3D       = ct_acc_3D * s_multiplier;
    }
  }

  if (rt_assert(containsNaN(ct_acc_3D) == false) == false) {
    return false;
  }

  ct_acc.block<3, 1>(0, 0) = ct_acc_3D;

  ct_vel.block<3, 1>(0, 0) = ZeroVector3;

  return true;
}

/**
 * This is a special case where we are provided with the information of the
 * sphere obstacle's center and radius, and we want to compute the points that
 * will be considered for obstacle avoidance features computation.
 *
 * @param endeff_cart_state_global The end-effector Cartesian state in the
 * global coordinate system
 * @param obssphctr_cart_state_global The obstacle's center Cartesian state in
 * the global coordinate system
 * @param obs_sphere_radius Radius of the sphere obstacle
 * @param point_obstacles_cart_state_global Vector of point obstacles' Cartesian
 * state in the global coordinate system (to be returned)
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::
    computePointObstaclesCartStateGlobalFromStaticSphereObstacle(
        const DMPState& endeff_cart_state_global,
        const DMPState& obssphctr_cart_state_global,
        const double& obs_sphere_radius,
        ObstacleStates& point_obstacles_cart_state_global) {
  // input checking
  // point_obstacles_cart_state_global needs to be pre-allocated to make sure
  // real-time performance is guaranteed
  if (rt_assert(point_obstacles_cart_state_global.capacity() >= 2) == false) {
    return false;
  }

  point_obstacles_cart_state_global.resize(2);

  Vector3 endeff_cart_position_global =
      endeff_cart_state_global.getX().block(0, 0, 3, 1);
  Vector3 obssphctr_cart_position_global =
      obssphctr_cart_state_global.getX().block(0, 0, 3, 1);
  Vector3 obssphctr_cart_velocity_global =
      obssphctr_cart_state_global.getXd().block(0, 0, 3, 1);

  // obs_p_global is a point on the obstacle (sphere)'s surface,
  // which is closest to endeff_cart_position_global:
  Vector3 obs_p_global = computeClosestPointOnSphereSurface(
      endeff_cart_position_global, obssphctr_cart_position_global,
      obs_sphere_radius);

  point_obstacles_cart_state_global[0] = obssphctr_cart_state_global;
  point_obstacles_cart_state_global[1] = DMPState(
      obs_p_global, obssphctr_cart_velocity_global, ZeroVector3, rt_assertor);

  return true;
}

/**
 * NON-REAL-TIME!!!\n
 * This is a special case as well, similar to
 * computePointObstaclesCartStateGlobalFromStaticSphereObstacle(), but this time
 * is with regard to an evaluation trajectory (endeff_cart_state_traj_global),
 * instead of an evaluation state. Please note that even though the sphere
 * obstacle is static, the closest point on the surface of the sphere obstacle
 * is dynamic, i.e. changing as the evaluation state (a state in
 * endeff_cart_state_traj_global) changes. This function is called from inside
 * TransformCouplingLearnObsAvoid::learnCouplingTerm() function.
 *
 * @param endeff_cart_state_traj_global The end-effector Cartesian state
 * trajectory in the global coordinate system
 * @param obssphctr_cart_state_global The obstacle's center Cartesian state in
 * the global coordinate system
 * @param obs_sphere_radius Radius of the sphere obstacle
 * @param point_obstacles_cart_state_traj_global Trajectory or vector of vectors
 * of point obstacles' Cartesian state \n in the global coordinate system (to be
 * returned)
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::
    computePointObstaclesCartStateGlobalTrajectoryFromStaticSphereObstacle(
        const Trajectory& endeff_cart_state_traj_global,
        const DMPState& obssphctr_cart_state_global,
        const double& obs_sphere_radius,
        ObstacleStatesVector& point_obstacles_cart_state_traj_global) {
  // input checking
  // point_obstacles_cart_state_traj_global needs to be pre-allocated to make
  // sure real-time performance is guaranteed
  /*if (rt_assert(point_obstacles_cart_state_traj_global.capacity() >=
  endeff_cart_state_traj_global.size()) == false)
  {
      return false;
  }*/

  // this command might NOT be real-time safe (depending on current
  // point_obstacles_cart_state_traj_global.capacity()):
  point_obstacles_cart_state_traj_global.resize(
      endeff_cart_state_traj_global.size());

  for (uint i = 0; i < endeff_cart_state_traj_global.size(); i++) {
    point_obstacles_cart_state_traj_global[i].resize(
        2);  // this command might NOT be real-time safe ...

    if (rt_assert(computePointObstaclesCartStateGlobalFromStaticSphereObstacle(
            endeff_cart_state_traj_global[i], obssphctr_cart_state_global,
            obs_sphere_radius, point_obstacles_cart_state_traj_global[i])) ==
        false) {
      return false;
    }
  }

  return true;
}

/**
 * This is simply a conversion function from a vector of position
 * (point_obstacles_cart_position_global) into a vector of state
 * (point_obstacles_cart_state_global)
 *
 * @param point_obstacles_cart_position_global The point obstacles' Cartesian
 * coordinate in the global coordinate system
 * @param point_obstacles_cart_state_global The point obstacles' Cartesian state
 * in the global coordinate system (to be returned)
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::
    computePointObstaclesCartStateGlobalFromStaticViconData(
        const Points& point_obstacles_cart_position_global,
        ObstacleStates& point_obstacles_cart_state_global) {
  // input checking
  // point_obstacles_cart_state_global needs to be pre-allocated to make sure
  // real-time performance is guaranteed (assuming
  // MIN_POINT_OBSTACLE_VECTOR_PREALLOCATION is always greater than any input
  // point_obstacles_cart_position_global.size())
  if (rt_assert(MIN_POINT_OBSTACLE_VECTOR_PREALLOCATION >=
                point_obstacles_cart_position_global.size()) == false) {
    return false;
  }
  if (rt_assert(point_obstacles_cart_state_global.capacity() >=
                MIN_POINT_OBSTACLE_VECTOR_PREALLOCATION) == false) {
    return false;
  }

  point_obstacles_cart_state_global.resize(
      point_obstacles_cart_position_global.size());

  for (uint i = 0; i < point_obstacles_cart_position_global.size(); i++) {
    point_obstacles_cart_state_global[i] =
        DMPState(point_obstacles_cart_position_global[i], ZeroVector3,
                 ZeroVector3, rt_assertor);
  }

  return true;
}

/**
 * NON-REAL-TIME!!!\n
 * This is a special case as well, similar to
 * computePointObstaclesCartStateGlobalFromStaticSphereObstacle(), but this time
 * is for point obstacles processed from Vicon data. Because the point obstacles
 * are static, the point_obstacles_cart_state_traj_global only has size 1 (1
 * state), which applies to all end-effector states in the end-effector
 * trajectory. This function is called from inside
 * TransformCouplingLearnObsAvoid::learnCouplingTerm() function.
 *
 * @param point_obstacles_cart_position_global The point obstacles' Cartesian
 * coordinate in the global coordinate system
 * @param point_obstacles_cart_state_traj_global Trajectory or vector of vectors
 * of point obstacles' Cartesian state \n in the global coordinate system (to be
 * returned)
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::
    computePointObstaclesCartStateGlobalTrajectoryFromStaticViconData(
        const Points& point_obstacles_cart_position_global,
        ObstacleStatesVector& point_obstacles_cart_state_traj_global) {
  // this command might NOT be real-time safe (depending on current
  // point_obstacles_cart_state_traj_global.capacity()):
  point_obstacles_cart_state_traj_global.resize(1);

  ObstacleStates point_obstacles_cart_state_global;
  point_obstacles_cart_state_global.reserve(
      MIN_POINT_OBSTACLE_VECTOR_PREALLOCATION);

  if (rt_assert(computePointObstaclesCartStateGlobalFromStaticViconData(
          point_obstacles_cart_position_global,
          point_obstacles_cart_state_global)) == false) {
    return false;
  }

  point_obstacles_cart_state_traj_global[0] = point_obstacles_cart_state_global;

  return true;
}

bool TransformCouplingLearnObsAvoid::startCollectingTrajectoryDataSet() {
  if (rt_assert(loa_data_io.reset()) == false) {
    return false;
  }

  is_collecting_trajectory_data = true;

  return true;
}

void TransformCouplingLearnObsAvoid::stopCollectingTrajectoryDataSet() {
  is_collecting_trajectory_data = false;
}

/**
 * NON-REAL-TIME!!!\n
 * If you want to maintain real-time performance,
 * do/call this function from a separate (non-real-time) thread!!!
 *
 * @param new_data_dir_path New data directory path \n
 *                          (if we want to save the data in a directory path
 * other than the data_directory_path)
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::saveTrajectoryDataSet(
    const char* new_data_dir_path) {
  // pre-conditions checking
  if (rt_assert(TransformCouplingLearnObsAvoid::isValid()) == false) {
    return false;
  }

  if (strcmp(new_data_dir_path, "") == 0) {
    return (rt_assert(loa_data_io.saveTrajectoryDataSet(data_directory_path)));
  } else {
    return (rt_assert(loa_data_io.saveTrajectoryDataSet(new_data_dir_path)));
  }
}

bool TransformCouplingLearnObsAvoid::saveWeights(const char* dir_path,
                                                 const char* file_name) {
  if (is_constraining_Rp_yd_relationship) {
    return (rt_assert(loa_data_io.writeMatrixToFile(
        dir_path, file_name, *(loa_feat_param->weights_Rp_yd_constrained))));
  } else {
    return (rt_assert(loa_data_io.writeMatrixToFile(
        dir_path, file_name, *(loa_feat_param->weights))));
  }
}

bool TransformCouplingLearnObsAvoid::saveWeights(const char* file_path) {
  if (is_constraining_Rp_yd_relationship) {
    return (rt_assert(loa_data_io.writeMatrixToFile(
        file_path, *(loa_feat_param->weights_Rp_yd_constrained))));
  } else {
    return (rt_assert(
        loa_data_io.writeMatrixToFile(file_path, *(loa_feat_param->weights))));
  }
}

bool TransformCouplingLearnObsAvoid::loadWeights(const char* dir_path,
                                                 const char* file_name) {
  if (is_constraining_Rp_yd_relationship) {
    return (rt_assert(loa_data_io.readMatrixFromFile(
        dir_path, file_name, *(loa_feat_param->weights_Rp_yd_constrained))));
  } else {
    return (rt_assert(loa_data_io.readMatrixFromFile(
        dir_path, file_name, *(loa_feat_param->weights))));
  }
}

bool TransformCouplingLearnObsAvoid::loadWeights(const char* file_path) {
  if (is_constraining_Rp_yd_relationship) {
    return (rt_assert(loa_data_io.readMatrixFromFile(
        file_path, *(loa_feat_param->weights_Rp_yd_constrained))));
  } else {
    return (rt_assert(
        loa_data_io.readMatrixFromFile(file_path, *(loa_feat_param->weights))));
  }
}

/**
 * NON-REAL-TIME!!!\n
 * If you want to maintain real-time performance,
 * do/call this function from a separate (non-real-time) thread!!!\n
 * Read sphere obstacle parameters from files in a specified directory.
 *
 * @param dir_path Directory path from where the data files will be read
 * @param obsctr_cart_position_global Center point coordinate of the obstacle
 * geometry, in Cartesian global coordinate system
 * @param radius Radius of the sphere obstacle
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::readSphereObstacleParametersFromFile(
    const char* dir_path, Vector3& obsctr_cart_position_global,
    double& radius) {
  return (rt_assert(loa_data_io.readSphereObstacleParametersFromFile(
      dir_path, obsctr_cart_position_global, radius)));
}

/**
 * NON-REAL-TIME!!!\n
 * If you want to maintain real-time performance,
 * do/call this function from a separate (non-real-time) thread!!!\n
 * Read point obstacles coordinates from file(s) in a specified directory.
 *
 * @param dir_path Directory path from where the data files will be read
 * @param point_obstacles_cart_position_global Vector of point obstacles'
 * coordinates in Cartesian global coordinate system
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::readPointObstaclesFromFile(
    const char* dir_path, Points& point_obstacles_cart_position_global) {
  return (rt_assert(loa_data_io.readPointObstaclesFromFile(
      dir_path, point_obstacles_cart_position_global)));
}

/**
 * NON-REAL-TIME!!!\n
 * If you want to maintain real-time performance,
 * do/call this function from a separate (non-real-time) thread!!!\n
 * Extract all DMP obstacle avoidance coupling term training data information \n
 * for point obstacles represented by Vicon markers: \n
 * demonstrations and point obstacles' coordinate in the global coordinate
 * system.
 *
 * @param in_data_dir_path The input data directory
 * @param demo_group_set_obs_avoid_global [to-be-filled/returned] Extracted
 * obstacle avoidance demonstrations \n in the Cartesian global coordinate
 * system
 * @param vector_point_obstacles_cart_position_global [to-be-filled/returned]
 * Vector of point obstacles' coordinates \n in Cartesian global coordinate
 * system, \n corresponding to a set of demonstrations
 * @param N_demo_settings [to-be-filled/returned] Total number of different
 * obstacle avoidance demonstration settings \n (or total number of different
 * obstacle positions)
 * @param max_num_trajs_per_setting [optional] Maximum number of trajectories
 * that could be loaded into a trajectory set \n of an obstacle avoidance
 * demonstration setting \n (setting = obstacle position) (default is 500)
 * @param selected_obs_avoid_setting_numbers [optional] If we are just
 * interested in learning a subset of the obstacle avoidance demonstration
 * settings,\n then specify the numbers (IDs) of the setting here (as a
 * vector).\n If not specified, then all settings will be considered.
 * @param demo_group_set_dmp_unroll_init_params [optional] DMP unroll
 * initialization parameters of each trajectory in \n the extracted
 * demonstrations (to be returned)
 * @return Success or failure
 */
bool TransformCouplingLearnObsAvoid::extractObstacleAvoidanceTrainingDataVicon(
    const char* in_data_dir_path,
    DemonstrationGroupSet& demo_group_set_obs_avoid_global,
    PointsVector& vector_point_obstacles_cart_position_global,
    uint& N_demo_settings, uint max_num_trajs_per_setting,
    std::vector<uint>* selected_obs_avoid_setting_numbers,
    VecVecDMPUnrollInitParams* demo_group_set_dmp_unroll_init_params) {
  return (rt_assert(loa_data_io.extractObstacleAvoidanceTrainingDataVicon(
      in_data_dir_path, demo_group_set_obs_avoid_global,
      vector_point_obstacles_cart_position_global, N_demo_settings,
      max_num_trajs_per_setting, selected_obs_avoid_setting_numbers,
      demo_group_set_dmp_unroll_init_params)));
}

TransformCouplingLearnObsAvoid::~TransformCouplingLearnObsAvoid() {}

}  // namespace dmp
