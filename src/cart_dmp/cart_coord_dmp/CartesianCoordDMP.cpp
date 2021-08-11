#include "dmp/cart_dmp/cart_coord_dmp/CartesianCoordDMP.h"

namespace dmp {

CartesianCoordDMP::CartesianCoordDMP()
    : transform_sys_discrete_cart_coord(TransformSystemDiscrete(
          3, NULL, nullptr, &logged_dmp_discrete_variables, NULL,
          std::vector<bool>(3, true))),
      DMPDiscrete(3, 0, NULL, &transform_sys_discrete_cart_coord,
                  _SCHAAL_LWR_METHOD_, NULL, ""),
      cart_coord_transformer(CartesianCoordTransformer()),
      ctraj_local_coord_selection(2) {
  ctraj_hmg_transform_local_to_global_matrix = ZeroMatrix4x4;
  ctraj_hmg_transform_global_to_local_matrix = ZeroMatrix4x4;

  ctraj_critical_states_global_coord.resize(5);
  ctraj_critical_states_local_coord.resize(5);
}

CartesianCoordDMP::CartesianCoordDMP(
    CanonicalSystemDiscrete* canonical_system_discrete, uint model_size_init,
    uint ts_formulation_type, uint learning_method,
    RealTimeAssertor* real_time_assertor,
    uint ctraj_local_coordinate_frame_selection,
    const char* opt_data_directory_path,
    std::vector<TransformCoupling*>* transform_couplers)
    : transform_sys_discrete_cart_coord(TransformSystemDiscrete(
          3, canonical_system_discrete,
          std::make_unique<FuncApproximatorDiscrete>(3, model_size_init,
                                                     canonical_system_discrete,
                                                     real_time_assertor),
          &logged_dmp_discrete_variables, real_time_assertor,
          std::vector<bool>(3, true), NULL, NULL, NULL, NULL,
          transform_couplers, 25.0, 25.0 / 4.0, ts_formulation_type)),
      DMPDiscrete(3, model_size_init, canonical_system_discrete,
                  &transform_sys_discrete_cart_coord, learning_method,
                  real_time_assertor, opt_data_directory_path),
      cart_coord_transformer(CartesianCoordTransformer(real_time_assertor)),
      ctraj_local_coord_selection(ctraj_local_coordinate_frame_selection) {
  // if using Schaal's DMP model:
  if (ts_formulation_type == _SCHAAL_DMP_) {
    // only (local) x-axis DMP is using scaling, (local) y and (local) z-axes do
    // NOT use scaling
    std::vector<bool> is_using_scaling_init = std::vector<bool>(3, true);
    if ((ctraj_local_coord_selection == _GSUTANTO_LOCAL_COORD_FRAME_) ||
        (ctraj_local_coord_selection == _SCHAAL_LOCAL_COORD_FRAME_)) {
      is_using_scaling_init[1] = false;  // turn-off scaling on y-axis
      is_using_scaling_init[2] = false;  // turn-off scaling on z-axis
    } else if (ctraj_local_coord_selection == _KROEMER_LOCAL_COORD_FRAME_) {
      is_using_scaling_init[2] = false;  // turn-off scaling on z-axis
    }
    DMPDiscrete::setScalingUsage(is_using_scaling_init);
    // (let me know if there is a better way of initializing std::vector<bool>
    // than this)
  }

  ctraj_hmg_transform_local_to_global_matrix = ZeroMatrix4x4;
  ctraj_hmg_transform_global_to_local_matrix = ZeroMatrix4x4;

  ctraj_critical_states_global_coord.resize(5);
  ctraj_critical_states_local_coord.resize(5);
}

bool CartesianCoordDMP::isValid() {
  if (rt_assert(transform_sys_discrete_cart_coord.isValid()) == false) {
    return false;
  }
  if (rt_assert(DMPDiscrete::isValid()) == false) {
    return false;
  }
  if (rt_assert(transform_sys_discrete == &transform_sys_discrete_cart_coord) ==
      false) {
    return false;
  }
  if (rt_assert(dmp_num_dimensions == 3) == false) {
    return false;
  }
  if (rt_assert(cart_coord_transformer.isValid()) == false) {
    return false;
  }
  if (rt_assert((rt_assert(ctraj_critical_states_global_coord.size() == 5)) &&
                (rt_assert(ctraj_critical_states_local_coord.size() == 5))) ==
      false) {
    return false;
  }
  if (rt_assert((rt_assert(ctraj_local_coord_selection >=
                           MIN_CTRAJ_LOCAL_COORD_OPTION_NO)) &&
                (rt_assert(ctraj_local_coord_selection <=
                           MAX_CTRAJ_LOCAL_COORD_OPTION_NO))) == false) {
    return false;
  }
  return true;
}

bool CartesianCoordDMP::preprocess(TrajectorySet& trajectory_set) {
  if (rt_assert(DMP::preprocess(trajectory_set)) == false) {
    return false;
  }

  mean_start_global_position = mean_start_position.block(0, 0, 3, 1);
  mean_goal_global_position = mean_goal_position.block(0, 0, 3, 1);

  int N_traj =
      trajectory_set.size();  // size of Cartesian coordinate trajectory
  for (uint i = 0; i < N_traj; ++i) {
    // computing the transformation to local coordinate system:
    if (rt_assert(cart_coord_transformer.computeCTrajCoordTransforms(
            trajectory_set[i], ctraj_hmg_transform_local_to_global_matrix,
            ctraj_hmg_transform_global_to_local_matrix,
            ctraj_local_coord_selection)) == false) {
      return false;
    }

    // computing trajectory representation in the local coordinate system:
    if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(
            trajectory_set[i], ctraj_hmg_transform_global_to_local_matrix,
            trajectory_set[i])) == false) {
      return false;
    }
  }

  if (rt_assert(cart_coord_transformer.computeCPosAtNewCoordSys(
          mean_start_global_position,
          ctraj_hmg_transform_global_to_local_matrix,
          mean_start_local_position)) == false) {
    return false;
  }

  if (rt_assert(cart_coord_transformer.computeCPosAtNewCoordSys(
          mean_goal_global_position, ctraj_hmg_transform_global_to_local_matrix,
          mean_goal_local_position)) == false) {
    return false;
  }

  return true;
}

bool CartesianCoordDMP::learn(const char* training_data_dir_or_file_path,
                              double robot_task_servo_rate, double* tau_learn,
                              Trajectory* critical_states_learn) {
  return (rt_assert(DMP::learn(training_data_dir_or_file_path,
                               robot_task_servo_rate, tau_learn,
                               critical_states_learn)));
}

bool CartesianCoordDMP::learn(const TrajectorySet& trajectory_set,
                              double robot_task_servo_rate) {
  bool is_real_time = false;

  // pre-conditions checking
  if (rt_assert(CartesianCoordDMP::isValid()) == false) {
    return false;
  }
  // input checking:
  if (rt_assert((rt_assert(robot_task_servo_rate > 0.0)) &&
                (rt_assert(robot_task_servo_rate <= MAX_TASK_SERVO_RATE))) ==
      false) {
    return false;
  }

  TrajectorySetPtr trajectory_set_preprocessed;
  if (rt_assert(allocateMemoryIfNonRealTime(
          is_real_time, trajectory_set_preprocessed)) == false) {
    return (-1);
  }
  (*trajectory_set_preprocessed) = trajectory_set;

  // pre-process trajectories
  if (rt_assert(CartesianCoordDMP::preprocess(*trajectory_set_preprocessed)) ==
      false) {
    return false;
  }

  if (rt_assert(learning_sys_discrete.learnApproximator(
          *trajectory_set_preprocessed, robot_task_servo_rate, mean_tau)) ==
      false) {
    return false;
  }

  if (rt_assert(mean_tau >= MIN_TAU) == false) {
    return false;
  }

  return true;
}

bool CartesianCoordDMP::learnMultiCartCoordDMPs(
    const char* in_data_dir_path, const char* out_data_dir_path,
    double robot_task_servo_rate, const char* in_data_subdir_subpath,
    bool save_unroll_traj_dataset, double tau_unroll) {
  bool is_real_time = false;

  // pre-conditions checking
  if (rt_assert(CartesianCoordDMP::isValid()) == false) {
    return false;
  }
  // input checking
  if (rt_assert(file_type(in_data_dir_path) == _DIR_) ==
      false)  // the in_data_dir_path directory must have existed!!!
  {
    return false;
  }

  if (rt_assert(createDirIfNotExistYet(out_data_dir_path)) == false) {
    return false;
  }

  std::filesystem::path in_data_dir(in_data_dir_path);
  std::filesystem::directory_iterator null_directory_iterator;
  char in_data_subdir_path[1000];
  char out_data_subdir_path[1000];

  for (std::filesystem::directory_iterator dir_iter(in_data_dir);
       dir_iter != null_directory_iterator; ++dir_iter) {
    char subdir_name[1000];
    snprintf(subdir_name, sizeof(subdir_name), "%s",
             dir_iter->path().filename().string().c_str());
    snprintf(in_data_subdir_path, sizeof(in_data_subdir_path), "%s/%s/",
             dir_iter->path().string().c_str(), in_data_subdir_subpath);
    snprintf(out_data_subdir_path, sizeof(out_data_subdir_path), "%s/%s/",
             out_data_dir_path, subdir_name);

    // Create out_data_subdir_path directory if it doesn't exist yet:
    if (rt_assert(createDirIfNotExistYet(out_data_subdir_path)) == false) {
      return false;
    }

    TrajectorySetPtr set_ctraj_input;
    if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, set_ctraj_input,
                                              0)) == false) {
      return false;
    }

    // Load the set of demonstrated Cartesian coordinate trajectories:
    if (rt_assert(CartesianCoordDMP::extractSetTrajectories(
            in_data_subdir_path, *set_ctraj_input, false)) == false) {
      return false;
    }

    // Learn the CartesianCoordDMP movement primitive from the demonstrated
    // Cartesian coordinate trajectories set:
    if (rt_assert(CartesianCoordDMP::learn(*set_ctraj_input,
                                           robot_task_servo_rate)) == false) {
      return false;
    }

    // Save the learned parameters of the CartesianCoordDMP movement primitive:
    if (rt_assert(CartesianCoordDMP::saveParams(out_data_subdir_path)) ==
        false) {
      return false;
    }

    if (save_unroll_traj_dataset) {
      // Set tau for movement reproduction:
      double dt = 1.0 / robot_task_servo_rate;

      // total number of states to be reproduced/unrolled:
      uint N_reprod_traj_states = round(tau_unroll * robot_task_servo_rate) + 1;

      // Set the critical states (start position and goal position):
      Trajectory critical_states(2);
      critical_states[0] =
          DMPState(CartesianCoordDMP::getMeanStartPosition(), rt_assertor);
      critical_states[1] =
          DMPState(CartesianCoordDMP::getMeanGoalPosition(), rt_assertor);

      // Start the BASELINE (w/o obstacle) DMP (including the computation of the
      // coordinate transformation):
      if (rt_assert(CartesianCoordDMP::start(critical_states, tau_unroll)) ==
          false) {
        return false;
      }

      // Reproduce/unroll the learned CartesianCoordDMP movement primitive
      // trajectory:
      std::cout << "Unrolling CartesianCoordDMP movement primitive #"
                << subdir_name << std::endl;
      if (rt_assert(CartesianCoordDMP::startCollectingTrajectoryDataSet()) ==
          false) {
        return false;
      }
      for (uint i = 0; i < N_reprod_traj_states; ++i) {
        // Get the next state of the Cartesian DMP
        DMPState endeff_cart_state_global(rt_assertor);

        if (rt_assert(CartesianCoordDMP::getNextState(
                dt, true, endeff_cart_state_global)) == false) {
          return false;
        }
      }
      CartesianCoordDMP::stopCollectingTrajectoryDataSet();
      if (rt_assert(CartesianCoordDMP::saveTrajectoryDataSet(
              out_data_subdir_path)) == false) {
        return false;
      }
    }
  }

  return true;
}

bool CartesianCoordDMP::start(const Trajectory& critical_states,
                              double tau_init) {
  // pre-conditions checking
  if (rt_assert(CartesianCoordDMP::isValid()) == false) {
    return false;
  }
  // input checking
  if (rt_assert((rt_assert(critical_states.size() >= 2)) &&
                (rt_assert(critical_states.size() <= MAX_TRAJ_SIZE))) ==
      false) {
    return false;
  }
  if (rt_assert(((ctraj_local_coord_selection == 3) &&
                 (critical_states.size() < 3)) == false) == false) {
    return false;
  }
  DMPState start_state_global_init = critical_states[0];
  DMPState approaching_ss_goal_state_global_init =
      critical_states[critical_states.size() -
                      2];  // approaching-steady-state goal state in the global
                           // coordinate frame
  DMPState ss_goal_state_global_init =
      critical_states[critical_states.size() -
                      1];  // steady-state goal state in the global coordinate
                           // frame
  if (rt_assert((rt_assert(start_state_global_init.isValid())) &&
                (rt_assert(approaching_ss_goal_state_global_init.isValid())) &&
                (rt_assert(ss_goal_state_global_init.isValid()))) == false) {
    return false;
  }
  if (rt_assert(
          (rt_assert(start_state_global_init.getDMPNumDimensions() == 3)) &&
          (rt_assert(
              approaching_ss_goal_state_global_init.getDMPNumDimensions() ==
              3)) &&
          (rt_assert(ss_goal_state_global_init.getDMPNumDimensions() == 3))) ==
      false) {
    return false;
  }
  if (rt_assert(tau_init >= MIN_TAU) ==
      false)  // tau_init is too small/does NOT make sense
  {
    return false;
  }

  // Compute critical states on the new local coordinate system:
  DMPState current_state_global_init = start_state_global_init;
  DMPState current_goal_state_global_init;
  if ((transform_sys_discrete_cart_coord.getFormulationType() ==
       _SCHAAL_DMP_) &&
      (canonical_sys_discrete->getOrder() == 2)) {
    current_goal_state_global_init = start_state_global_init;
  } else {
    current_goal_state_global_init = ss_goal_state_global_init;
  }
  ctraj_critical_states_global_coord[0] = start_state_global_init;
  ctraj_critical_states_global_coord[1] = current_state_global_init;
  ctraj_critical_states_global_coord[2] = current_goal_state_global_init;
  ctraj_critical_states_global_coord[ctraj_critical_states_global_coord.size() -
                                     2] = approaching_ss_goal_state_global_init;
  ctraj_critical_states_global_coord[ctraj_critical_states_global_coord.size() -
                                     1] = ss_goal_state_global_init;

  mean_start_global_position = start_state_global_init.getX();
  mean_goal_global_position = ss_goal_state_global_init.getX();

  // Compute new coordinate system transformation (representing new local
  // coordinate system) for trajectory reproduction:
  if (rt_assert(cart_coord_transformer.computeCTrajCoordTransforms(
          ctraj_critical_states_global_coord,
          ctraj_hmg_transform_local_to_global_matrix,
          ctraj_hmg_transform_global_to_local_matrix,
          ctraj_local_coord_selection)) == false) {
    return false;
  }

  if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(
          ctraj_critical_states_global_coord,
          ctraj_hmg_transform_global_to_local_matrix,
          ctraj_critical_states_local_coord)) == false) {
    return false;
  }

  if (rt_assert(tau_sys->setTauBase(tau_init)) == false) {
    return false;
  }
  if (rt_assert(canonical_sys_discrete->start()) == false) {
    return false;
  }
  if (rt_assert(transform_sys_discrete_cart_coord.start(
          ctraj_critical_states_local_coord[0],
          ctraj_critical_states_local_coord
              [ctraj_critical_states_local_coord.size() - 1])) == false) {
    return false;
  }

  mean_start_local_position = ctraj_critical_states_local_coord[0].getX();
  mean_goal_local_position = ctraj_critical_states_local_coord
                                 [ctraj_critical_states_local_coord.size() - 1]
                                     .getX();

  is_started = true;

  return true;
}

bool CartesianCoordDMP::getNextState(
    double dt, bool update_canonical_state, DMPState& result_dmp_state_global,
    VectorN* transform_sys_forcing_term,
    VectorN* transform_sys_coupling_term_acc,
    VectorN* transform_sys_coupling_term_vel,
    VectorM* func_approx_basis_functions,
    VectorM* func_approx_normalized_basis_func_vector_mult_phase_multiplier) {
  DMPState result_dmp_state_local(3, rt_assertor);

  return (rt_assert(CartesianCoordDMP::getNextState(
      dt, update_canonical_state, result_dmp_state_global,
      result_dmp_state_local, transform_sys_forcing_term,
      transform_sys_coupling_term_acc, transform_sys_coupling_term_vel,
      func_approx_basis_functions,
      func_approx_normalized_basis_func_vector_mult_phase_multiplier)));
}

bool CartesianCoordDMP::getNextState(
    double dt, bool update_canonical_state, DMPState& result_dmp_state_global,
    DMPState& result_dmp_state_local, VectorN* transform_sys_forcing_term,
    VectorN* transform_sys_coupling_term_acc,
    VectorN* transform_sys_coupling_term_vel,
    VectorM* func_approx_basis_functions,
    VectorM* func_approx_normalized_basis_func_vector_mult_phase_multiplier) {
  // pre-conditions checking
  if (rt_assert(is_started == true) == false) {
    return false;
  }
  if (rt_assert(CartesianCoordDMP::isValid()) == false) {
    return false;
  }

  if (rt_assert(transform_sys_discrete_cart_coord.getNextState(
          dt, result_dmp_state_local, transform_sys_forcing_term,
          transform_sys_coupling_term_acc, transform_sys_coupling_term_vel,
          func_approx_basis_functions,
          func_approx_normalized_basis_func_vector_mult_phase_multiplier)) ==
      false) {
    return false;
  }

  if (update_canonical_state) {
    if (rt_assert(canonical_sys_discrete->updateCanonicalState(dt)) == false) {
      return false;
    }
  }

  if (rt_assert(transform_sys_discrete_cart_coord.updateCurrentGoalState(dt)) ==
      false) {
    return false;
  }

  // update critical states in local coordinate frame (only current state and
  // current goal state changes)
  // ctraj_critical_states_local_coord[0]    =
  // transform_sys_discrete_cart_coord.getStartState();
  ctraj_critical_states_local_coord[1] =
      result_dmp_state_local;  // =
                               // transform_sys_discrete_cart_coord.getCurrentState();
  ctraj_critical_states_local_coord[2] =
      transform_sys_discrete_cart_coord.getCurrentGoalState();
  // ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-2]
  // =
  // ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-2];
  // ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-1]
  // = DMPState(transform_sys_discrete_cart_coord.getSteadyStateGoalPosition());

  // update critical states in global coordinate frame
  if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(
          ctraj_critical_states_local_coord,
          ctraj_hmg_transform_local_to_global_matrix,
          ctraj_critical_states_global_coord)) == false) {
    return false;
  }

  result_dmp_state_global = ctraj_critical_states_global_coord[1];

  /**************** For Trajectory Data Logging Purposes (Start)
   * ****************/
  logged_dmp_discrete_variables.transform_sys_state_global =
      result_dmp_state_global;
  logged_dmp_discrete_variables.goal_state_global =
      ctraj_critical_states_global_coord[2];
  logged_dmp_discrete_variables.steady_state_goal_position_global =
      ctraj_critical_states_global_coord
          [ctraj_critical_states_global_coord.size() - 1]
              .getX();

  if (is_collecting_trajectory_data) {
    if (rt_assert(data_logger_loader_discrete.collectTrajectoryDataSet(
            logged_dmp_discrete_variables)) == false) {
      return false;
    }
  }
  /**************** For Trajectory Data Logging Purposes (End) ****************/

  return true;
}

DMPState CartesianCoordDMP::getCurrentState() {
  // update critical states in local coordinate frame (only current state and
  // current goal state changes)
  // ctraj_critical_states_local_coord[0]    =
  // transform_sys_discrete_cart_coord.getStartState();
  ctraj_critical_states_local_coord[1] =
      transform_sys_discrete_cart_coord.getCurrentState();
  ctraj_critical_states_local_coord[2] =
      transform_sys_discrete_cart_coord.getCurrentGoalState();
  // ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-2]
  // =
  // ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-2];
  // ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-1]
  // = DMPState(transform_sys_discrete_cart_coord.getSteadyStateGoalPosition());

  // update critical states in global coordinate frame
  rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(
      ctraj_critical_states_local_coord,
      ctraj_hmg_transform_local_to_global_matrix,
      ctraj_critical_states_global_coord));

  return (ctraj_critical_states_global_coord[1]);
}

bool CartesianCoordDMP::getCurrentGoalPosition(Vector3& current_goal_global) {
  DMPState current_goal_state_local =
      transform_sys_discrete_cart_coord.getCurrentGoalState();
  DMPState current_goal_state_global(3, rt_assertor);

  // Compute current goal state on global coordinate system:
  if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(
          current_goal_state_local, ctraj_hmg_transform_local_to_global_matrix,
          current_goal_state_global)) == false) {
    return false;
  }

  current_goal_global = current_goal_state_global.getX().block(0, 0, 3, 1);
  return true;
}

bool CartesianCoordDMP::getSteadyStateGoalPosition(
    Vector3& steady_state_goal_global) {
  VectorN steady_state_goal_local(3);
  steady_state_goal_local = ZeroVector3;
  Vector4 steady_state_goal_local_H = ZeroVector4;
  Vector4 steady_state_goal_global_H = ZeroVector4;

  // Get steady-state goal position on the local coordinate system:
  if (rt_assert(DMPDiscrete::getSteadyStateGoalPosition(
          steady_state_goal_local)) == false) {
    return false;
  }
  steady_state_goal_local_H.block(0, 0, 3, 1) =
      steady_state_goal_local.block(0, 0, 3, 1);
  steady_state_goal_local_H(3, 0) = 1.0;
  steady_state_goal_global_H(3, 0) = 1.0;

  // Compute steady-state goal position on the global coordinate system:
  if (rt_assert(cart_coord_transformer.computeCTrajAtNewCoordSys(
          &steady_state_goal_local_H, NULL, NULL, NULL,
          ctraj_hmg_transform_local_to_global_matrix,
          &steady_state_goal_global_H, NULL, NULL, NULL)) == false) {
    return false;
  }

  steady_state_goal_global = steady_state_goal_global_H.block(0, 0, 3, 1);
  return true;
}

bool CartesianCoordDMP::setNewSteadyStateGoalPosition(const VectorN& new_G) {
  return (rt_assert(
      CartesianCoordDMP::setNewSteadyStateGoalPosition(new_G, new_G)));
}

bool CartesianCoordDMP::setNewSteadyStateGoalPosition(
    const Vector3& new_G_global, const Vector3& new_approaching_G_global) {
  ctraj_critical_states_local_coord[0] =
      transform_sys_discrete_cart_coord.getStartState();
  ctraj_critical_states_local_coord[1] =
      transform_sys_discrete_cart_coord.getCurrentState();
  ctraj_critical_states_local_coord[2] =
      transform_sys_discrete_cart_coord.getCurrentGoalState();
  ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size() -
                                    2] = ctraj_critical_states_local_coord
      [ctraj_critical_states_local_coord.size() - 2];
  ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size() -
                                    1] =
      DMPState(transform_sys_discrete_cart_coord.getSteadyStateGoalPosition(),
               rt_assertor);

  // Compute critical states on the global coordinate system
  // (using the OLD coordinate transformation):
  if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(
          ctraj_critical_states_local_coord,
          ctraj_hmg_transform_local_to_global_matrix,
          ctraj_critical_states_global_coord)) == false) {
    return false;
  }

  // Compute the NEW coordinate system transformation (representing the new
  // local coordinate system), based on the original start state, the new
  // approaching-steady-state goal, the new steady-state goal (G, NOT the
  // current/transient/non-steady-state goal), both on global coordinate system,
  // for trajectory reproduction:
  ctraj_critical_states_global_coord[ctraj_critical_states_global_coord.size() -
                                     2] =
      DMPState(new_approaching_G_global, rt_assertor);
  ctraj_critical_states_global_coord[ctraj_critical_states_global_coord.size() -
                                     1] = DMPState(new_G_global, rt_assertor);
  if (rt_assert(cart_coord_transformer.computeCTrajCoordTransforms(
          ctraj_critical_states_global_coord,
          ctraj_hmg_transform_local_to_global_matrix,
          ctraj_hmg_transform_global_to_local_matrix,
          ctraj_local_coord_selection)) == false) {
    return false;
  }

  // Compute the critical states on the NEW local coordinate system:
  if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(
          ctraj_critical_states_global_coord,
          ctraj_hmg_transform_global_to_local_matrix,
          ctraj_critical_states_local_coord)) == false) {
    return false;
  }

  if (rt_assert(transform_sys_discrete_cart_coord.setStartState(
          ctraj_critical_states_local_coord[0])) == false) {
    return false;
  }
  if (rt_assert(transform_sys_discrete_cart_coord.setCurrentState(
          ctraj_critical_states_local_coord[1])) == false) {
    return false;
  }
  if (rt_assert(transform_sys_discrete_cart_coord.setCurrentGoalState(
          ctraj_critical_states_local_coord[2])) == false) {
    return false;
  }
  if (rt_assert(transform_sys_discrete_cart_coord.setSteadyStateGoalPosition(
          ctraj_critical_states_local_coord
              [ctraj_critical_states_local_coord.size() - 1]
                  .getX())) == false) {
    return false;
  }

  return true;
}

bool CartesianCoordDMP::convertToGlobalDMPState(const DMPState& dmp_state_local,
                                                DMPState& dmp_state_global) {
  return (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(
      dmp_state_local, ctraj_hmg_transform_local_to_global_matrix,
      dmp_state_global)));
}

bool CartesianCoordDMP::convertToLocalDMPState(const DMPState& dmp_state_global,
                                               DMPState& dmp_state_local) {
  return (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(
      dmp_state_global, ctraj_hmg_transform_global_to_local_matrix,
      dmp_state_local)));
}

Matrix4x4 CartesianCoordDMP::getHomogeneousTransformMatrixLocalToGlobal() {
  return (ctraj_hmg_transform_local_to_global_matrix);
}

Matrix4x4 CartesianCoordDMP::getHomogeneousTransformMatrixGlobalToLocal() {
  return (ctraj_hmg_transform_global_to_local_matrix);
}

Vector3 CartesianCoordDMP::getMeanStartGlobalPosition() {
  return (mean_start_global_position);
}

Vector3 CartesianCoordDMP::getMeanGoalGlobalPosition() {
  return (mean_goal_global_position);
}

Vector3 CartesianCoordDMP::getMeanStartLocalPosition() {
  return (mean_start_local_position);
}

Vector3 CartesianCoordDMP::getMeanGoalLocalPosition() {
  return (mean_goal_local_position);
}

bool CartesianCoordDMP::writeCartCoordTrajectoryToFile(
    const Trajectory& cart_coord_trajectory, const char* file_path,
    bool is_comma_separated) {
  return (rt_assert(data_logger_loader_discrete.writeCartCoordTrajectoryToFile(
      cart_coord_trajectory, file_path, is_comma_separated)));
}

bool CartesianCoordDMP::extractSetTrajectories(const char* dir_or_file_path,
                                               TrajectorySet& trajectory_set,
                                               bool is_comma_separated,
                                               uint max_num_trajs) {
  return (rt_assert(data_logger_loader_discrete.extractSetCartCoordTrajectories(
      dir_or_file_path, trajectory_set, is_comma_separated, max_num_trajs)));
}

bool CartesianCoordDMP::writeSetCartCoordTrajectoriesToDir(
    const TrajectorySet& cart_coord_trajectory_set, const char* dir_path,
    bool is_comma_separated) {
  return (
      rt_assert(data_logger_loader_discrete.writeSetCartCoordTrajectoriesToDir(
          cart_coord_trajectory_set, dir_path, is_comma_separated)));
}

bool CartesianCoordDMP::saveParams(
    const char* dir_path, const char* file_name_weights,
    const char* file_name_A_learn,
    const char* file_name_mean_start_position_global,
    const char* file_name_mean_goal_position_global,
    const char* file_name_mean_tau,
    const char* file_name_mean_start_position_local,
    const char* file_name_mean_goal_position_local,
    const char* file_name_ctraj_hmg_transform_local_to_global_matrix,
    const char* file_name_ctraj_hmg_transform_global_to_local_matrix) {
  return (rt_assert(DMPDiscrete::saveParams(
              dir_path, file_name_weights, file_name_A_learn,
              file_name_mean_start_position_global,
              file_name_mean_goal_position_global, file_name_mean_tau)) &&
          rt_assert(data_logger_loader_discrete.writeMatrixToFile(
              dir_path, file_name_mean_start_position_local,
              mean_start_local_position)) &&
          rt_assert(data_logger_loader_discrete.writeMatrixToFile(
              dir_path, file_name_mean_goal_position_local,
              mean_goal_local_position)) &&
          rt_assert(data_logger_loader_discrete.writeMatrixToFile(
              dir_path, file_name_mean_start_position_global,
              mean_start_global_position)) &&
          rt_assert(data_logger_loader_discrete.writeMatrixToFile(
              dir_path, file_name_mean_goal_position_global,
              mean_goal_global_position)) &&
          rt_assert(data_logger_loader_discrete.writeMatrixToFile(
              dir_path, file_name_ctraj_hmg_transform_local_to_global_matrix,
              ctraj_hmg_transform_local_to_global_matrix)) &&
          rt_assert(data_logger_loader_discrete.writeMatrixToFile(
              dir_path, file_name_ctraj_hmg_transform_global_to_local_matrix,
              ctraj_hmg_transform_global_to_local_matrix)));
}

// bool CartesianCoordDMP::loadParams(const char* dir_path,
//                                    const char* file_name_weights,
//                                    const char* file_name_A_learn,
//                                    const char*
//                                    file_name_mean_start_position_global,
//                                    const char*
//                                    file_name_mean_goal_position_global, const
//                                    char* file_name_mean_tau, const char*
//                                    file_name_mean_start_position_local, const
//                                    char* file_name_mean_goal_position_local,
//                                    const char*
//                                    file_name_ctraj_hmg_transform_local_to_global_matrix,
//                                    const char*
//                                    file_name_ctraj_hmg_transform_global_to_local_matrix)
//{
//     return (rt_assert(DMPDiscrete::loadParams(dir_path,
//                                               file_name_weights,
//                                               file_name_A_learn,
//                                               file_name_mean_start_position_global,
//                                               file_name_mean_goal_position_global,
//                                               file_name_mean_tau))  &&
//             rt_assert(data_logger_loader_discrete.readMatrixFromFile(dir_path,
//             file_name_mean_start_position_local, mean_start_local_position))
//             &&
//             rt_assert(data_logger_loader_discrete.readMatrixFromFile(dir_path,
//             file_name_mean_goal_position_local, mean_goal_local_position)) &&
//             rt_assert(data_logger_loader_discrete.readMatrixFromFile(dir_path,
//             file_name_mean_start_position_global,
//             mean_start_global_position))    &&
//             rt_assert(data_logger_loader_discrete.readMatrixFromFile(dir_path,
//             file_name_mean_goal_position_global, mean_goal_global_position))
//             &&
//             rt_assert(data_logger_loader_discrete.readMatrixFromFile(dir_path,
//             file_name_ctraj_hmg_transform_local_to_global_matrix,
//             ctraj_hmg_transform_local_to_global_matrix))    &&
//             rt_assert(data_logger_loader_discrete.readMatrixFromFile(dir_path,
//             file_name_ctraj_hmg_transform_global_to_local_matrix,
//             ctraj_hmg_transform_global_to_local_matrix)));
// }

CartesianCoordDMP::~CartesianCoordDMP() {}

}  // namespace dmp
