#include "dmp/cart_dmp/quat_dmp/QuaternionDMP.h"

namespace dmp {

QuaternionDMP::QuaternionDMP()
    : transform_sys_discrete_quat(TransformSystemQuaternion(
          NULL, &func_approx_discrete, &logged_dmp_discrete_variables, NULL,
        std::vector<bool>(3, true), NULL)),
      DMPDiscrete(1, 0, NULL, &transform_sys_discrete_quat, _SCHAAL_LWR_METHOD_,
                  NULL, "") {
  mean_start_position.resize(4);
  mean_goal_position.resize(4);
}

QuaternionDMP::QuaternionDMP(
    uint model_size_init, CanonicalSystemDiscrete* canonical_system_discrete,
    uint learning_method, RealTimeAssertor* real_time_assertor,
    const char* opt_data_directory_path,
    std::vector<TransformCoupling*>* transform_couplers)
    : transform_sys_discrete_quat(TransformSystemQuaternion(
          canonical_system_discrete, &func_approx_discrete,
          &logged_dmp_discrete_variables, real_time_assertor,
          std::vector<bool>(3, true), transform_couplers)),
      DMPDiscrete(3, model_size_init, canonical_system_discrete,
                  &transform_sys_discrete_quat, learning_method,
                  real_time_assertor, opt_data_directory_path) {
  mean_start_position.resize(4);
  mean_goal_position.resize(4);
}

bool QuaternionDMP::isValid() {
  if (rt_assert(transform_sys_discrete_quat.isValid()) == false) {
    return false;
  }
  if (rt_assert(DMPDiscrete::isValid()) == false) {
    return false;
  }
  if (rt_assert(transform_sys_discrete == &transform_sys_discrete_quat) ==
      false) {
    return false;
  }
  if (rt_assert(dmp_num_dimensions == 3) == false) {
    return false;
  }
  return true;
}

bool QuaternionDMP::startQuaternionDMP(
    const QuaternionTrajectory& critical_states, double tau_init) {
  // pre-conditions checking
  if (rt_assert(this->isValid()) == false) {
    return false;
  }
  // input checking
  uint traj_size = critical_states.size();
  if (rt_assert((rt_assert(traj_size >= 2)) &&
                (rt_assert(traj_size <= MAX_TRAJ_SIZE))) == false) {
    return false;
  }
  QuaternionDMPState start_state_init = critical_states[0];
  QuaternionDMPState goal_state_init = critical_states[traj_size - 1];
  if (rt_assert((rt_assert(start_state_init.isValid())) &&
                (rt_assert(goal_state_init.isValid()))) == false) {
    return false;
  }
  if (rt_assert((rt_assert(start_state_init.getDMPNumDimensions() ==
                           dmp_num_dimensions)) &&
                (rt_assert(goal_state_init.getDMPNumDimensions() ==
                           dmp_num_dimensions))) == false) {
    return false;
  }
  if (rt_assert(tau_init >= MIN_TAU) ==
      false)  // tau_init is too small/does NOT make sense
  {
    return false;
  }

  if (rt_assert(tau_sys->setTauBase(tau_init)) == false) {
    return false;
  }
  if (rt_assert(canonical_sys_discrete->start()) == false) {
    return false;
  }
  if (rt_assert(transform_sys_discrete_quat.startTransformSystemQuaternion(
          start_state_init, goal_state_init)) == false) {
    return false;
  }

  is_started = true;

  // post-conditions checking
  return (rt_assert(this->isValid()));
}

bool QuaternionDMP::getNextQuaternionState(
    double dt, bool update_canonical_state, QuaternionDMPState& next_state,
    VectorN* transform_sys_forcing_term,
    VectorN* transform_sys_coupling_term_acc,
    VectorN* transform_sys_coupling_term_vel,
    VectorM* func_approx_basis_functions,
    VectorM* func_approx_normalized_basis_func_vector_mult_phase_multiplier) {
  // pre-conditions checking
  if (rt_assert(is_started == true) == false) {
    return false;
  }
  if (rt_assert(this->isValid()) == false) {
    return false;
  }

  QuaternionDMPState result_dmp_state(rt_assertor);
  if (rt_assert(transform_sys_discrete_quat.getNextQuaternionState(
          dt, result_dmp_state, transform_sys_forcing_term,
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

  if (rt_assert(transform_sys_discrete_quat.updateCurrentGoalState(dt)) ==
      false) {
    return false;
  }

  next_state = result_dmp_state;

  /**************** For Trajectory Data Logging Purposes (Start)
   * ****************/
  //        logged_dmp_discrete_variables.transform_sys_state_global        =
  //        logged_dmp_discrete_variables.transform_sys_state_local;
  //        logged_dmp_discrete_variables.goal_state_global                 =
  //        logged_dmp_discrete_variables.goal_state_local;
  //        logged_dmp_discrete_variables.steady_state_goal_position_global =
  //        logged_dmp_discrete_variables.steady_state_goal_position_local;
  //
  //        if (is_collecting_trajectory_data)
  //        {
  //            if
  //            (rt_assert(data_logger_loader_discrete.collectTrajectoryDataSet(logged_dmp_discrete_variables))
  //            == false)
  //            {
  //                return false;
  //            }
  //        }
  /**************** For Trajectory Data Logging Purposes (End) ****************/
  // post-conditions checking
  return (rt_assert(this->isValid()));
}

QuaternionDMPState QuaternionDMP::getCurrentQuaternionState() {
  return (transform_sys_discrete_quat.getCurrentQuaternionState());
}

bool QuaternionDMP::getTargetQuaternionCouplingTermAndUpdateStates(
    const QuaternionDMPState& current_state_demo_local, const double& dt,
    VectorN& ct_acc_target) {
  if (rt_assert(is_started == true) == false) {
    return false;
  }
  if (rt_assert(transform_sys_discrete_quat.getTargetQuaternionCouplingTerm(
          current_state_demo_local, ct_acc_target)) == false) {
    return false;
  }
  if (rt_assert(canonical_sys->updateCanonicalState(dt)) == false) {
    return false;
  }
  if (rt_assert(transform_sys_discrete_quat.updateCurrentGoalState(dt)) ==
      false) {
    return false;
  }

  return true;
}

bool QuaternionDMP::extractSetTrajectories(const char* dir_or_file_path,
                                           TrajectorySet& trajectory_set,
                                           bool is_comma_separated,
                                           uint max_num_trajs) {
  std::cout
      << "ERROR: Method extractSetTrajectories(...) has NOT been implemented!"
      << std::endl;
  return false;
}

QuaternionDMP::~QuaternionDMP() {}

}  // namespace dmp
