#include "dmp/dmp_discrete/DMPDiscrete.h"

namespace dmp {

DMPDiscrete::DMPDiscrete()
    : DMP(),
      canonical_sys_discrete(nullptr),
      transform_sys_discrete(nullptr),
      learning_sys_discrete(LearningSystemDiscrete()),
      data_logger_loader_discrete(DMPDataIODiscrete()),
      logged_dmp_discrete_variables(LoggedDMPDiscreteVariables()) {}

DMPDiscrete::DMPDiscrete(uint dmp_num_dimensions_init, uint model_size_init,
                         CanonicalSystemDiscrete* canonical_system_discrete,
                         TransformSystemDiscrete* transform_system_discrete,
                         uint learning_method,
                         RealTimeAssertor* real_time_assertor,
                         const char* opt_data_directory_path)
    : DMP(dmp_num_dimensions_init, model_size_init, canonical_system_discrete,
          transform_system_discrete, &learning_sys_discrete,
          &data_logger_loader_discrete, &logged_dmp_discrete_variables,
          real_time_assertor, opt_data_directory_path),
      canonical_sys_discrete(canonical_system_discrete),
      transform_sys_discrete(transform_system_discrete),
      learning_sys_discrete(LearningSystemDiscrete(
          dmp_num_dimensions_init, model_size_init, transform_system_discrete,
          &data_logger_loader_discrete, learning_method, real_time_assertor,
          opt_data_directory_path)),
      data_logger_loader_discrete(
          DMPDataIODiscrete(dmp_num_dimensions_init, model_size_init,
                            transform_system_discrete, real_time_assertor)),
      logged_dmp_discrete_variables(LoggedDMPDiscreteVariables(
          dmp_num_dimensions_init, model_size_init, real_time_assertor)) {}

bool DMPDiscrete::isValid() {
  if (rt_assert(DMP::isValid()) == false) {
    return false;
  }
  if (rt_assert((rt_assert(canonical_sys_discrete != nullptr)) &&
                (rt_assert(transform_sys_discrete != nullptr))) == false) {
    return false;
  }
  if (rt_assert((rt_assert(canonical_sys_discrete->isValid())) &&
                (rt_assert(transform_sys_discrete->isValid())) &&
                (rt_assert(learning_sys_discrete.isValid())) &&
                (rt_assert(data_logger_loader_discrete.isValid()))) == false) {
    return false;
  }
  if (rt_assert(rt_assert(
          transform_sys_discrete ==
          data_logger_loader_discrete.getTransformSysDiscretePointer())) ==
      false) {
    return false;
  }
  return true;
}

bool DMPDiscrete::learn(const TrajectorySet& trajectory_set,
                        double robot_task_servo_rate) {
  bool is_real_time = false;

  // pre-conditions checking
  if (rt_assert(DMPDiscrete::isValid()) == false) {
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
  if (rt_assert(DMPDiscrete::preprocess(*trajectory_set_preprocessed)) ==
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

bool DMPDiscrete::start(const Trajectory& critical_states, double tau_init) {
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
  DMPState start_state_init = critical_states[0];
  DMPState goal_state_init = critical_states[traj_size - 1];
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
  if (rt_assert(transform_sys_discrete->start(start_state_init,
                                              goal_state_init)) == false) {
    return false;
  }

  is_started = true;

  // post-conditions checking
  return (rt_assert(this->isValid()));
}

bool DMPDiscrete::getNextState(
    double dt, bool update_canonical_state, DMPState& next_state,
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

  DMPState result_dmp_state(dmp_num_dimensions, rt_assertor);
  if (rt_assert(transform_sys_discrete->getNextState(
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

  if (rt_assert(transform_sys_discrete->updateCurrentGoalState(dt)) == false) {
    return false;
  }

  next_state = result_dmp_state;

  /**************** For Trajectory Data Logging Purposes (Start)
   * ****************/
  logged_dmp_discrete_variables.transform_sys_state_global =
      logged_dmp_discrete_variables.transform_sys_state_local;
  logged_dmp_discrete_variables.goal_state_global =
      logged_dmp_discrete_variables.goal_state_local;
  logged_dmp_discrete_variables.steady_state_goal_position_global =
      logged_dmp_discrete_variables.steady_state_goal_position_local;

  if (is_collecting_trajectory_data) {
    if (rt_assert(data_logger_loader_discrete.collectTrajectoryDataSet(
            logged_dmp_discrete_variables)) == false) {
      return false;
    }
  }
  /**************** For Trajectory Data Logging Purposes (End) ****************/
  // post-conditions checking
  return (rt_assert(this->isValid()));
}

DMPState DMPDiscrete::getCurrentState() {
  return (transform_sys_discrete->getCurrentState());
}

bool DMPDiscrete::getForcingTerm(VectorN& result_f,
                                 VectorM* basis_function_vector) {
  return (
      rt_assert(transform_sys_discrete->getFuncApproxPointer()->getForcingTerm(
          result_f, basis_function_vector)));
}

bool DMPDiscrete::getCurrentGoalPosition(VectorN& current_goal) {
  // pre-conditions checking
  if (rt_assert(DMPDiscrete::isValid()) == false) {
    return false;
  }

  current_goal = transform_sys_discrete->getCurrentGoalState().getX();
  return true;
}

bool DMPDiscrete::getSteadyStateGoalPosition(VectorN& steady_state_goal) {
  // pre-conditions checking
  if (rt_assert(DMPDiscrete::isValid()) == false) {
    return false;
  }

  steady_state_goal = transform_sys_discrete->getSteadyStateGoalPosition();
  return true;
}

bool DMPDiscrete::setNewSteadyStateGoalPosition(const VectorN& new_G) {
  // pre-conditions checking
  if (rt_assert(DMPDiscrete::isValid()) == false) {
    return false;
  }

  return (rt_assert(transform_sys_discrete->setSteadyStateGoalPosition(new_G)));
}

bool DMPDiscrete::setScalingUsage(
    const std::vector<bool>& is_using_scaling_init) {
  return (rt_assert(
      transform_sys_discrete->setScalingUsage(is_using_scaling_init)));
}

bool DMPDiscrete::getParams(MatrixNxM& weights_buffer,
                            VectorN& A_learn_buffer) {
  A_learn_buffer = transform_sys_discrete->getLearningAmplitude();

  return (rt_assert(DMPDiscrete::getWeights(weights_buffer)));
}

bool DMPDiscrete::setParams(const MatrixNxM& new_weights,
                            const VectorN& new_A_learn) {
  if (rt_assert(transform_sys_discrete->setLearningAmplitude(new_A_learn)) ==
      false) {
    return false;
  }

  return (rt_assert(DMPDiscrete::setWeights(new_weights)));
}

bool DMPDiscrete::saveParams(const char* dir_path,
                             const char* file_name_weights,
                             const char* file_name_A_learn,
                             const char* file_name_mean_start_position,
                             const char* file_name_mean_goal_position,
                             const char* file_name_mean_tau) {
  return (rt_assert(data_logger_loader_discrete.saveWeights(
              dir_path, file_name_weights)) &&
          rt_assert(data_logger_loader_discrete.saveALearn(
              dir_path, file_name_A_learn)) &&
          rt_assert(data_logger_loader_discrete.writeMatrixToFile(
              dir_path, file_name_mean_start_position, mean_start_position)) &&
          rt_assert(data_logger_loader_discrete.writeMatrixToFile(
              dir_path, file_name_mean_goal_position, mean_goal_position)) &&
          rt_assert(data_logger_loader_discrete.writeMatrixToFile(
              dir_path, file_name_mean_tau, mean_tau)));
}

bool DMPDiscrete::loadParams(const char* dir_path,
                             const char* file_name_weights,
                             const char* file_name_A_learn,
                             const char* file_name_mean_start_position,
                             const char* file_name_mean_goal_position,
                             const char* file_name_mean_tau) {
  return (rt_assert(data_logger_loader_discrete.loadWeights(
              dir_path, file_name_weights)) &&
          rt_assert(data_logger_loader_discrete.loadALearn(
              dir_path, file_name_A_learn)) &&
          rt_assert(data_logger_loader_discrete.readMatrixFromFile(
              dir_path, file_name_mean_start_position, mean_start_position)) &&
          rt_assert(data_logger_loader_discrete.readMatrixFromFile(
              dir_path, file_name_mean_goal_position, mean_goal_position)) &&
          rt_assert(data_logger_loader_discrete.readMatrixFromFile(
              dir_path, file_name_mean_tau, mean_tau)));
}

bool DMPDiscrete::saveBasisFunctions(const char* file_path) {
  return (rt_assert(data_logger_loader_discrete.saveBasisFunctions(file_path)));
}

bool DMPDiscrete::getWeights(MatrixNxM& weights_buffer) {
  return (rt_assert(transform_sys_discrete->getFuncApproxPointer()->getWeights(
      weights_buffer)));
}

bool DMPDiscrete::setWeights(const MatrixNxM& new_weights) {
  return (rt_assert(
      transform_sys_discrete->getFuncApproxPointer()->setWeights(new_weights)));
}

bool DMPDiscrete::startCollectingTrajectoryDataSet() {
  if (rt_assert(data_logger_loader_discrete.reset()) == false) {
    return false;
  }

  is_collecting_trajectory_data = true;

  return true;
}

bool DMPDiscrete::saveTrajectoryDataSet(const char* new_data_dir_path) {
  // pre-conditions checking
  if (rt_assert(DMPDiscrete::isValid()) == false) {
    return false;
  }

  if (strcmp(new_data_dir_path, "") == 0) {
    return (rt_assert(data_logger_loader_discrete.saveTrajectoryDataSet(
        data_directory_path)));
  } else {
    return (rt_assert(
        data_logger_loader_discrete.saveTrajectoryDataSet(new_data_dir_path)));
  }
}

uint DMPDiscrete::getCanonicalSysOrder() {
  return (canonical_sys_discrete->getOrder());
}

uint DMPDiscrete::getFuncApproxModelSize() {
  return (transform_sys_discrete->getModelSize());
}

uint DMPDiscrete::getTransformSysFormulationType() {
  return (transform_sys_discrete->getFormulationType());
}

uint DMPDiscrete::getLearningSysMethod() {
  return (learning_sys_discrete.getLearningMethod());
}

FuncApproximatorDiscrete* DMPDiscrete::getFuncApproxDiscretePointer() {
  return (transform_sys_discrete->getFuncApproxDiscretePointer());
}

DMPDiscrete::~DMPDiscrete() {}

}  // namespace dmp
