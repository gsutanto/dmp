#include "dmp/dmp_base/DMP.h"

#include <memory>

namespace dmp {

DMP::DMP()
    : dmp_num_dimensions(0),
      model_size(0),
      is_started(false),
      tau_sys(nullptr),
      canonical_sys(nullptr),
      transform_sys(nullptr),
      learning_sys(nullptr),
      data_logger_loader(nullptr),
      logged_dmp_variables(nullptr),
      mean_start_position(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
      mean_goal_position(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
      mean_tau(0.0),
      is_collecting_trajectory_data(false),
      rt_assertor(nullptr) {
  mean_start_position.resize(dmp_num_dimensions);
  mean_goal_position.resize(dmp_num_dimensions);
  snprintf(data_directory_path, sizeof(data_directory_path), "");
}

DMP::DMP(uint dmp_num_dimensions_init, uint model_size_init,
         CanonicalSystem* canonical_system,
         TransformationSystem* transform_system,
         LearningSystem* learning_system, DMPDataIO* dmp_data_io,
         LoggedDMPVariables* logged_dmp_vars,
         RealTimeAssertor* real_time_assertor,
         const char* opt_data_directory_path)
    : dmp_num_dimensions(dmp_num_dimensions_init),
      model_size(model_size_init),
      is_started(false),
      tau_sys(canonical_system->getTauSystemPointer()),
      canonical_sys(canonical_system),
      transform_sys(transform_system),
      learning_sys(learning_system),
      data_logger_loader(dmp_data_io),
      logged_dmp_variables(logged_dmp_vars),
      mean_start_position(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
      mean_goal_position(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
      mean_tau(0.0),
      is_collecting_trajectory_data(false),
      rt_assertor(real_time_assertor) {
  mean_start_position.resize(dmp_num_dimensions);
  mean_goal_position.resize(dmp_num_dimensions);
  snprintf(data_directory_path, sizeof(data_directory_path), "%s",
           opt_data_directory_path);
}

bool DMP::isValid() {
  if (rt_assert((rt_assert(dmp_num_dimensions > 0)) &&
                (rt_assert(dmp_num_dimensions <= MAX_DMP_NUM_DIMENSIONS))) ==
      false) {
    return false;
  }
  if (rt_assert((rt_assert(model_size > 0)) &&
                (rt_assert(model_size <= MAX_MODEL_SIZE))) == false) {
    return false;
  }
  if (rt_assert((rt_assert(tau_sys != nullptr)) &&
                (rt_assert(canonical_sys != nullptr)) &&
                (rt_assert(transform_sys != nullptr)) &&
                (rt_assert(learning_sys != nullptr)) &&
                (rt_assert(data_logger_loader != nullptr))) == false) {
    return false;
  }
  if (rt_assert((rt_assert(tau_sys->isValid())) &&
                (rt_assert(canonical_sys->isValid())) &&
                (rt_assert(transform_sys->isValid())) &&
                (rt_assert(learning_sys->isValid())) &&
                (rt_assert(data_logger_loader->isValid()))) == false) {
    return false;
  }
  if (rt_assert((rt_assert(transform_sys->getDMPNumDimensions() ==
                           dmp_num_dimensions)) &&
                (rt_assert(learning_sys->getDMPNumDimensions() ==
                           dmp_num_dimensions))) == false) {
    return false;
  }
  if (rt_assert((rt_assert(model_size == transform_sys->getModelSize())) &&
                (rt_assert(model_size == learning_sys->getModelSize()))) ==
      false) {
    return false;
  }
  if (rt_assert((rt_assert(tau_sys == canonical_sys->getTauSystemPointer())) &&
                (rt_assert(tau_sys == transform_sys->getTauSystemPointer()))) ==
      false) {
    return false;
  }
  if (rt_assert(rt_assert(canonical_sys ==
                          transform_sys->getCanonicalSystemPointer())) ==
      false) {
    return false;
  }
  if (rt_assert(transform_sys ==
                learning_sys->getTransformationSystemPointer()) == false) {
    return false;
  }
  if (rt_assert(
          (rt_assert(mean_start_position.rows() ==
                     transform_sys->getCurrentGoalState().getX().rows())) &&
          (rt_assert(mean_goal_position.rows() ==
                     transform_sys->getCurrentGoalState().getX().rows()))) ==
      false) {
    return false;
  }
  if (rt_assert(mean_tau >= 0.0) == false) {
    return false;
  }
  return true;
}

bool DMP::preprocess(TrajectorySet& trajectory_set) {
  // pre-conditions checking
  if (rt_assert(DMP::isValid()) == false) {
    return false;
  }

  int N_traj = trajectory_set.size();

  // computing average start and goal positions across trajectories in the set:
  mean_start_position = ZeroVectorN(dmp_num_dimensions);
  mean_goal_position = ZeroVectorN(dmp_num_dimensions);
  for (uint i = 0; i < N_traj; ++i) {
    int traj_size = trajectory_set[i].size();
    if (rt_assert((rt_assert(traj_size > 0)) &&
                  (rt_assert(traj_size <= MAX_TRAJ_SIZE))) == false) {
      return false;
    }
    if (rt_assert(
            (rt_assert(trajectory_set[i][0].getDMPNumDimensions() ==
                       dmp_num_dimensions)) &&
            (rt_assert(trajectory_set[i][traj_size - 1].getDMPNumDimensions() ==
                       dmp_num_dimensions))) == false) {
      return false;
    }
    mean_start_position +=
        trajectory_set[i][0].getX().block(0, 0, dmp_num_dimensions, 1);
    mean_goal_position += trajectory_set[i][traj_size - 1].getX().block(
        0, 0, dmp_num_dimensions, 1);
  }
  mean_start_position = mean_start_position / (N_traj * 1.0);
  mean_goal_position = mean_goal_position / (N_traj * 1.0);

  return true;
}

bool DMP::learn(const char* training_data_dir_or_file_path,
                double robot_task_servo_rate, double* tau_learn,
                Trajectory* critical_states_learn) {
  bool is_real_time = false;

  // Load sample input trajectory(ies):
  TrajectorySetPtr set_traj_input;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, set_traj_input)) ==
      false) {
    return false;
  }

  if (rt_assert(this->extractSetTrajectories(
          training_data_dir_or_file_path, *set_traj_input, false)) == false) {
    return false;
  }

  // Learn the trajectory
  if (rt_assert(this->learn(*set_traj_input, robot_task_servo_rate)) == false) {
    return false;
  }

  if (tau_learn != nullptr) {
    *tau_learn = mean_tau;
  }

  if (critical_states_learn != nullptr) {
    if (critical_states_learn->size() >= 2) {
      uint end_idx = critical_states_learn->size() - 1;

      (*critical_states_learn)[0] = DMPState(mean_start_position, rt_assertor);
      (*critical_states_learn)[end_idx] =
          DMPState(mean_goal_position, rt_assertor);
    }
  }

  return true;
}

bool DMP::start(const DMPUnrollInitParams& dmp_unroll_init_parameters) {
  // input checking
  if (rt_assert(dmp_unroll_init_parameters.isValid()) == false) {
    return false;
  }

  return (rt_assert(start(dmp_unroll_init_parameters.critical_states,
                          dmp_unroll_init_parameters.tau)));
}

bool DMP::getTargetCouplingTermAndUpdateStates(
    const DMPState& current_state_demo_local, const double& dt,
    VectorN& ct_acc_target) {
  if (rt_assert(is_started == true) == false) {
    return false;
  }
  if (rt_assert(transform_sys->getTargetCouplingTerm(current_state_demo_local,
                                                     ct_acc_target)) == false) {
    return false;
  }
  if (rt_assert(canonical_sys->updateCanonicalState(dt)) == false) {
    return false;
  }
  if (rt_assert(transform_sys->updateCurrentGoalState(dt)) == false) {
    return false;
  }

  return true;
}

VectorN DMP::getMeanStartPosition() { return (mean_start_position); }

VectorN DMP::getMeanGoalPosition() { return (mean_goal_position); }

double DMP::getMeanTau() { return (mean_tau); }

bool DMP::setDataDirectory(const char* new_data_dir_path) {
  if (rt_assert(createDirIfNotExistYet(new_data_dir_path)) == false) {
    return false;
  }

  snprintf(data_directory_path, sizeof(data_directory_path), "%s",
           new_data_dir_path);

  return true;
}

bool DMP::startCollectingTrajectoryDataSet() {
  if (rt_assert(data_logger_loader->reset()) == false) {
    return false;
  }

  is_collecting_trajectory_data = true;

  return true;
}

void DMP::stopCollectingTrajectoryDataSet() {
  is_collecting_trajectory_data = false;
}

bool DMP::saveTrajectoryDataSet(const char* new_data_dir_path) {
  // pre-conditions checking
  if (rt_assert(DMP::isValid()) == false) {
    return false;
  }

  if (strcmp(new_data_dir_path, "") == 0) {
    return (rt_assert(
        data_logger_loader->saveTrajectoryDataSet(data_directory_path)));
  } else {
    return (rt_assert(
        data_logger_loader->saveTrajectoryDataSet(new_data_dir_path)));
  }
}

bool DMP::setTransformSystemCouplingTermUsagePerDimensions(
    const std::vector<bool>&
        is_using_transform_sys_coupling_term_at_dimension_init) {
  return (rt_assert(transform_sys->setCouplingTermUsagePerDimensions(
      is_using_transform_sys_coupling_term_at_dimension_init)));
}

FunctionApproximator* DMP::getFunctionApproximatorPointer() {
  return transform_sys->getFuncApproxPointer();
}

DMP::~DMP() {}

}  // namespace dmp
