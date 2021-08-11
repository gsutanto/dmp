#include "dmp/dmp_multi_dim/DMPDiscreteMultiDim.h"

namespace dmp {

DMPDiscreteMultiDim::DMPDiscreteMultiDim()
    : transform_sys_discrete_multi_dim(TransformSystemDiscrete(
          0, NULL, nullptr, &logged_dmp_discrete_variables, NULL,
          std::vector<bool>(0, true))),
      DMPDiscrete(0, 0, NULL, &transform_sys_discrete_multi_dim,
                  _SCHAAL_LWR_METHOD_, NULL, "") {}

DMPDiscreteMultiDim::DMPDiscreteMultiDim(
    uint dmp_num_dimensions_init, uint model_size_init,
    CanonicalSystemDiscrete* canonical_system_discrete,
    uint ts_formulation_type, uint learning_method,
    RealTimeAssertor* real_time_assertor, const char* opt_data_directory_path,
    std::vector<TransformCoupling*>* transform_couplers)
    : transform_sys_discrete_multi_dim(TransformSystemDiscrete(
          dmp_num_dimensions_init, canonical_system_discrete,
          std::make_unique<FuncApproximatorDiscrete>(
              dmp_num_dimensions_init, model_size_init,
              canonical_system_discrete, real_time_assertor),
          &logged_dmp_discrete_variables, real_time_assertor,
          std::vector<bool>(dmp_num_dimensions_init, true), NULL, NULL, NULL,
          NULL, transform_couplers, 25.0, 25.0 / 4.0, ts_formulation_type)),
      DMPDiscrete(dmp_num_dimensions_init, model_size_init,
                  canonical_system_discrete, &transform_sys_discrete_multi_dim,
                  learning_method, real_time_assertor,
                  opt_data_directory_path) {}

bool DMPDiscreteMultiDim::isValid() {
  if (rt_assert(transform_sys_discrete_multi_dim.isValid()) == false) {
    return false;
  }
  if (rt_assert(DMPDiscrete::isValid()) == false) {
    return false;
  }
  if (rt_assert(transform_sys_discrete == &transform_sys_discrete_multi_dim) ==
      false) {
    return false;
  }
  if (rt_assert(dmp_num_dimensions > 0) == false) {
    return false;
  }
  return true;
}

bool DMPDiscreteMultiDim::extractSetTrajectories(
    const char* dir_or_file_path, TrajectorySet& trajectory_set,
    bool is_comma_separated,  // unused function parameter
    uint max_num_trajs) {
  bool is_real_time = false;

  if (trajectory_set.size() != 0) {
    trajectory_set.clear();
  }

  TrajectoryPtr trajectory;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, trajectory)) ==
      false) {
    return false;
  }

  if (rt_assert(data_logger_loader_discrete.extractNDTrajectory(
          dir_or_file_path, *trajectory)) == false) {
    return false;
  }

  trajectory_set.push_back(*trajectory);

  return true;
}

bool DMPDiscreteMultiDim::learn(const char* training_data_dir_or_file_path,
                                double robot_task_servo_rate, double* tau_learn,
                                Trajectory* critical_states_learn) {
  return (rt_assert(DMP::learn(training_data_dir_or_file_path,
                               robot_task_servo_rate, tau_learn,
                               critical_states_learn)));
}

DMPDiscreteMultiDim::~DMPDiscreteMultiDim() {}

}  // namespace dmp
