#include "dmp/dmp_1D/DMPDiscrete1D.h"

#include <memory>

namespace dmp {

DMPDiscrete1D::DMPDiscrete1D()
    : transform_sys_discrete_1D(TransformSystemDiscrete(
          1, NULL, nullptr, &logged_dmp_discrete_variables, NULL,
          std::vector<bool>(1, true))),
      DMPDiscrete(1, 0, NULL, &transform_sys_discrete_1D, _SCHAAL_LWR_METHOD_,
                  NULL, "") {}

DMPDiscrete1D::DMPDiscrete1D(
    uint model_size_init, CanonicalSystemDiscrete* canonical_system_discrete,
    uint ts_formulation_type, uint learning_method,
    RealTimeAssertor* real_time_assertor, const char* opt_data_directory_path,
    std::vector<TransformCoupling*>* transform_couplers)
    : transform_sys_discrete_1D(TransformSystemDiscrete(
          1, canonical_system_discrete,
          std::make_unique<FuncApproximatorDiscrete>(1, model_size_init,
                                                     canonical_system_discrete,
                                                     real_time_assertor),
          &logged_dmp_discrete_variables, real_time_assertor,
          std::vector<bool>(1, true), NULL, NULL, NULL, NULL,
          transform_couplers, 25.0, 25.0 / 4.0, ts_formulation_type)),
      DMPDiscrete(1, model_size_init, canonical_system_discrete,
                  &transform_sys_discrete_1D, learning_method,
                  real_time_assertor, opt_data_directory_path) {}

bool DMPDiscrete1D::isValid() {
  if (rt_assert(transform_sys_discrete_1D.isValid()) == false) {
    return false;
  }
  if (rt_assert(DMPDiscrete::isValid()) == false) {
    return false;
  }
  if (rt_assert(transform_sys_discrete == &transform_sys_discrete_1D) ==
      false) {
    return false;
  }
  if (rt_assert(dmp_num_dimensions == 1) == false) {
    return false;
  }
  return true;
}

bool DMPDiscrete1D::extractSetTrajectories(const char* dir_or_file_path,
                                           TrajectorySet& trajectory_set,
                                           bool is_comma_separated,
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

  if (rt_assert(data_logger_loader_discrete.extract1DTrajectory(
          dir_or_file_path, *trajectory, is_comma_separated)) == false) {
    return false;
  }

  trajectory_set.push_back(*trajectory);

  return true;
}

bool DMPDiscrete1D::write1DTrajectoryToFile(const Trajectory& trajectory,
                                            const char* file_path,
                                            bool is_comma_separated) {
  return (rt_assert(data_logger_loader_discrete.write1DTrajectoryToFile(
      trajectory, file_path, is_comma_separated)));
}

bool DMPDiscrete1D::learn(const char* training_data_dir_or_file_path,
                          double robot_task_servo_rate, double* tau_learn,
                          Trajectory* critical_states_learn) {
  return (rt_assert(DMP::learn(training_data_dir_or_file_path,
                               robot_task_servo_rate, tau_learn,
                               critical_states_learn)));
}

DMPDiscrete1D::~DMPDiscrete1D() {}

}  // namespace dmp
