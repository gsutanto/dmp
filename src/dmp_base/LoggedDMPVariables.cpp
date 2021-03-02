#include "dmp/dmp_base/LoggedDMPVariables.h"

namespace dmp {

LoggedDMPVariables::LoggedDMPVariables()
    : dmp_num_dimensions(0),
      model_size(0),
      rt_assertor(NULL),
      tau(0.0),
      transform_sys_state_local(DMPState()),
      transform_sys_state_global(DMPState()),
      goal_state_local(DMPState()),
      goal_state_global(DMPState()),
      steady_state_goal_position_local(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
      steady_state_goal_position_global(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
      basis_functions(ZeroVectorM(MAX_MODEL_SIZE)),
      forcing_term(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
      transform_sys_ct_acc(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)) {
  steady_state_goal_position_local.resize(dmp_num_dimensions);
  steady_state_goal_position_global.resize(dmp_num_dimensions);
  basis_functions.resize(model_size);
  forcing_term.resize(dmp_num_dimensions);
  transform_sys_ct_acc.resize(dmp_num_dimensions);
}

LoggedDMPVariables::LoggedDMPVariables(uint dmp_num_dimensions_init,
                                       uint num_basis_functions,
                                       RealTimeAssertor* real_time_assertor)
    : dmp_num_dimensions(dmp_num_dimensions_init),
      model_size(num_basis_functions),
      rt_assertor(real_time_assertor),
      tau(0.0),
      transform_sys_state_local(DMPState(dmp_num_dimensions, rt_assertor)),
      transform_sys_state_global(DMPState(dmp_num_dimensions, rt_assertor)),
      goal_state_local(DMPState(dmp_num_dimensions, rt_assertor)),
      goal_state_global(DMPState(dmp_num_dimensions, rt_assertor)),
      steady_state_goal_position_local(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
      steady_state_goal_position_global(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
      basis_functions(ZeroVectorM(MAX_MODEL_SIZE)),
      forcing_term(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
      transform_sys_ct_acc(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)) {
  steady_state_goal_position_local.resize(dmp_num_dimensions);
  steady_state_goal_position_global.resize(dmp_num_dimensions);
  basis_functions.resize(model_size);
  forcing_term.resize(dmp_num_dimensions);
  transform_sys_ct_acc.resize(dmp_num_dimensions);
}

bool LoggedDMPVariables::isValid() const {
  if (rt_assert((rt_assert(dmp_num_dimensions > 0)) &&
                (rt_assert(dmp_num_dimensions <= MAX_DMP_NUM_DIMENSIONS))) ==
      false) {
    return false;
  }
  if (rt_assert((rt_assert(model_size > 0)) &&
                (rt_assert(model_size <= MAX_MODEL_SIZE))) == false) {
    return false;
  }
  if (rt_assert(tau >= MIN_TAU) == false) {
    return false;
  }
  if (rt_assert((rt_assert(transform_sys_state_local.isValid())) &&
                (rt_assert(transform_sys_state_global.isValid())) &&
                (rt_assert(goal_state_local.isValid())) &&
                (rt_assert(goal_state_global.isValid()))) == false) {
    return false;
  }
  if (rt_assert((rt_assert(dmp_num_dimensions ==
                           transform_sys_state_local.getDMPNumDimensions())) &&
                (rt_assert(dmp_num_dimensions ==
                           transform_sys_state_global.getDMPNumDimensions())) &&
                (rt_assert(dmp_num_dimensions ==
                           goal_state_local.getDMPNumDimensions())) &&
                (rt_assert(dmp_num_dimensions ==
                           goal_state_global.getDMPNumDimensions())) &&
                (rt_assert(dmp_num_dimensions ==
                           steady_state_goal_position_local.rows())) &&
                (rt_assert(dmp_num_dimensions ==
                           steady_state_goal_position_global.rows())) &&
                (rt_assert(model_size == basis_functions.rows())) &&
                (rt_assert(dmp_num_dimensions == forcing_term.rows())) &&
                (rt_assert(dmp_num_dimensions ==
                           transform_sys_ct_acc.rows()))) == false) {
    return false;
  }
  return true;
}

uint LoggedDMPVariables::getDMPNumDimensions() const {
  return (dmp_num_dimensions);
}

uint LoggedDMPVariables::getModelSize() const { return (model_size); }

LoggedDMPVariables::~LoggedDMPVariables() {}

}  // namespace dmp
