#include "dmp/dmp_goal_system/QuaternionGoalSystem.h"

#include <memory>

namespace dmp {

QuaternionGoalSystem::QuaternionGoalSystem() : GoalSystem() {}

QuaternionGoalSystem::QuaternionGoalSystem(TauSystem* tau_system,
                                           RealTimeAssertor* real_time_assertor,
                                           double alpha_init)
    : GoalSystem(3, std::make_unique<QuaternionDMPState>(real_time_assertor), 4,
                 tau_system, real_time_assertor, alpha_init) {}

bool QuaternionGoalSystem::isValid() {
  if (rt_assert(GoalSystem::isValid()) == false) {
    return false;
  }
  if (rt_assert((static_cast<QuaternionDMPState*>(current_goal_state.get()))
                    ->isValid()) == false) {
    return false;
  }
  return true;
}

bool QuaternionGoalSystem::startQuaternionGoalSystem(
    const QuaternionDMPState& current_goal_state_init, const VectorN& G_init) {
  // pre-condition(s) checking
  if (rt_assert(this->isValid()) == false) {
    return false;
  }

  if (rt_assert(this->setCurrentQuaternionGoalState(current_goal_state_init)) ==
      false) {
    return false;
  }
  if (rt_assert(this->setSteadyStateGoalPosition(G_init)) == false) {
    return false;
  }

  is_started = true;

  // post-condition(s) checking
  return (rt_assert(this->isValid()));
}

QuaternionDMPState QuaternionGoalSystem::getCurrentQuaternionGoalState() {
  QuaternionDMPState current_quaternion_goal_state =
      *(static_cast<QuaternionDMPState*>(current_goal_state.get()));
  return current_quaternion_goal_state;
}

bool QuaternionGoalSystem::setCurrentQuaternionGoalState(
    const QuaternionDMPState& new_current_goal_state) {
  // pre-condition(s) checking
  if (rt_assert(this->isValid()) == false) {
    return false;
  }

  // input checking
  if (rt_assert(new_current_goal_state.isValid()) == false) {
    return false;
  }
  if (rt_assert(new_current_goal_state.getDMPNumDimensions() ==
                dmp_num_dimensions) &&
      rt_assert(new_current_goal_state.getX().rows() ==
                current_goal_state->getX().rows())) {
    *current_goal_state = new_current_goal_state;
  } else {
    return false;
  }

  // post-condition(s) checking
  return (rt_assert(this->isValid()));
}

bool QuaternionGoalSystem::updateCurrentGoalState(const double& dt) {
  // pre-condition(s) checking
  if (rt_assert(is_started == true) == false) {
    return false;
  }
  if (rt_assert(this->isValid()) == false) {
    return false;
  }
  // input checking
  if (rt_assert(dt > 0.0) == false)  // dt does NOT make sense
  {
    return false;
  }

  double tau;
  if (rt_assert(tau_sys->getTauRelative(tau)) == false) {
    return false;
  }

  // some aliases:
  QuaternionDMPState& current_quaternion_goal_state =
      *(static_cast<QuaternionDMPState*>(current_goal_state.get()));

  Vector4 QG = G;
  Vector4 Qg = current_quaternion_goal_state.getQ();
  Vector3 omegag = current_quaternion_goal_state.getOmega();
  // Vector3 omegagd = current_quaternion_goal_state->getOmegad();
  Vector3 log_quat_diff_g = ZeroVector3;
  if (rt_assert(computeLogQuaternionDifference(QG, Qg, log_quat_diff_g)) ==
      false) {
    return false;
  }
  Vector3 next_omegag = (alpha / tau) * log_quat_diff_g;
  Vector4 next_Qg = ZeroVector4;
  if (rt_assert(integrateQuaternion(Qg, next_omegag, dt, next_Qg)) == false) {
    return false;
  }
  Vector3 next_omegagd = (next_omegag - omegag) / dt;

  if (rt_assert(current_quaternion_goal_state.setQ(next_Qg)) == false) {
    return false;
  }
  if (rt_assert(current_quaternion_goal_state.setOmega(next_omegag)) == false) {
    return false;
  }
  if (rt_assert(current_quaternion_goal_state.setOmegad(next_omegagd)) ==
      false) {
    return false;
  }
  if (rt_assert(current_quaternion_goal_state.computeQdAndQdd()) == false) {
    return false;
  }

  // post-condition(s) checking
  return (rt_assert(this->isValid()));
}

QuaternionGoalSystem::~QuaternionGoalSystem() {}

}  // namespace dmp
