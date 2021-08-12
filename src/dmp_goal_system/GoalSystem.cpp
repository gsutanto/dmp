#include "dmp/dmp_goal_system/GoalSystem.h"

#include <memory>

namespace dmp {

GoalSystem::GoalSystem()
    : dmp_num_dimensions(0),
      alpha(25.0 / 2.0),
      current_goal_state(nullptr),
      G(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
      is_started(false),
      tau_sys(NULL),
      rt_assertor(NULL) {
  G.resize(0);
}

GoalSystem::GoalSystem(uint dmp_num_dimensions_init, TauSystem* tau_system,
                       RealTimeAssertor* real_time_assertor, double alpha_init)
    : dmp_num_dimensions(dmp_num_dimensions_init),
      current_goal_state(std::make_unique<DMPState>(dmp_num_dimensions_init,
                                                    real_time_assertor)),
      G(ZeroVectorN(dmp_num_dimensions_init)),
      is_started(false),
      tau_sys(tau_system),
      alpha(alpha_init),
      rt_assertor(real_time_assertor) {}

GoalSystem::GoalSystem(uint dmp_num_dimensions_init,
                       std::unique_ptr<DMPState> current_goal_state_ptr,
                       uint goal_num_dimensions_init, TauSystem* tau_system,
                       RealTimeAssertor* real_time_assertor, double alpha_init)
    : dmp_num_dimensions(dmp_num_dimensions_init),
      current_goal_state(std::move(current_goal_state_ptr)),
      G(ZeroVectorN(goal_num_dimensions_init)),
      is_started(false),
      tau_sys(tau_system),
      alpha(alpha_init),
      rt_assertor(real_time_assertor) {}

bool GoalSystem::isValid() {
  if (rt_assert((rt_assert(dmp_num_dimensions > 0)) &&
                (rt_assert(dmp_num_dimensions <= MAX_DMP_NUM_DIMENSIONS))) ==
      false) {
    return false;
  }
  if (rt_assert(current_goal_state != NULL) == false) {
    return false;
  }
  if (rt_assert(current_goal_state->isValid()) == false) {
    return false;
  }
  if (rt_assert(current_goal_state->getDMPNumDimensions() ==
                dmp_num_dimensions) == false) {
    return false;
  }
  if (rt_assert(G.rows() == current_goal_state->getX().rows()) == false) {
    return false;
  }
  if (rt_assert(tau_sys != NULL) == false) {
    return false;
  }
  if (rt_assert(tau_sys->isValid()) == false) {
    return false;
  }
  if (rt_assert(alpha > 0.0) == false) {
    return false;
  }
  return true;
}

bool GoalSystem::start(const DMPState& current_goal_state_init,
                       const VectorN& G_init) {
  // pre-condition(s) checking
  if (rt_assert(this->isValid()) == false) {
    return false;
  }

  if (rt_assert(this->setCurrentGoalState(current_goal_state_init)) == false) {
    return false;
  }
  if (rt_assert(this->setSteadyStateGoalPosition(G_init)) == false) {
    return false;
  }

  is_started = true;

  // post-condition(s) checking
  return (rt_assert(this->isValid()));
}

uint GoalSystem::getDMPNumDimensions() { return dmp_num_dimensions; }

VectorN GoalSystem::getSteadyStateGoalPosition() { return G; }

bool GoalSystem::setSteadyStateGoalPosition(const VectorN& new_G) {
  // pre-condition(s) checking
  if (rt_assert(this->isValid()) == false) {
    return false;
  }

  if (rt_assert(new_G.rows() == G.rows())) {
    G = new_G;
  } else {
    return false;
  }

  // post-condition(s) checking
  return (rt_assert(this->isValid()));
}

DMPState GoalSystem::getCurrentGoalState() { return (*current_goal_state); }

bool GoalSystem::setCurrentGoalState(const DMPState& new_current_goal_state) {
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

bool GoalSystem::updateCurrentGoalState(const double& dt) {
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

  VectorN g = current_goal_state->getX();
  VectorN gd = alpha * (G - g) / tau;
  g = g + (gd * dt);

  if (rt_assert(current_goal_state->setX(g)) == false) {
    return false;
  }
  if (rt_assert(current_goal_state->setXd(gd)) == false) {
    return false;
  }
  if (rt_assert(current_goal_state->setTime(current_goal_state->getTime() +
                                            dt)) == false) {
    return false;
  }

  // post-condition(s) checking
  return (rt_assert(this->isValid()));
}

TauSystem* GoalSystem::getTauSystemPointer() { return tau_sys; }

GoalSystem::~GoalSystem() {}

}  // namespace dmp
