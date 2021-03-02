/*
 * QuaternionGoalSystem.h
 *
 * System that represents the current Quaternion DMP goal state (including
 * orientation, angular velocity, angular acceleration, and time (if used)) and
 * its evolution toward the final (steady-state) Quaternion goal position.
 *
 *  Created on: June 21, 2017
 *  Author: Giovanni Sutanto
 */

#ifndef QUATERNION_GOAL_SYSTEM_H
#define QUATERNION_GOAL_SYSTEM_H

#include "dmp/dmp_goal_system/GoalSystem.h"
#include "dmp/dmp_state/QuaternionDMPState.h"
#include "dmp/utility/utility_quaternion.h"

namespace dmp {

class QuaternionGoalSystem : public GoalSystem {
 protected:
  QuaternionDMPState
      current_quaternion_goal_state;  // current Quaternion DMP goal state
                                      // (orientation, angular velocity, angular
                                      // acceleration, and time (if used))

 public:
  QuaternionGoalSystem();

  /**
   * @param tau_system Tau system that computes the tau parameter for this
   * goal-changing system
   * @param alpha_init Spring parameter (default value = 25/2)
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  QuaternionGoalSystem(TauSystem* tau_system,
                       RealTimeAssertor* real_time_assertor,
                       double alpha_init = 25.0 / 2.0);

  /**
   * Checks whether this tau system is valid or not.
   *
   * @return Tau system is valid (true) or tau system is invalid (false)
   */
  bool isValid();

  /**
   * Start the goal (evolution) system, by setting the states properly.
   *
   * @param current_goal_state_init Initialization value for the current goal
   * state
   * @param G_init Initialization value for steady-state goal position vector
   * (G)
   * @return Success or failure
   */
  bool startQuaternionGoalSystem(
      const QuaternionDMPState& current_goal_state_init, const VectorN& G_init);

  /**
   * Returns the current transient/non-steady-state goal state vector.
   *
   * @return The current transient/non-steady-state goal state vector
   */
  QuaternionDMPState getCurrentQuaternionGoalState();

  /**
   * Sets new current (evolving) goal state.
   *
   * @param new_current_goal_state New current (evolving) goal state to be set
   * @return Success or failure
   */
  bool setCurrentQuaternionGoalState(
      const QuaternionDMPState& new_current_goal_state);

  /**
   * Runs one time-step ahead of the goal state system and updates the current
   * goal state vector based on the time difference dt from the previous
   * execution.
   *
   * @param dt Time difference between the current and the previous execution
   * @return Success or failure
   */
  bool updateCurrentGoalState(const double& dt);

  ~QuaternionGoalSystem();

 private:
  // hide some methods:
  using GoalSystem::getCurrentGoalState;
  using GoalSystem::setCurrentGoalState;
  using GoalSystem::start;
};
}  // namespace dmp
#endif
