/*
 * GoalSystem.h
 *
 * System that represents the current DMP goal state (including position,
 * velocity, acceleration, and time (if used)) and its evolution toward the
 * final (steady-state) goal position.
 *
 *  Created on: June 18, 2015
 *  Author: Giovanni Sutanto
 */

#ifndef GOAL_SYSTEM_H
#define GOAL_SYSTEM_H

#include "dmp/dmp_param/TauSystem.h"
#include "dmp/utility/DefinitionsDerived.h"

namespace dmp {

class GoalSystem {
 private:
  bool is_current_goal_state_obj_created_here;

 protected:
  uint dmp_num_dimensions;  // number of dimensions in the DMP, for example:
                            // Cartesian DMP has 3 dimensions, Quaternion DMP
                            // has 3 dimensions, etc.
  double alpha;

  // The following field(s) are written as RAW pointers for extensibility in
  // Object-Oriented Programming (OOP)
  DMPState* current_goal_state;  // current DMP goal state (position, velocity,
                                 // acceleration, and time (if used))

  VectorN G;  // final (steady-state) goal position

  bool is_started;

  // The following field(s) are written as RAW pointers because we just want to
  // point to other data structure(s) that might outlive the instance of this
  // class:
  TauSystem* tau_sys;  // tau system that computes the tau parameter for this
                       // goal system
  RealTimeAssertor* rt_assertor;

 public:
  GoalSystem();

  /**
   * @param dmp_num_dimensions_init Initialization of the number of dimensions
   * that will be used in the DMP
   * @param tau_system Tau system that computes the tau parameter for this
   * goal-changing system
   * @param alpha_init Spring parameter (default value = 25/2)
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  GoalSystem(uint dmp_num_dimensions_init, TauSystem* tau_system,
             RealTimeAssertor* real_time_assertor,
             double alpha_init = 25.0 / 2.0);

  /**
   * @param dmp_num_dimensions_init Initialization of the number of dimensions
   * that will be used in the DMP
   * @param current_goal_state_ptr Pointer to DMPState that will play the role
   * of current_goal_state
   * @param goal_num_dimensions_init Initialization of the number of dimensions
   * of the goal position
   * @param tau_system Tau system that computes the tau parameter for this
   * goal-changing system
   * @param alpha_init Spring parameter (default value = 25/2)
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  GoalSystem(uint dmp_num_dimensions_init, DMPState* current_goal_state_ptr,
             uint goal_num_dimensions_init, TauSystem* tau_system,
             RealTimeAssertor* real_time_assertor,
             double alpha_init = 25.0 / 2.0);

  /**
   * Checks whether this tau system is valid or not.
   *
   * @return Tau system is valid (true) or tau system is invalid (false)
   */
  virtual bool isValid();

  /**
   * Start the goal (evolution) system, by setting the states properly.
   *
   * @param current_goal_state_init Initialization value for the current goal
   * state
   * @param G_init Initialization value for steady-state goal position vector
   * (G)
   * @return Success or failure
   */
  virtual bool start(const DMPState& current_goal_state_init,
                     const VectorN& G_init);

  /**
   * Returns the number of dimensions in the DMP.
   *
   * @return Number of dimensions in the DMP
   */
  virtual uint getDMPNumDimensions();

  /**
   * Returns steady-state goal position vector (G).
   *
   * @return steady-state goal position vector (G)
   */
  virtual VectorN getSteadyStateGoalPosition();

  /**
   * Sets new steady-state goal position vector (G).
   *
   * @param new_G New steady-state goal position vector (G) to be set
   * @return Success or failure
   */
  virtual bool setSteadyStateGoalPosition(const VectorN& new_G);

  /**
   * Returns the current transient/non-steady-state goal state vector.
   *
   * @return The current transient/non-steady-state goal state vector
   */
  virtual DMPState getCurrentGoalState();

  /**
   * Sets new current (evolving) goal state.
   *
   * @param new_current_goal_state New current (evolving) goal state to be set
   * @return Success or failure
   */
  virtual bool setCurrentGoalState(const DMPState& new_current_goal_state);

  /**
   * Runs one time-step ahead of the goal state system and updates the current
   * goal state vector based on the time difference dt from the previous
   * execution.
   *
   * @param dt Time difference between the current and the previous execution
   * @return Success or failure
   */
  virtual bool updateCurrentGoalState(const double& dt);

  /**
   * Returns the tau system pointer.
   */
  virtual TauSystem* getTauSystemPointer();

  virtual ~GoalSystem();
};
}  // namespace dmp
#endif
