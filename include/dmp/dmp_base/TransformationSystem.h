/*
 * CanonicalSystem.h
 *
 *  Abstract class that contains a transformation system of the DMP.
 *
 *  Created on: Dec 12, 2014
 *  Author: Yevgen Chebotar and Giovanni Sutanto
 */

#ifndef TRANSFORMATION_SYSTEM_H
#define TRANSFORMATION_SYSTEM_H

#include <stddef.h>
#include <stdlib.h>

#include <limits>
#include <memory>
#include <vector>

#include "dmp/dmp_base/CanonicalSystem.h"
#include "dmp/dmp_base/FunctionApproximator.h"
#include "dmp/dmp_base/LoggedDMPVariables.h"
#include "dmp/dmp_coupling/base/TransformCoupling.h"
#include "dmp/dmp_goal_system/GoalSystem.h"
#include "dmp/dmp_param/TauSystem.h"
#include "dmp/dmp_state/DMPState.h"
#include "dmp/utility/RealTimeAssertor.h"
#include "dmp/utility/utility.h"

#define MIN_FABS_AMPLITUDE 1.e-3  // minimum of fabs(amplitude)
#define MIN_K 0.01  // minimum of K (in Hoffman et al.'s formulation)

namespace dmp {

class TransformationSystem {
 private:
  bool is_start_state_obj_created_here;
  bool is_current_state_obj_created_here;
  bool is_current_velocity_state_obj_created_here;
  bool is_goal_sys_obj_created_here;

 protected:
  uint dmp_num_dimensions;

  bool is_started;

  std::vector<bool> is_using_coupling_term_at_dimension;

  TauSystem* tau_sys;
  FunctionApproximator* func_approx;  // No ownership.
  LoggedDMPVariables* logged_dmp_variables;
  std::vector<TransformCoupling*>* transform_coupling;

  DMPState* start_state;
  DMPState* current_state;
  DMPState* current_velocity_state;
  GoalSystem* goal_sys;  // including the goal state

  RealTimeAssertor* rt_assertor;

 public:
  TransformationSystem();

  /**
   * @param dmp_num_dimensions_init Number of DMP dimensions (N) to use
   * @param tau_system Tau system that computes the tau parameter for this
   * transformation system
   * @param canonical_system Canonical system that drives this transformation
   * system
   * @param function_approximator Function approximator that produces forcing
   * term for this transformation system
   * @param logged_dmp_vars (Pointer to the) buffer for logged DMP variables
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   * @param start_dmpstate (Pointer to the) start state of the DMP's
   * transformation system
   * @param current_dmpstate (Pointer to the) current state of the DMP's
   * transformation system
   * @param current_velocity_dmpstate (Pointer to the) current velocity state of
   * the DMP's transformation system
   * @param goal_system (Pointer to the) goal system for DMP's transformation
   * system
   * @param transform_couplers All spatial couplings associated with the
   * transformation system (please note that this is a pointer to vector of
   * pointer to TransformCoupling data structure, for flexibility and
   * re-usability)
   */
  TransformationSystem(
      uint dmp_num_dimensions_init, TauSystem* tau_system,
      CanonicalSystem* canonical_system,
      FunctionApproximator* function_approximator,
      LoggedDMPVariables* logged_dmp_vars, RealTimeAssertor* real_time_assertor,
      DMPState* start_dmpstate = NULL, DMPState* current_dmpstate = NULL,
      DMPState* current_velocity_dmpstate = NULL,
      GoalSystem* goal_system = NULL,
      std::vector<TransformCoupling*>* transform_couplers = NULL);

  /**
   * Checks whether this transformation system is valid or not.
   *
   * @return Transformation system is valid (true) or transformation system is
   * invalid (false)
   */
  virtual bool isValid();

  /**
   * Start the transformation system, by setting the states properly.
   *
   * @param start_state_init Initialization of start state
   * @param goal_state_init Initialization of goal state
   * @return Success or failure
   */
  virtual bool start(const DMPState& start_state_init,
                     const DMPState& goal_state_init) = 0;

  /**
   * Returns the next state (e.g. position, velocity, and acceleration) of the
   * DMP. [IMPORTANT]: Before calling this function, the TransformationSystem's
   * start_state, current_state, and goal_sys need to be initialized first using
   * TransformationSystem::start() function, using the desired start and goal
   * states.
   *
   * @param dt Time difference between current and previous steps
   * @param next_state Current DMP state (that will be updated with the computed
   * next state)
   * @param forcing_term (optional) If you also want to know the forcing term
   * value of the next state, \n please put the pointer here
   * @param coupling_term_acc (optional) If you also want to know the
   * acceleration-level \n coupling term value of the next state, please put the
   * pointer here
   * @param coupling_term_vel (optional) If you also want to know the
   * velocity-level \n coupling term value of the next state, please put the
   * pointer here
   * @param basis_functions_out (optional) (Pointer to) vector of basis
   * functions constructing the forcing term
   * @param normalized_basis_func_vector_mult_phase_multiplier (optional) If
   * also interested in \n the <normalized basis function vector multiplied by
   * the canonical phase multiplier (phase position or phase velocity)> value,
   * \n put its pointer here (return variable)
   * @return Success or failure
   */
  virtual bool getNextState(
      double dt, DMPState& next_state, VectorN* forcing_term = NULL,
      VectorN* coupling_term_acc = NULL, VectorN* coupling_term_vel = NULL,
      VectorM* basis_functions_out = NULL,
      VectorM* normalized_basis_func_vector_mult_phase_multiplier = NULL) = 0;

  /**
   * Based on the start state, current state, evolving goal state and
   * steady-state goal position, compute the target value of the forcing term.
   * Needed for learning the forcing function based on a given sequence of DMP
   * states (trajectory). [IMPORTANT]: Before calling this function, the
   * TransformationSystem's start_state, current_state, and goal_sys need to be
   * initialized first using TransformationSystem::start() function, using the
   * demonstrated trajectory.
   *
   * @param current_state_demo_local Demonstrated current DMP state in the local
   * coordinate system \n (if there are coordinate transformations)
   * @param f_target Target value of the forcing term at the demonstrated
   * current state,\n given the demonstrated start state, evolving goal state,
   * and \n demonstrated steady-state goal position \n (to be computed and
   * returned by this function)
   * @return Success or failure
   */
  virtual bool getTargetForcingTerm(const DMPState& current_state_demo_local,
                                    VectorN& f_target) = 0;

  /**
   * Based on the start state, current state, evolving goal state and
   * steady-state goal position, compute the target value of the coupling term.
   * Needed for learning the coupling term function based on a given sequence of
   * DMP states (trajectory). [IMPORTANT]: Before calling this function, the
   * TransformationSystem's start_state, current_state, and goal_sys need to be
   * initialized first using TransformationSystem::start() function, and the
   * forcing term weights need to have already been learned beforehand, using
   * the demonstrated trajectory.
   *
   * @param current_state_demo_local Demonstrated current DMP state in the local
   * coordinate system \n (if there are coordinate transformations)
   * @param ct_acc_target Target value of the acceleration-level coupling term
   * at the demonstrated current state,\n given the demonstrated start state,
   * evolving goal state, \n demonstrated steady-state goal position, and the
   * learned forcing term weights \n (to be computed and returned by this
   * function)
   * @return Success or failure
   */
  virtual bool getTargetCouplingTerm(const DMPState& current_state_demo_local,
                                     VectorN& ct_acc_target) = 0;

  /**
   * If there are coupling terms on the transformation system,
   * this function will iterate over each coupling term, and returns the
   * accumulated coupling term. Otherwise it will return a zero VectorN.
   *
   * @param accumulated_ct_acc The accumulated acceleration-level transformation
   * system coupling term value (to be returned)
   * @param accumulated_ct_vel The accumulated velocity-level transformation
   * system coupling term value (to be returned)
   * @return Success or failure
   */
  virtual bool getCouplingTerm(VectorN& accumulated_ct_acc,
                               VectorN& accumulated_ct_vel);

  /**
   * Returns the number of dimensions in the DMP.
   *
   * @return Number of dimensions in the DMP
   */
  virtual uint getDMPNumDimensions();

  /**
   * Returns model size used to represent the forcing term function.
   *
   * @return Model size used to represent the forcing term function
   */
  virtual uint getModelSize();

  /**
   * Returns transformation system's start state.
   *
   * @return Transformation system's start state (DMPState)
   */
  virtual DMPState getStartState();

  /**
   * Sets new start state.
   *
   * @param new_start_state New start state to be set
   * @return Success or failure
   */
  virtual bool setStartState(const DMPState& new_start_state);

  /**
   * Returns current transformation system state.
   *
   * @return Current transformation system state (DMPState)
   */
  virtual DMPState getCurrentState();

  /**
   * Sets new current state.
   *
   * @param new_current_state New current state to be set
   * @return Success or failure
   */
  virtual bool setCurrentState(const DMPState& new_current_state);

  /**
   * Update current velocity state from current (position) state
   *
   * @return Success or failure
   */
  virtual bool updateCurrentVelocityStateFromCurrentState();

  /**
   * Returns the current transient/non-steady-state goal state vector.
   *
   * @return The current transient/non-steady-state goal state vector
   */
  virtual DMPState getCurrentGoalState();

  /**
   * Sets new current goal state.
   *
   * @param new_current_goal_state New current goal state to be set
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
   * Returns steady-state goal position vector (G).
   *
   * @return Steady-state goal position vector (G)
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
   * Returns the tau system pointer.
   */
  virtual TauSystem* getTauSystemPointer();

  /**
   * Returns the canonical system pointer.
   */
  virtual CanonicalSystem* getCanonicalSystemPointer();

  /**
   * Returns the function approximator pointer.
   */
  virtual FunctionApproximator* getFuncApproxPointer();

  virtual bool setCouplingTermUsagePerDimensions(
      const std::vector<bool>& is_using_coupling_term_at_dimension_init);

  virtual ~TransformationSystem();
};
}  // namespace dmp
#endif
