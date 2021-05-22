/*
 * TransformSystemQuaternion.h
 *
 *  Spring-damper based transformation system of a Quaternion DMP for a discrete
 * (non-rhythmic) orientation movement
 *
 *  See G. Sutanto, Z. Su, G. Loeb, F. Meier, and S. Schaal;
 *      Learning Associative Skill Memory-guided Feedback Term for Reactive
 * Planning; in preparation, June 2017
 *
 *  Created on: June 21, 2017
 *  Author: Giovanni Sutanto
 */

#ifndef TRANSFORM_SYSTEM_QUATERNION_H
#define TRANSFORM_SYSTEM_QUATERNION_H

#include "dmp/dmp_discrete/TransformSystemDiscrete.h"
#include "dmp/dmp_goal_system/QuaternionGoalSystem.h"
#include "dmp/utility/utility_quaternion.h"

namespace dmp {

class TransformSystemQuaternion : public TransformSystemDiscrete {
 protected:
  QuaternionDMPState start_quat_state;
  QuaternionDMPState current_quat_state;
  DMPState current_angular_velocity_state;

  QuaternionGoalSystem quat_goal_sys;

 public:
  TransformSystemQuaternion();

  /**
   * Original formulation by Schaal, AMAM 2003 (default formulation type) is
   * selected.
   *
   * @param canonical_system_discrete Discrete canonical system that drives this
   * discrete transformation system
   * @param func_approximator_discrete Discrete function approximator that
   * produces forcing term for this discrete transformation system
   * @param logged_dmp_discrete_vars (Pointer to the) buffer for logged discrete
   * DMP variables
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   * @param is_using_scaling_init Is transformation system using scaling
   * (true/false)?
   * @param transform_couplers All spatial couplings associated with the
   * transformation system \n (please note that this is a pointer to vector of
   * pointer to TransformCoupling data structure, for flexibility and
   * re-usability)
   * @param ts_alpha Spring parameter\n
   *        Best practice: set it to 25
   * @param ts_beta Damping parameter\n
   *        Best practice: set it to alpha / 4 for a critically damped system
   */
  TransformSystemQuaternion(
      CanonicalSystemDiscrete* canonical_system_discrete,
      std::shared_ptr<FuncApproximatorDiscrete> func_approximator_discrete,
      LoggedDMPDiscreteVariables* logged_dmp_discrete_vars,
      RealTimeAssertor* real_time_assertor,
      std::vector<bool> is_using_scaling_init,
      std::vector<TransformCoupling*>* transform_couplers = NULL,
      double ts_alpha = 25.0, double ts_beta = (25.0 / 4.0));

  /**
   * Checks whether this transformation system is valid or not.
   *
   * @return Transformation system is valid (true) or transformation system is
   * invalid (false)
   */
  bool isValid();

  /**
   * Start the transformation system, by setting the states properly.
   *
   * @param start_state_init Initialization of start state
   * @param goal_state_init Initialization of goal state
   * @return Success or failure
   */
  bool startTransformSystemQuaternion(
      const QuaternionDMPState& start_state_init,
      const QuaternionDMPState& goal_state_init);

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
  bool getNextQuaternionState(
      double dt, QuaternionDMPState& next_state, VectorN* forcing_term = NULL,
      VectorN* coupling_term_acc = NULL, VectorN* coupling_term_vel = NULL,
      VectorM* basis_functions_out = NULL,
      VectorM* normalized_basis_func_vector_mult_phase_multiplier = NULL);

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
  bool getTargetQuaternionForcingTerm(
      const QuaternionDMPState& current_state_demo_local, VectorN& f_target);

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
  bool getTargetQuaternionCouplingTerm(
      const QuaternionDMPState& current_state_demo_local,
      VectorN& ct_acc_target);

  /**
   * Returns transformation system's start state.
   *
   * @return Transformation system's start state (DMPState)
   */
  QuaternionDMPState getQuaternionStartState();

  /**
   * Sets new start state.
   *
   * @param new_start_state New start state to be set
   * @return Success or failure
   */
  bool setQuaternionStartState(const QuaternionDMPState& new_start_state);

  /**
   * Returns current transformation system state.
   *
   * @return Current transformation system state (DMPState)
   */
  QuaternionDMPState getCurrentQuaternionState();

  /**
   * Sets new current state.
   *
   * @param new_current_state New current state to be set
   * @return Success or failure
   */
  bool setCurrentQuaternionState(const QuaternionDMPState& new_current_state);

  /**
   * Update current velocity state from current (position) state
   *
   * @return Success or failure
   */
  bool updateCurrentVelocityStateFromCurrentState();

  /**
   * Returns the current transient/non-steady-state goal state vector.
   *
   * @return The current transient/non-steady-state goal state vector
   */
  QuaternionDMPState getCurrentQuaternionGoalState();

  /**
   * Sets new current goal state.
   *
   * @param new_current_goal_state New current goal state to be set
   * @return Success or failure
   */
  bool setCurrentQuaternionGoalState(
      const QuaternionDMPState& new_current_goal_state);

  ~TransformSystemQuaternion();

 private:
  // hide some methods:
  using TransformSystemDiscrete::getCurrentGoalState;
  using TransformSystemDiscrete::getCurrentState;
  using TransformSystemDiscrete::getNextState;
  using TransformSystemDiscrete::getStartState;
  using TransformSystemDiscrete::getTargetCouplingTerm;
  using TransformSystemDiscrete::getTargetForcingTerm;
  using TransformSystemDiscrete::setCurrentGoalState;
  using TransformSystemDiscrete::setCurrentState;
  using TransformSystemDiscrete::setStartState;
  using TransformSystemDiscrete::start;
};

}  // namespace dmp
#endif
