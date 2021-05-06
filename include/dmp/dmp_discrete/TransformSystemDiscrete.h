/*
 * TransformSystemDiscrete.h
 *
 *  Spring-damper based transformation system of a DMP for a discrete movement
 * (non-rhythmic)
 *
 *  See A.J. Ijspeert, J. Nakanishi, H. Hoffmann, P. Pastor, and S. Schaal;
 *      Dynamical movement primitives: Learning attractor models for motor
 * behaviors; Neural Comput. 25, 2 (February 2013), 328-373
 *
 *  Created on: Dec 12, 2014
 *  Author: Yevgen Chebotar and Giovanni Sutanto
 */

#ifndef TRANSFORM_SYSTEM_DISCRETE_H
#define TRANSFORM_SYSTEM_DISCRETE_H

#include "dmp/dmp_base/TransformationSystem.h"
#include "dmp/dmp_discrete/CanonicalSystemDiscrete.h"
#include "dmp/dmp_discrete/FuncApproximatorDiscrete.h"
#include "dmp/dmp_discrete/LoggedDMPDiscreteVariables.h"

namespace dmp {

class TransformSystemDiscrete : public TransformationSystem {
 protected:
  uint formulation_type;
  double alpha;
  double beta;
  std::vector<bool> is_using_scaling;  // only relevant for Schaal's DMP Model
  VectorN A_learn;  // amplitude = (G - y_0) value during learning

  FuncApproximatorDiscrete* func_approx_discrete;
  LoggedDMPDiscreteVariables* logged_dmp_discrete_variables;

 public:
  TransformSystemDiscrete();

  /**
   * @param dmp_num_dimensions_init Number of DMP dimensions (N) to use
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
   * @param start_dmpstate_discrete (Pointer to the) start state of the discrete
   * DMP's transformation system
   * @param current_dmpstate_discrete (Pointer to the) current state of the
   * discrete DMP's transformation system
   * @param current_velocity_dmpstate_discrete (Pointer to the) current velocity
   * state of the discrete DMP's transformation system
   * @param goal_system_discrete (Pointer to the) goal system for discrete DMP's
   * transformation system
   * @param transform_couplers All spatial couplings associated with the
   * transformation system (please note that this is a pointer to vector of
   * pointer to TransformCoupling data structure, for flexibility and
   * re-usability)
   * @param ts_alpha Spring parameter\n
   *        Best practice: set it to 25
   * @param ts_beta Damping parameter\n
   *        Best practice: set it to alpha / 4 for a critically damped system
   * @param ts_formulation_type Chosen formulation\n
   *        _SCHAAL_DMP_: original formulation by Schaal, AMAM 2003 (default
   * formulation_type)\n _HOFFMANN_DMP_: formulation by Hoffmann et al., ICRA
   * 2009
   */
  TransformSystemDiscrete(
      uint dmp_num_dimensions_init,
      CanonicalSystemDiscrete* canonical_system_discrete,
      FuncApproximatorDiscrete* func_approximator_discrete,
      LoggedDMPDiscreteVariables* logged_dmp_discrete_vars,
      RealTimeAssertor* real_time_assertor,
      std::vector<bool> is_using_scaling_init,
      DMPState* start_dmpstate_discrete = NULL,
      DMPState* current_dmpstate_discrete = NULL,
      DMPState* current_velocity_dmpstate_discrete = NULL,
      GoalSystem* goal_system_discrete = NULL,
      std::vector<TransformCoupling*>* transform_couplers = NULL,
      double ts_alpha = 25.0,
      double ts_beta = 25.0 / 4.0,
      uint ts_formulation_type = _SCHAAL_DMP_);

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
  bool start(const DMPState& start_state_init, const DMPState& goal_state_init);

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
  bool getNextState(
      double dt, DMPState& next_state, VectorN* forcing_term = NULL,
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
  bool getTargetForcingTerm(const DMPState& current_state_demo_local,
                            VectorN& f_target);

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
  bool getTargetCouplingTerm(const DMPState& current_state_demo_local,
                             VectorN& ct_acc_target);

  /**
   * Set the scaling usage in each DMP dimension.
   *
   * @param Scaling usage (boolean) vector, each component corresponds to each
   * DMP dimension
   */
  bool setScalingUsage(const std::vector<bool>& is_using_scaling_init);

  /**
   * Get the learning amplitude (A_learn) = G_learn - y_0_learn in each DMP
   * dimension.
   *
   * @return Current learning amplitude (A_learn)
   */
  VectorN getLearningAmplitude();

  /**
   * Set a new learning amplitude (A_learn) = G_learn - y_0_learn in each DMP
   * dimension. This will be called during learning from a demonstrated
   * trajectory.
   *
   * @param new_A_learn New learning amplitude (A_learn)
   * @return Success or failure
   */
  bool setLearningAmplitude(const VectorN& new_A_learn);

  /**
   * @return The chosen formulation for the discrete transformation system\n
   *         0: original formulation by Schaal, AMAM 2003 (default
   * formulation_type)\n 1: formulation by Hoffman et al., ICRA 2009
   */
  uint getFormulationType();

  ~TransformSystemDiscrete();
};

}  // namespace dmp
#endif
