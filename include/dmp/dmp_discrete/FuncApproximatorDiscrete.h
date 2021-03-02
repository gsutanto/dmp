/*
 * FuncApproximatorDiscrete.h
 *
 *  Non-linear function approximator based on weighted Gaussian basis functions
 * for discrete movements. Can be learned with the help of locally-weighted
 * regression (LWR).
 *
 *  See A.J. Ijspeert, J. Nakanishi, H. Hoffmann, P. Pastor, and S. Schaal;
 *      Dynamical movement primitives: Learning attractor models for motor
 * behaviors; Neural Comput. 25, 2 (February 2013), 328-373
 *
 *  Additional Note: If anyone is asking the reason why FuncApproximatorDiscrete
 * (FunctionApproximator) is being separated from LearningSystemDiscrete
 * (LearningSystem), please see the description (or top comments) in
 * FunctionApproximator.h.
 *
 *  Created on: Dec 12, 2014
 *  Author: Yevgen Chebotar and Giovanni Sutanto
 */

#ifndef FUNC_APPROXIMATOR_DISCRETE_H
#define FUNC_APPROXIMATOR_DISCRETE_H

#include "dmp/dmp_base/FunctionApproximator.h"
#include "dmp/dmp_discrete/CanonicalSystemDiscrete.h"
#include "dmp/dmp_state/DMPState.h"
#include "math.h"

namespace dmp {

class FuncApproximatorDiscrete : public FunctionApproximator {
 private:
  // The following field(s) are written as SMART pointers to reduce size
  // overhead in the stack, by allocating them in the heap:
  VectorMPtr centers;
  VectorMPtr bandwidths;
  VectorMPtr psi;  // psi vector is the same across all DMP dimensions, so we
                   // don't need a matrix here

  double sum_psi;  // the sum of all components of the psi vector

  /**
   * NEVER RUN IN REAL-TIME, ONLY RUN IN INIT ROUTINE
   * Initializes centers and bandwidths of the Gaussian basis functions.
   */
  void initBasisFunctions();

 public:
  FuncApproximatorDiscrete();

  /**
   * NEVER RUN IN REAL-TIME, ONLY RUN IN INIT ROUTINE
   *
   * @param dmp_num_dimensions_init Number of DMP dimensions (N) to use
   * @param num_basis_functions Number of basis functions (M) to use
   * @param canonical_system_discrete Discrete canonical system that drives this
   * discrete function approximator
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  FuncApproximatorDiscrete(uint dmp_num_dimensions_init,
                           uint num_basis_functions,
                           CanonicalSystemDiscrete* canonical_system_discrete,
                           RealTimeAssertor* real_time_assertor);

  /**
   * Checks whether this discrete function approximator is valid or not.
   *
   * @return Discrete function approximator is valid (true) or discrete function
   * approximator is invalid (false)
   */
  bool isValid();

  /**
   * Returns the value of the approximated function at current canonical
   * position and \n current canonical multiplier, for each DMP dimensions (as a
   * vector).
   *
   * @param result_f Computed forcing term vector at current canonical position
   * and \n current canonical multiplier, each vector component for each DMP
   * dimension (return variable)
   * @param basis_function_vector [optional] If also interested in the basis
   * function vector value, put its pointer here (return variable)
   * @param normalized_basis_func_vector_mult_phase_multiplier [optional] If
   * also interested in \n the <normalized basis function vector multiplied by
   * the canonical phase multiplier (phase position or phase velocity)> value,
   * \n put its pointer here (return variable)
   * @return Success or failure
   */
  bool getForcingTerm(
      VectorN& result_f, VectorM* basis_function_vector = NULL,
      VectorM* normalized_basis_func_vector_mult_phase_multiplier = NULL);

  /**
   * Returns the vector variables required for learning the weights of the basis
   * functions.
   *
   * @param current_psi Current psi vector associated with current phase
   * (canonical system multiplier) variable (return variable)
   * @param current_sum_psi Sum of current_psi vector (return variable)
   * @param current_xi Current phase (canonical system multiplier) variable
   * (return variable)
   * @return Success or failure
   */
  bool getFuncApproxLearningComponents(VectorM& current_psi,
                                       double& current_sum_psi,
                                       double& current_xi);

  bool getNormalizedBasisFunctionVectorMultipliedPhaseMultiplier(
      VectorM& normalized_basis_func_vector_mult_phase_multiplier);

  /**
   * Updates (and returns) the (unnormalized) vector of basis function
   * evaluations at the given position (x).
   *
   * @param x Position to be evaluated at (usually this is the canonical state
   * position)
   * @param result_vector Vector that will be filled with the result (return
   * variable).\n If NULL then nothing gets returned (only updating basis
   * functions vector (psi) and its sum (sum_psi)).
   * @return Success or failure
   */
  bool getBasisFunctionVector(double x, VectorM& result_vector);

  /**
   * Returns the value of a specified basis function at the given position (x).
   *
   * @param x Position to be evaluated at (usually this is the canonical state
   * position)
   * @param center Center of the basis function curve
   * @param bandwidth Bandwidth of the basis function curve
   * @return The value of a specified basis function at the given position (x)
   */
  double getBasisFunctionValue(double x, double center, double bandwidth);

  /**
   * Get the current weights of the basis functions.
   *
   * @param weights_buffer Weights vector buffer, to which the current weights
   * will be copied onto (to be returned)
   * @return Success or failure
   */
  bool getWeights(MatrixNxM& weights_buffer);

  /**
   * Set the new weights of the basis functions.
   *
   * @param new_weights New weights to be set
   * @return Success or failure
   */
  bool setWeights(const MatrixNxM& new_weights);

  ~FuncApproximatorDiscrete();
};

}  // namespace dmp
#endif
