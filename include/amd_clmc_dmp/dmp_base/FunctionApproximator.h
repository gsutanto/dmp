/*
 * FunctionApproximator.h
 *
 *  Abstract class for a non-linear function approximator (forcing function)
 *  that can approximate a given trajectory. There are 2 reasons why this (FunctionApproximator)
 *  is being separated from the LearningSystem, as follows:
 *  (1) For Modularity: FunctionApproximator here is solely/ONLY dependent on canonical state variable(s)
 *      (of CanonicalSystem), and nothing else. Hence it is easily modifiable
 *      without affecting other systems too much.
 *  (2) For Real-Time-Safety: most functions in FunctionApproximator are required to be real-time-safe,
 *      while most functions in LearningSystem are NOT real-time-safe. Hence it is better to separate
 *      these two.
 *
 *  Created on: Dec 12, 2014
 *  Author: Yevgen Chebotar and Giovanni Sutanto
 */

#ifndef FUNCTION_APPROXIMATOR_H
#define FUNCTION_APPROXIMATOR_H

#include <vector>

#include "amd_clmc_dmp/utility/utility.h"
#include "amd_clmc_dmp/utility/RealTimeAssertor.h"
#include "amd_clmc_dmp/dmp_state/DMPState.h"
#include "amd_clmc_dmp/dmp_base/CanonicalSystem.h"

namespace dmp
{

class FunctionApproximator
{

protected:

    uint                dmp_num_dimensions;
    uint                model_size;

    // The following field(s) are written as RAW pointers because we just want to point
    // to other data structure(s) that might outlive the instance of this class:
    CanonicalSystem*    canonical_sys;
    RealTimeAssertor*   rt_assertor;

    // The following field(s) are written as SMART pointers to reduce size overhead
    // in the stack, by allocating them in the heap:
    MatrixNxMPtr        weights;

public:

    FunctionApproximator();

    /**
     * @param dmp_num_dimensions_init Number of DMP dimensions (N) to use
     * @param model_size_init Size of the model (M: number of basis functions or others) used to represent the function
     * @param canonical_system Canonical system that drives this function approximator
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    FunctionApproximator(uint dmp_num_dimensions_init, uint model_size_init,
                         CanonicalSystem* canonical_system, RealTimeAssertor* real_time_assertor);

    /**
     * Checks whether this function approximator is valid or not.
     *
     * @return Function approximator is valid (true) or function approximator is invalid (false)
     */
    virtual bool isValid();

    /**
     * Returns the number of dimensions in the DMP.
     *
     * @return Number of dimensions in the DMP
     */
    virtual uint getDMPNumDimensions();

    /**
     * Returns model size used to represent the function.
     *
     * @return Model size used to represent the function
     */
    virtual uint getModelSize();

    /**
     * Returns the value of the approximated function at current canonical position and current canonical multiplier.
     *
     * @param result_f Computed forcing term at current canonical position and current canonical multiplier (return variable)
     * @param basis_function_vector [optional] If also interested in the basis function vector value, put its pointer here (return variable)
     * @param normalized_basis_func_vector_mult_phase_multiplier [optional] If also interested in \n
     *              the <normalized basis function vector multiplied by the canonical phase multiplier (phase position or phase velocity)> value, \n
     *              put its pointer here (return variable)
     * @return Success or failure
     */
    virtual bool getForcingTerm(VectorN& result_f,
                                VectorM* basis_function_vector=NULL,
                                VectorM* normalized_basis_func_vector_mult_phase_multiplier=NULL) = 0;

    /**
     * Get the current weights of the basis functions.
     *
     * @param weights_buffer Weights vector buffer, to which the current weights will be copied onto (to be returned)
     * @return Success or failure
     */
    virtual bool getWeights(MatrixNxM& weights_buffer) = 0;

    /**
     * Set the new weights of the basis functions.
     *
     * @param new_weights New weights to be set
     * @return Success or failure
     */
    virtual bool setWeights(const MatrixNxM& new_weights) = 0;

    /**
     * Returns the canonical system pointer.
     */
    virtual CanonicalSystem* getCanonicalSystemPointer();

    ~FunctionApproximator();

private:

};
}
#endif
