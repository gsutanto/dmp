#include "dmp/dmp_base/FunctionApproximator.h"

namespace dmp
{

    FunctionApproximator::FunctionApproximator():
        dmp_num_dimensions(0), model_size(0), canonical_sys(NULL),
        weights(new MatrixNxM(MAX_DMP_NUM_DIMENSIONS, MAX_MODEL_SIZE)), rt_assertor(NULL)
    {}

    /**
     * @param dmp_num_dimensions_init Number of DMP dimensions (N) to use
     * @param model_size_init Size of the model (M: number of basis functions or others) used to represent the function
     * @param canonical_system Canonical system that drives this function approximator
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    FunctionApproximator::FunctionApproximator(uint dmp_num_dimensions_init, uint model_size_init,
                                               CanonicalSystem* canonical_system,
                                               RealTimeAssertor* real_time_assertor):
        dmp_num_dimensions(dmp_num_dimensions_init),
        model_size(model_size_init),
        canonical_sys(canonical_system),
        weights(new MatrixNxM(dmp_num_dimensions, model_size)),
        rt_assertor(real_time_assertor)
    {}

    /**
     * Checks whether this function approximator is valid or not.
     *
     * @return Function approximator is valid (true) or function approximator is invalid (false)
     */
    bool FunctionApproximator::isValid()
    {
        if (rt_assert((rt_assert(dmp_num_dimensions > 0)) &&
                      (rt_assert(dmp_num_dimensions <= MAX_DMP_NUM_DIMENSIONS))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(model_size > 0)) && (rt_assert(model_size <= MAX_MODEL_SIZE))) == false)
        {
            return false;
        }
        if (rt_assert(rt_assert(canonical_sys   != NULL) &&
                      rt_assert(weights         != NULL)) == false)
        {
            return false;
        }
        if (rt_assert(canonical_sys->isValid()) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(weights->rows() == dmp_num_dimensions)) &&
                      (rt_assert(weights->cols() == model_size))) == false)
        {
            return false;
        }
        return true;
    }

    /**
     * Returns the number of dimensions in the DMP.
     *
     * @return Number of dimensions in the DMP
     */
    uint FunctionApproximator::getDMPNumDimensions()
    {
        return dmp_num_dimensions;
    }

    /**
     * Returns model size used to represent the function.
     *
     * @return Model size used to represent the function
     */
    uint FunctionApproximator::getModelSize()
    {
        return model_size;
    }

    /**
     * Returns the canonical system pointer.
     */
    CanonicalSystem* FunctionApproximator::getCanonicalSystemPointer()
    {
        return	canonical_sys;
    }

    FunctionApproximator::~FunctionApproximator()
    {}

}
