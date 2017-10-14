#include "amd_clmc_dmp/dmp_coupling/learn_tactile_feedback/TransformCouplingLearnTactileFeedback.h"

namespace dmp
{

/**
 * NON-REAL-TIME!!!
 */
TransformCouplingLearnTactileFeedback::TransformCouplingLearnTactileFeedback():
    TransformCoupling(),
    start_index(0),
    pmnn_input_num_dimensions(0), pmnn_phase_kernel_model_size(0), pmnn_output_num_dimensions(0),
    pmnn(NULL), pmnn_input_vector(NULL), pmnn_phase_kernel_modulation(NULL), pmnn_output_vector(NULL)
{}

/**
 * NON-REAL-TIME!!!\n
 *
 * @param dmp_num_dimensions_init Initialization of the number of dimensions that will be used in the DMP
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 */
TransformCouplingLearnTactileFeedback::TransformCouplingLearnTactileFeedback(uint dmp_num_dimensions_init,
                                                                             uint start_index_init,
                                                                             uint pmnn_input_num_dimensions_init,
                                                                             uint pmnn_phase_kernel_model_size_init,
                                                                             uint pmnn_output_num_dimensions_init,
                                                                             VectorNN_N* pmnn_input_vector_ptr,
                                                                             VectorNN_N* pmnn_phase_kernel_modulation_ptr,
                                                                             VectorNN_N* pmnn_output_vector_ptr,
                                                                             RealTimeAssertor* real_time_assertor):
    TransformCoupling(dmp_num_dimensions_init, real_time_assertor),
    start_index(start_index_init),
    pmnn_input_num_dimensions(pmnn_input_num_dimensions_init),
    pmnn_phase_kernel_model_size(pmnn_phase_kernel_model_size_init),
    pmnn_output_num_dimensions(pmnn_output_num_dimensions_init),
    pmnn(NULL),    // this should be set manually later
    pmnn_input_vector(pmnn_input_vector_ptr), pmnn_phase_kernel_modulation(pmnn_phase_kernel_modulation_ptr),
    pmnn_output_vector(pmnn_output_vector_ptr)
{}

/**
 * Checks whether this transformation coupling term for learning tactile feedback by demonstration is valid or not.
 *
 * @return Valid (true) or invalid (false)
 */
bool TransformCouplingLearnTactileFeedback::isValid()
{
    if (rt_assertor == NULL)
    {
        return false;
    }
    if (rt_assert(TransformCoupling::isValid()) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(start_index >= 0)) &&
                  (rt_assert(start_index < pmnn_output_num_dimensions))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(pmnn_input_num_dimensions >= 0)) &&
                  (rt_assert(pmnn_input_num_dimensions <= MAX_NN_NUM_NODES_PER_LAYER))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(pmnn_phase_kernel_model_size >= 0)) &&
                  (rt_assert(pmnn_phase_kernel_model_size <= MAX_MODEL_SIZE))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(pmnn_output_num_dimensions >= 0)) &&
                  (rt_assert(pmnn_output_num_dimensions > start_index + dmp_num_dimensions - 1))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(pmnn != NULL)) &&
                  (rt_assert(pmnn_input_vector != NULL)) &&
                  (rt_assert(pmnn_phase_kernel_modulation != NULL)) &&
                  (rt_assert(pmnn_output_vector != NULL))) == false)
    {
        return false;
    }
    if (rt_assert(pmnn->isValid()) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(pmnn_input_vector->rows() == pmnn_input_num_dimensions)) &&
                  (rt_assert(pmnn_phase_kernel_modulation->rows() == pmnn_phase_kernel_model_size)) &&
                  (rt_assert(pmnn_output_vector->rows() == pmnn_output_num_dimensions))) == false)
    {
        return false;
    }
    return true;
}

/**
 * Returns both acceleration-level coupling value (C_t_{Acc}) and
 * velocity-level coupling value (C_t_{Vel}) that will be used in the transformation system
 * to adjust motion of a DMP.
 *
 * @param ct_acc Acceleration-level coupling value (C_t_{Acc}) that will be used to adjust the acceleration of DMP transformation system
 * @param ct_vel Velocity-level coupling value (C_t_{Vel}) that will be used to adjust the velocity of DMP transformation system
 * @return Success or failure
 */
bool TransformCouplingLearnTactileFeedback::getValue(VectorN& ct_acc, VectorN& ct_vel)
{
    // pre-conditions checking
    if (rt_assert(TransformCouplingLearnTactileFeedback::isValid()) == false)
    {
        return false;
    }
    // input checking
    if (rt_assert((ct_acc.rows() == dmp_num_dimensions) &&
                  (ct_vel.rows() == dmp_num_dimensions)) == false)
    {
        return false;
    }

    pmnn->computePrediction(*pmnn_input_vector, *pmnn_phase_kernel_modulation, *pmnn_output_vector,
                            start_index, start_index+dmp_num_dimensions-1);

    ct_acc.block(0,0,dmp_num_dimensions,1)  = pmnn_output_vector->block(start_index,0,dmp_num_dimensions,1);

    ct_vel                                  = ZeroVectorN(dmp_num_dimensions);

    return true;
}

TransformCouplingLearnTactileFeedback::~TransformCouplingLearnTactileFeedback()
{}

}
