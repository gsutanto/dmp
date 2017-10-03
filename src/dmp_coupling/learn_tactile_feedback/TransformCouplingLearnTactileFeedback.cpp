#include "amd_clmc_dmp/dmp_coupling/learn_tactile_feedback/TransformCouplingLearnTactileFeedback.h"

namespace dmp
{

/**
 * NON-REAL-TIME!!!
 */
TransformCouplingLearnTactileFeedback::TransformCouplingLearnTactileFeedback():
    TransformCoupling(),
    start_index(0),
    ffnn_lwr_input_num_dimensions(0), ffnn_lwr_phase_kernel_model_size(0), ffnn_lwr_output_num_dimensions(0),
    ffnn_lwr_per_dims(NULL), ffnn_lwr_input_vector(NULL), ffnn_lwr_phase_kernel_modulation(NULL), ffnn_lwr_output_vector(NULL)
{}

/**
 * NON-REAL-TIME!!!\n
 *
 * @param dmp_num_dimensions_init Initialization of the number of dimensions that will be used in the DMP
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 */
TransformCouplingLearnTactileFeedback::TransformCouplingLearnTactileFeedback(uint dmp_num_dimensions_init,
                                                                             uint start_index_init,
                                                                             uint ffnn_lwr_input_num_dimensions_init,
                                                                             uint ffnn_lwr_phase_kernel_model_size_init,
                                                                             uint ffnn_lwr_output_num_dimensions_init,
                                                                             VectorNN_N* ffnn_lwr_input_vector_ptr,
                                                                             VectorNN_N* ffnn_lwr_phase_kernel_modulation_ptr,
                                                                             VectorNN_N* ffnn_lwr_output_vector_ptr,
                                                                             RealTimeAssertor* real_time_assertor):
    TransformCoupling(dmp_num_dimensions_init, real_time_assertor),
    start_index(start_index_init),
    ffnn_lwr_input_num_dimensions(ffnn_lwr_input_num_dimensions_init),
    ffnn_lwr_phase_kernel_model_size(ffnn_lwr_phase_kernel_model_size_init),
    ffnn_lwr_output_num_dimensions(ffnn_lwr_output_num_dimensions_init),
    ffnn_lwr_per_dims(NULL),    // this should be set manually later
    ffnn_lwr_input_vector(ffnn_lwr_input_vector_ptr), ffnn_lwr_phase_kernel_modulation(ffnn_lwr_phase_kernel_modulation_ptr),
    ffnn_lwr_output_vector(ffnn_lwr_output_vector_ptr)
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
                  (rt_assert(start_index < ffnn_lwr_output_num_dimensions))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(ffnn_lwr_input_num_dimensions >= 0)) &&
                  (rt_assert(ffnn_lwr_input_num_dimensions <= MAX_NN_NUM_NODES_PER_LAYER))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(ffnn_lwr_phase_kernel_model_size >= 0)) &&
                  (rt_assert(ffnn_lwr_phase_kernel_model_size <= MAX_MODEL_SIZE))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(ffnn_lwr_output_num_dimensions >= 0)) &&
                  (rt_assert(ffnn_lwr_output_num_dimensions > start_index + dmp_num_dimensions - 1))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(ffnn_lwr_per_dims != NULL)) &&
                  (rt_assert(ffnn_lwr_input_vector != NULL)) &&
                  (rt_assert(ffnn_lwr_phase_kernel_modulation != NULL)) &&
                  (rt_assert(ffnn_lwr_output_vector != NULL))) == false)
    {
        return false;
    }
    if (rt_assert(ffnn_lwr_per_dims->isValid()) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(ffnn_lwr_input_vector->rows() == ffnn_lwr_input_num_dimensions)) &&
                  (rt_assert(ffnn_lwr_phase_kernel_modulation->rows() == ffnn_lwr_phase_kernel_model_size)) &&
                  (rt_assert(ffnn_lwr_output_vector->rows() == ffnn_lwr_output_num_dimensions))) == false)
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

    ffnn_lwr_per_dims->computePrediction(*ffnn_lwr_input_vector, *ffnn_lwr_phase_kernel_modulation, *ffnn_lwr_output_vector,
                                         start_index, start_index+dmp_num_dimensions-1);

    ct_acc.block(0,0,dmp_num_dimensions,1)  = ffnn_lwr_output_vector->block(start_index,0,dmp_num_dimensions,1);

    ct_vel                                  = ZeroVectorN(dmp_num_dimensions);

    return true;
}

TransformCouplingLearnTactileFeedback::~TransformCouplingLearnTactileFeedback()
{}

}
