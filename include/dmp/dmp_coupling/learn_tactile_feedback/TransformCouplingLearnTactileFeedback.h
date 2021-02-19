/*
 * TransformCouplingLearnTactileFeedback.h
 *
 *  Class defining coupling term for tactile feedback of the transformation system of a DMP.
 *
 *  Created on: Jun 27, 2017
 *  Author: Giovanni Sutanto
 */

#ifndef TRANSFORM_COUPLING_LEARN_TACTILE_FEEDBACK_H
#define TRANSFORM_COUPLING_LEARN_TACTILE_FEEDBACK_H

#include "dmp/dmp_coupling/base/TransformCoupling.h"
#include "dmp/neural_nets/PMNN.h"

namespace dmp
{

class TransformCouplingLearnTactileFeedback : public TransformCoupling
{

protected:

    uint                            start_index;                    // >=0, i.e. the queried coupling term will be
                                                                    // the values from pmnn_output_vector[start_index]
                                                                    // to pmnn_output_vector[start_index+dmp_num_dimensions-1], inclusive

    uint                            pmnn_input_num_dimensions;      // size of input vector
    uint                            pmnn_phase_kernel_model_size;   // model size of the phase kernel modulator
    uint                            pmnn_output_num_dimensions;     // total size of the usable output vector this MUST be bigger than (start_index + dmp_num_dimensions - 1)

public:

    PMNN*                           pmnn;                           // (pointer to) the prediction model; this pointer might be shared by other TransformCouplingLearnTactileFeedback object

    VectorNN_N*                     pmnn_input_vector;              // (pointer to) the input vector; this pointer might be shared by other TransformCouplingLearnTactileFeedback object
    VectorNN_N*                     pmnn_phase_kernel_modulation;   // (pointer to) the phase kernel modulator, modulating the final hidden layer of the pmnn model; this pointer might be shared by other TransformCouplingLearnTactileFeedback object
    VectorNN_N*                     pmnn_output_vector;             // (pointer to) the output vector

    TransformCouplingLearnTactileFeedback();

    /**
     * NON-REAL-TIME!!!\n
     *
     * @param dmp_num_dimensions_init Initialization of the number of dimensions that will be used in the DMP
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    TransformCouplingLearnTactileFeedback(uint dmp_num_dimensions_init,
                                          uint start_index_init,
                                          uint pmnn_input_num_dimensions_init,
                                          uint pmnn_phase_kernel_model_size_init,
                                          uint pmnn_output_num_dimensions_init,
                                          VectorNN_N* pmnn_input_vector_ptr,
                                          VectorNN_N* pmnn_phase_kernel_modulation_ptr,
                                          VectorNN_N* pmnn_output_vector_ptr,
                                          RealTimeAssertor* real_time_assertor);

    bool isValid();

    /**
     * According to the paper:
     *      A. Gams, B. Nemec, A.J. Ijspeert, and A. Ude;
     *      Coupling Movement Primitives: Interaction with the Environment and Bimanual Tasks;
     *      IEEE Transactions on Robotics 30, 4 (August 2014), 816-830
     *      (specifically on page 818-819)
     * DMP performs better when using both acceleration-level and velocity-level coupling term,
     * as compared to when just using either acceleration-level-only coupling term,
     * or velocity-level-only coupling term. Usually the velocity-level coupling term is related
     * to the acceleration-level coupling term by differentiation (with respect to time).
     *
     * @param ct_acc Acceleration-level coupling value (C_t_{Acc}) that will be used to adjust the acceleration of DMP transformation system
     * @param ct_vel Velocity-level coupling value (C_t_{Vel}) that will be used to adjust the velocity of DMP transformation system
     * @return Success or failure
     */
    bool getValue(VectorN& ct_acc, VectorN& ct_vel);

    ~TransformCouplingLearnTactileFeedback();

};

}
#endif
