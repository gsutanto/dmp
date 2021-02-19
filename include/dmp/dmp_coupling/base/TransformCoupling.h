/*
 * TransformCoupling.h
 *
 *  Abstract class defining coupling term for a transformation system of a DMP.
 *
 *  Created on: Jul 9, 2015
 *  Author: Giovanni Sutanto
 */

#ifndef TRANSFORM_COUPLING_H
#define TRANSFORM_COUPLING_H

#include "dmp/utility/DefinitionsBase.h"
#include "Coupling.h"

namespace dmp
{

class TransformCoupling : public Coupling
{

protected:

    uint                dmp_num_dimensions;

public:

    TransformCoupling():
        Coupling(),
        dmp_num_dimensions(0)
    {}

    /**
     * @param dmp_num_dimensions_init Initialization of the number of dimensions that will be used in the DMP
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    TransformCoupling(uint dmp_num_dimensions_init, RealTimeAssertor* real_time_assertor):
        Coupling(real_time_assertor),
        dmp_num_dimensions(dmp_num_dimensions_init)
    {}

    /**
     * Checks whether this transformation coupling term is valid or not.
     *
     * @return Valid (true) or invalid (false)
     */
    bool isValid()
    {
        if (rt_assertor != NULL)
        {
            if (rt_assert((rt_assert(dmp_num_dimensions >= 0)) &&
                          (rt_assert(dmp_num_dimensions <= MAX_DMP_NUM_DIMENSIONS))) == false)
            {
                return false;
            }
        }
        return true;
    }

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
    virtual bool getValue(VectorN& ct_acc, VectorN& ct_vel) = 0;

    virtual ~TransformCoupling() {}

};
}
#endif
