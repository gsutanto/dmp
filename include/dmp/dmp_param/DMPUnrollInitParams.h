/*
 * DMPUnrollInitParams.h
 *
 *  Class defining initialization parameters of DMP's unrolling and its methods.
 *
 *  Created on: Jul 23, 2016
 *  Author: Giovanni Sutanto
 */

#ifndef DMP_UNROLL_INIT_PARAMS_H
#define DMP_UNROLL_INIT_PARAMS_H

#include <stdlib.h>
#include <stddef.h>
#include <vector>
#include <memory>

#include "amd_clmc_dmp/utility/RealTimeAssertor.h"
#include "amd_clmc_dmp/utility/DefinitionsDerived.h"

namespace dmp
{

class DMPUnrollInitParams
{

    friend class        DMP;
    friend class        DMPDiscrete;
    friend class        CartesianCoordDMP;
    friend class        TransformCouplingLearnObsAvoid;

private:

    RealTimeAssertor*   rt_assertor;

public:

    DMPStates           critical_states;
    double              tau;

    /**
     * NON-REAL-TIME!!!
     */
    DMPUnrollInitParams();

    /**
     * NON-REAL-TIME!!!
     */
    DMPUnrollInitParams(const double& tau_init,
                        const DMPState& start_state,
                        const DMPState& goal_state,
                        RealTimeAssertor* real_time_assertor,
                        bool is_zeroing_out_velocity_and_acceleration=true);

    /**
     * NON-REAL-TIME!!!
     */
    DMPUnrollInitParams(const double& tau_init,
                        const Trajectory& trajectory,
                        RealTimeAssertor* real_time_assertor,
                        bool is_zeroing_out_velocity_and_acceleration=true);

    /**
     * NON-REAL-TIME!!!
     */
    DMPUnrollInitParams(const Trajectory& trajectory,
                        RealTimeAssertor* real_time_assertor,
                        bool is_zeroing_out_velocity_and_acceleration=true);

    /**
     * NON-REAL-TIME!!!
     */
    DMPUnrollInitParams(const Trajectory& trajectory,
                        const double& robot_task_servo_rate,
                        RealTimeAssertor* real_time_assertor,
                        bool is_zeroing_out_velocity_and_acceleration=true);

    /**
     * Checks validity of this data structure.
     *
     * @return Valid (true) or Invalid (false)
     */
    bool isValid() const;

    ~DMPUnrollInitParams();

};

typedef std::vector< DMPUnrollInitParams >              VecDMPUnrollInitParams;
typedef std::shared_ptr< VecDMPUnrollInitParams >       VecDMPUnrollInitParamsPtr;

typedef std::vector< VecDMPUnrollInitParams >           VecVecDMPUnrollInitParams;
typedef std::shared_ptr< VecVecDMPUnrollInitParams >    VecVecDMPUnrollInitParamsPtr;

}
#endif
