/*
 * LoggedDMPVariables.h
 *
 *  Temporary buffer to store the current value of the DMP variables to-be-logged.
 *  The values stored here will later be copied into data structures of trajectories in DMPDataIO,
 *  and eventually be dumped into a file (for logging purposes).
 *
 *  Created on: Jan 3, 2016
 *  Author: Giovanni Sutanto
 */

#ifndef LOGGED_DMP_VARIABLES_H
#define LOGGED_DMP_VARIABLES_H

#include "dmp/utility/DefinitionsBase.h"
#include "dmp/utility/RealTimeAssertor.h"
#include "dmp/dmp_param/TauSystem.h"
#include "dmp/dmp_state/DMPState.h"

namespace dmp
{

class LoggedDMPVariables
{

protected:

    uint                dmp_num_dimensions;
    uint                model_size;
    RealTimeAssertor*   rt_assertor;

public:

    double              tau;
    DMPState            transform_sys_state_local;
    DMPState            transform_sys_state_global;
    DMPState            goal_state_local;
    DMPState            goal_state_global;
    VectorN             steady_state_goal_position_local;
    VectorN             steady_state_goal_position_global;
    VectorM             basis_functions;
    VectorN             forcing_term;
    VectorN             transform_sys_ct_acc;

    LoggedDMPVariables();

    /**
     * @param dmp_num_dimensions_init Number of DMP dimensions (N) to use
     * @param num_basis_functions Number of basis functions (M) to use
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    LoggedDMPVariables(uint dmp_num_dimensions_init,
                       uint num_basis_functions,
                       RealTimeAssertor* real_time_assertor);

    virtual bool isValid() const;

    virtual uint getDMPNumDimensions() const;

    virtual uint getModelSize() const;

    ~LoggedDMPVariables();

};

}
#endif
