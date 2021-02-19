#include "dmp/dmp_discrete/LoggedDMPDiscreteVariables.h"

namespace dmp
{

LoggedDMPDiscreteVariables::LoggedDMPDiscreteVariables():
    LoggedDMPVariables(), canonical_sys_state(ZeroVector3)
{}

/**
 * @param dmp_num_dimensions_init Number of DMP dimensions (N) to use
 * @param num_basis_functions Number of basis functions (M) to use
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 */
LoggedDMPDiscreteVariables::LoggedDMPDiscreteVariables(uint dmp_num_dimensions_init,
                                                       uint num_basis_functions,
                                                       RealTimeAssertor* real_time_assertor):
    LoggedDMPVariables(dmp_num_dimensions_init, num_basis_functions, real_time_assertor),
    canonical_sys_state(ZeroVector3)
{}

bool LoggedDMPDiscreteVariables::isValid() const
{
    if (rt_assert(LoggedDMPVariables::isValid()) == false)
    {
        return false;
    }
    return true;
}

LoggedDMPDiscreteVariables::~LoggedDMPDiscreteVariables()
{}

}
