/*
 * LoggedDMPDiscreteVariables.h
 *
 *  Temporary buffer to store the current value of the discrete DMP variables
 * to-be-logged. The values stored here will later be copied into data
 * structures of trajectories in DMPDataIODiscrete, and eventually be dumped
 * into a file (for logging purposes).
 *
 *  Created on: Jan 3, 2016
 *  Author: Giovanni Sutanto
 */

#ifndef LOGGED_DMP_DISCRETE_VARIABLES_H
#define LOGGED_DMP_DISCRETE_VARIABLES_H

#include "dmp/dmp_base/LoggedDMPVariables.h"

namespace dmp {

class LoggedDMPDiscreteVariables : public LoggedDMPVariables {
 public:
  Vector3 canonical_sys_state;

  LoggedDMPDiscreteVariables();

  /**
   * @param dmp_num_dimensions_init Number of DMP dimensions (N) to use
   * @param num_basis_functions Number of basis functions (M) to use
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  LoggedDMPDiscreteVariables(uint dmp_num_dimensions_init,
                             uint num_basis_functions,
                             RealTimeAssertor* real_time_assertor);

  bool isValid() const;

  ~LoggedDMPDiscreteVariables();
};

}  // namespace dmp
#endif
