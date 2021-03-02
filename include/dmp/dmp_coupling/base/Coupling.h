/*
 * Coupling.h
 *
 *  Abstract class for DMP's coupling terms.
 *
 *  Created on: Jul 9, 2015
 *  Author: Giovanni Sutanto
 */

#ifndef COUPLING_H
#define COUPLING_H

#include "dmp/utility/RealTimeAssertor.h"

namespace dmp {
class Coupling {
 protected:
  RealTimeAssertor* rt_assertor;

 public:
  Coupling() : rt_assertor(NULL) {}

  /**
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  Coupling(RealTimeAssertor* real_time_assertor)
      : rt_assertor(real_time_assertor) {}

  /**
   * Checks whether this coupling term is valid or not.
   *
   * @return Function approximator is valid (true) or function approximator is
   * invalid (false)
   */
  virtual bool isValid() { return true; }

  virtual ~Coupling() {}
};
}  // namespace dmp
#endif
