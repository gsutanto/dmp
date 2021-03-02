/*
 * TauCoupling.h
 *
 *  Abstract class defining coupling term for tau parameter of a DMP.
 *
 *  Created on: Jul 9, 2015
 *  Author: Giovanni Sutanto
 */

#ifndef TAU_COUPLING_H
#define TAU_COUPLING_H

#include "Coupling.h"

namespace dmp {
class TauCoupling : public Coupling {
 public:
  TauCoupling() : Coupling() {}

  /**
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  TauCoupling(RealTimeAssertor* real_time_assertor)
      : Coupling(real_time_assertor) {}

  /**
   * @param ctau Coupling value (C_tau) that affects tau parameter (of canonical
   * system, transformation system, and goal changer) of a DMP
   * @return Success or failure
   */
  virtual bool getValue(double& ctau) = 0;

  virtual ~TauCoupling() {}

 protected:
 private:
};
}  // namespace dmp
#endif
