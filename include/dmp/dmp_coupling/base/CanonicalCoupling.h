/*
 * CanonicalCoupling.h
 *
 *  Abstract class defining coupling term for a canonical system of a DMP.
 *
 *  Created on: Jul 9, 2015
 *  Author: Giovanni Sutanto
 */

#ifndef CANONICAL_COUPLING_H
#define CANONICAL_COUPLING_H

#include "Coupling.h"

namespace dmp {

class CanonicalCoupling : public Coupling {
 public:
  /**
   * @param cc Coupling value (C_c) that will be used in the canonical system to
   * adjust time development of a DMP
   * @return Success or failure
   */
  virtual bool getValue(double& cc) = 0;

  virtual ~CanonicalCoupling() {}
};
}  // namespace dmp
#endif
