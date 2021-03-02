#include "dmp/dmp_base/CanonicalSystem.h"

namespace dmp {

CanonicalSystem::CanonicalSystem()
    : tau_sys(NULL),
      canonical_coupling(NULL),
      is_started(false),
      rt_assertor(NULL) {}

CanonicalSystem::CanonicalSystem(
    TauSystem* tau_system, RealTimeAssertor* real_time_assertor,
    std::vector<CanonicalCoupling*>* canonical_couplers)
    : tau_sys(tau_system),
      canonical_coupling(canonical_couplers),
      is_started(false),
      rt_assertor(real_time_assertor) {}

bool CanonicalSystem::isValid() {
  if (rt_assert(tau_sys != NULL) == false) {
    return false;
  }
  if (rt_assert(tau_sys->isValid()) == false) {
    return false;
  }
  return true;
}

bool CanonicalSystem::getCouplingTerm(double& accumulated_cc) {
  accumulated_cc = 0.0;

  if (canonical_coupling) {
    for (uint i = 0; i < canonical_coupling->size(); ++i) {
      if ((*canonical_coupling)[i] == NULL) {
        continue;
      }

      double cc = 0.0;
      if (rt_assert((*canonical_coupling)[i]->getValue(cc)) == false) {
        return false;
      }
      if (rt_assert(std::isnan(cc) == 0) == false) {
        return false;
      }

      accumulated_cc += cc;
    }
  }

  return true;
}

TauSystem* CanonicalSystem::getTauSystemPointer() { return tau_sys; }

CanonicalSystem::~CanonicalSystem() {}
}  // namespace dmp
