#include "dmp/dmp_param/TauSystem.h"

namespace dmp {

TauSystem::TauSystem()
    : tau_base(0.0),
      tau_reference(0.0),
      tau_coupling(NULL),
      rt_assertor(NULL) {}

TauSystem::TauSystem(double tau_base_init, RealTimeAssertor* real_time_assertor,
                     double tau_ref, std::vector<TauCoupling*>* tau_couplers)
    : tau_base(tau_base_init),
      tau_reference(tau_ref),
      tau_coupling(tau_couplers),
      rt_assertor(real_time_assertor) {}

bool TauSystem::isValid() {
  if (rt_assert(tau_base >= MIN_TAU) == false) {
    return false;
  }
  if (rt_assert(tau_reference >= MIN_TAU) == false) {
    return false;
  }
  return true;
}

bool TauSystem::setTauBase(double tau_base_new) {
  // input checking
  if (rt_assert(tau_base_new >= MIN_TAU) ==
      false)  // new tau_base is too small/does NOT make sense
  {
    return false;
  }

  tau_base = tau_base_new;
  return true;
}

bool TauSystem::getTauBase(double& tau_base_return) {
  // pre-condition(s) checking
  if (rt_assert(TauSystem::isValid()) == false) {
    return false;
  }

  tau_base_return = tau_base;
  return true;
}

bool TauSystem::getTauRelative(double& tau_relative) {
  // pre-condition(s) checking
  if (rt_assert(TauSystem::isValid()) == false) {
    return false;
  }

  double C_tau = 0.0;
  if (rt_assert(TauSystem::getCouplingTerm(C_tau)) == false) {
    return false;
  }

  tau_relative = (1.0 + C_tau) * (tau_base / tau_reference);
  if (rt_assert((tau_relative * tau_reference) >= MIN_TAU) ==
      false)  // too small/does NOT make sense
  {
    tau_relative = 0.0;
    return false;
  }

  return true;
}

bool TauSystem::getCouplingTerm(double& accumulated_ctau) {
  accumulated_ctau = 0.0;

  if (tau_coupling) {
    for (int i = 0; i < tau_coupling->size(); ++i) {
      if ((*tau_coupling)[i] == NULL) {
        continue;
      }

      double ctau = 0.0;
      if (rt_assert((*tau_coupling)[i]->getValue(ctau)) == false) {
        return false;
      }
      if (rt_assert(std::isnan(ctau) == 0) == false) {
        return false;
      }

      accumulated_ctau += ctau;
    }
  }

  return true;
}

double TauSystem::getTauReference() { return tau_reference; }

TauSystem::~TauSystem() {}

}  // namespace dmp
