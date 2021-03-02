#include "dmp/dmp_discrete/CanonicalSystemDiscrete.h"

namespace dmp {

CanonicalSystemDiscrete::CanonicalSystemDiscrete()
    : CanonicalSystem(),
      order(2),
      alpha(25.0),
      beta(25.0 / 4.0),
      x(0),
      xd(0),
      xdd(0),
      v(0),
      vd(0) {}

CanonicalSystemDiscrete::CanonicalSystemDiscrete(
    TauSystem* tau_system, RealTimeAssertor* real_time_assertor, uint cs_order,
    std::vector<CanonicalCoupling*>* canonical_couplers)
    : CanonicalSystem(tau_system, real_time_assertor, canonical_couplers) {
  order = cs_order;
  if (order == 2) {
    alpha = 25.0;
    beta = alpha / 4.0;
  } else {
    order = 1;
    alpha = 25.0 / 3.0;
    beta = 0.0;
  }
  x = 1.0;
  xd = 0.0;
  xdd = 0.0;
  v = 0.0;
  vd = 0.0;
}

CanonicalSystemDiscrete::CanonicalSystemDiscrete(
    TauSystem* tau_system, uint cs_order, double cs_alpha,
    RealTimeAssertor* real_time_assertor,
    std::vector<CanonicalCoupling*>* canonical_couplers)
    : CanonicalSystem(tau_system, real_time_assertor, canonical_couplers) {
  order = cs_order;
  alpha = cs_alpha;
  if (order == 2) {
    beta = alpha / 4.0;
  } else {
    order = 1;
    beta = 0.0;
  }
  x = 1.0;
  xd = 0.0;
  xdd = 0.0;
  v = 0.0;
  vd = 0.0;
}

CanonicalSystemDiscrete::CanonicalSystemDiscrete(
    TauSystem* tau_system, uint cs_order, double cs_alpha, double cs_beta,
    RealTimeAssertor* real_time_assertor,
    std::vector<CanonicalCoupling*>* canonical_couplers)
    : CanonicalSystem(tau_system, real_time_assertor, canonical_couplers) {
  order = cs_order;
  alpha = cs_alpha;
  beta = cs_beta;
  x = 1.0;
  xd = 0.0;
  xdd = 0.0;
  v = 0.0;
  vd = 0.0;
}

bool CanonicalSystemDiscrete::isValid() {
  if (rt_assert(CanonicalSystem::isValid()) == false) {
    return false;
  }
  if (rt_assert((rt_assert(order >= 1)) && (rt_assert(order <= 2))) == false) {
    return false;
  }
  if (rt_assert(alpha > 0.0) == false) {
    return false;
  }
  if (rt_assert(beta >= 0.0) == false) {
    return false;
  }
  return true;
}

bool CanonicalSystemDiscrete::start() {
  if (rt_assert(CanonicalSystemDiscrete::isValid()) == false) {
    return false;
  }

  x = 1.0;
  xd = 0.0;
  xdd = 0.0;
  v = 0.0;
  vd = 0.0;

  is_started = true;

  return true;
}

uint CanonicalSystemDiscrete::getOrder() { return order; }

double CanonicalSystemDiscrete::getAlpha() { return alpha; }

double CanonicalSystemDiscrete::getBeta() { return beta; }

double CanonicalSystemDiscrete::getX() { return x; }

double CanonicalSystemDiscrete::getXd() { return xd; }

double CanonicalSystemDiscrete::getXdd() { return xdd; }

double CanonicalSystemDiscrete::getCanonicalPosition() { return x; }

double CanonicalSystemDiscrete::getCanonicalMultiplier() {
  if (order == 1) {
    return (x);
  } else if (order == 2) {
    return (v);
  } else {
    return 0.0;
  }
}

Vector3 CanonicalSystemDiscrete::getCanonicalState() {
  Vector3 canonical_state;
  canonical_state << x, xd, xdd;
  return (canonical_state);
}

bool CanonicalSystemDiscrete::updateCanonicalState(double dt) {
  if (rt_assert(is_started == true) == false) {
    return false;
  }
  if (rt_assert(CanonicalSystemDiscrete::isValid()) == false) {
    return false;
  }
  // input checking
  if (rt_assert(dt > 0.0) == false)  // dt does NOT make sense
  {
    return false;
  }

  double tau;
  if (rt_assert(tau_sys->getTauRelative(tau)) == false) {
    return false;
  }

  double C_c = 0.0;
  if (rt_assert(CanonicalSystem::getCouplingTerm(C_c)) == false) {
    return false;
  }

  if (order == 2) {
    xdd = vd / tau;
    vd = ((alpha * ((beta * (0 - x)) - v)) + C_c) / tau;
    xd = v / tau;
  } else {  // if (order == 1)
    xdd = vd / tau;
    vd = 0.0;  // vd  is irrelevant for 1st order canonical system, no need for
               // computation
    xd = ((alpha * (0 - x)) + C_c) / tau;
  }
  x = x + (xd * dt);
  v = v + (vd * dt);

  if (rt_assert(x >= 0.0) == false) {
    return false;
  }

  return true;
}

CanonicalSystemDiscrete::~CanonicalSystemDiscrete() {}

}  // namespace dmp
