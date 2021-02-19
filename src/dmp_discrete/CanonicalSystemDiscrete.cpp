#include "dmp/dmp_discrete/CanonicalSystemDiscrete.h"

namespace dmp
{

CanonicalSystemDiscrete::CanonicalSystemDiscrete():
    CanonicalSystem(), order(2), alpha(25.0), beta(25.0/4.0), x(0), xd(0), xdd(0), v(0), vd(0)
{}

/**
 * @param tau_system Tau system that computes the tau parameter for this canonical system
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 * @param cs_order The order of the dynamical system describing the canonical system
 * @param canonical_couplers All temporal couplings associated with the temporal DMP development
 * (please note that this is a pointer to vector of pointer to CanonicalCoupling data structure, for flexibility and re-usability)
 */
CanonicalSystemDiscrete::CanonicalSystemDiscrete(TauSystem* tau_system, RealTimeAssertor* real_time_assertor,
                                                 uint cs_order,
                                                 std::vector<CanonicalCoupling*>* canonical_couplers):
    CanonicalSystem(tau_system, real_time_assertor, canonical_couplers)
{
    order       = cs_order;
    if (order == 2)
    {
        alpha   = 25.0;
        beta    = alpha/4.0;
    }
    else
    {
        order   = 1;
        alpha   = 25.0/3.0;
        beta    = 0.0;
    }
    x           = 1.0;
    xd          = 0.0;
    xdd         = 0.0;
    v           = 0.0;
    vd          = 0.0;
}

/**
 * @param tau_system Tau system that computes the tau parameter for this canonical system
 * @param cs_order The order of the dynamical system describing the canonical system
 * @param cs_alpha Parameter alpha of the canonical system\n
 *        Best practice:\n
 *        If order==1, set it to 1/3 of alpha-parameter of the transformation system\n
 *        If order==2, set it equal to alpha-parameter of the transformation system (= 25.0)
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 * @param canonical_couplers All temporal couplings associated with the temporal DMP development
 * (please note that this is a pointer to vector of pointer to CanonicalCoupling data structure, for flexibility and re-usability)
 */
CanonicalSystemDiscrete::CanonicalSystemDiscrete(TauSystem* tau_system,
                                                 uint cs_order, double cs_alpha,
                                                 RealTimeAssertor* real_time_assertor,
                                                 std::vector<CanonicalCoupling*>* canonical_couplers):
    CanonicalSystem(tau_system, real_time_assertor, canonical_couplers)
{
    order       = cs_order;
    alpha       = cs_alpha;
	if (order == 2)
	{
        beta    = alpha/4.0;
	}
	else
	{
        order   = 1;
        beta    = 0.0;
    }
    x           = 1.0;
    xd          = 0.0;
    xdd         = 0.0;
    v           = 0.0;
    vd          = 0.0;
}

/**
 * @param tau_system Tau system that computes the tau parameter for this canonical system
 * @param cs_order The order of the dynamical system describing the canonical system
 * @param cs_alpha Parameter alpha of the canonical system\n
 *        Best practice:\n
 *        If order==1, set it to 1/3 of alpha-parameter of the transformation system\n
 *        If order==2, set it equal to alpha-parameter of the transformation system (= 25.0)
 * @param cs_beta Parameter beta of the canonical system\n
 *        Best practice: set it to 1/4 of alpha-parameter of the canonical system, to get a critically-damped response
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 * @param canonical_couplers All temporal couplings associated with the temporal DMP development
 * (please note that this is a pointer to vector of pointer to CanonicalCoupling data structure, for flexibility and re-usability)
 */
CanonicalSystemDiscrete::CanonicalSystemDiscrete(TauSystem* tau_system, uint cs_order,
                                                 double cs_alpha, double cs_beta,
                                                 RealTimeAssertor* real_time_assertor,
                                                 std::vector<CanonicalCoupling*>* canonical_couplers):
    CanonicalSystem(tau_system, real_time_assertor, canonical_couplers)
{
    order       = cs_order;
    alpha       = cs_alpha;
    beta        = cs_beta;
    x           = 1.0;
    xd          = 0.0;
    xdd         = 0.0;
    v           = 0.0;
    vd          = 0.0;
}

/**
 * Checks whether this discrete canonical system is valid or not.
 *
 * @return Discrete canonical system is valid (true) or discrete canonical system is invalid (false)
 */
bool CanonicalSystemDiscrete::isValid()
{
    if (rt_assert(CanonicalSystem::isValid()) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(order >= 1)) && (rt_assert(order <= 2))) == false)
    {
        return false;
    }
    if (rt_assert(alpha > 0.0) == false)
    {
        return false;
    }
    if (rt_assert(beta >= 0.0) == false)
    {
        return false;
    }
    return true;
}

/**
 * Resets canonical system and sets needed parameters.
 * Call this function before running any DMPs.
 *
 * @return Success or failure
 */
bool CanonicalSystemDiscrete::start()
{
    if (rt_assert(CanonicalSystemDiscrete::isValid()) == false)
    {
        return false;
    }

    x           = 1.0;
    xd          = 0.0;
    xdd         = 0.0;
    v           = 0.0;
    vd          = 0.0;

    is_started  = true;

    return true;
}

/**
 * Returns the order of the canonical system (1st or 2nd order).
 */
uint CanonicalSystemDiscrete::getOrder()
{
    return order;
}

/**
 * Returns the alpha parameter of the canonical system.
 */
double CanonicalSystemDiscrete::getAlpha()
{
    return alpha;
}

/**
 * Returns the beta parameter of the canonical system.
 */
double CanonicalSystemDiscrete::getBeta()
{
    return beta;
}

/**
 * Returns the current canonical state position.
 */
double CanonicalSystemDiscrete::getX()
{
    return x;
}

/**
 * Returns the current canonical state velocity.
 */
double CanonicalSystemDiscrete::getXd()
{
    return xd;
}

/**
 * Returns the current canonical state acceleration.
 */
double CanonicalSystemDiscrete::getXdd()
{
    return xdd;
}

/**
 * Returns canonical state position.
 */
double CanonicalSystemDiscrete::getCanonicalPosition()
{
    return x;
}

/**
 * Returns the canonical multiplier of the system:\n
 * if order==1, returns the canonical state position;\n
 * if order==2, returns the canonical state velocity.
 */
double CanonicalSystemDiscrete::getCanonicalMultiplier()
{
    if (order == 1)
    {
        return (x);
    }
    else if (order == 2)
    {
        return (v);
    }
    else
    {
        return 0.0;
    }
}

/**
 * Returns the state (position, velocity, and acceleration) of the canonical system.
 */
Vector3 CanonicalSystemDiscrete::getCanonicalState()
{
    Vector3 canonical_state;
    canonical_state << x, xd, xdd;
    return (canonical_state);
}

/**
 * Runs one time-step ahead of the canonical system and updates the current canonical state
 * based on the time difference dt from the previous execution.
 *
 * @param dt Time difference between the current and the previous execution
 * @return Success or failure
 */
bool CanonicalSystemDiscrete::updateCanonicalState(double dt)
{
    if (rt_assert(is_started == true) == false)
    {
        return false;
    }
    if (rt_assert(CanonicalSystemDiscrete::isValid()) == false)
    {
        return false;
    }
    // input checking
    if (rt_assert(dt > 0.0) == false)  // dt does NOT make sense
    {
        return false;
    }

    double  tau;
    if (rt_assert(tau_sys->getTauRelative(tau)) == false)
    {
        return false;
    }

    double C_c = 0.0;
    if (rt_assert(CanonicalSystem::getCouplingTerm(C_c)) == false)
    {
        return false;
    }

	if (order == 2)
	{
        xdd             = vd / tau;
        vd				= ((alpha * ((beta * (0 - x)) - v)) + C_c) / tau;
        xd				= v / tau;
	}
	else	// if (order == 1)
	{
        xdd             = vd / tau;
        vd              = 0.0;  // vd  is irrelevant for 1st order canonical system, no need for computation
        xd				= ((alpha * (0 - x)) + C_c) / tau;
	}
	x					= x + (xd * dt);
    v					= v + (vd * dt);

    if (rt_assert(x >= 0.0) == false)
    {
        return false;
    }

	return	true;
}

CanonicalSystemDiscrete::~CanonicalSystemDiscrete()
{}

}
