#include "amd_clmc_dmp/dmp_param/TauSystem.h"

namespace dmp
{

TauSystem::TauSystem():
    tau_base(0.0), tau_reference(0.0), tau_coupling(NULL), rt_assertor(NULL)
{}

/**
 * @param tau_base_init Basic tau value (without any addition of coupling terms)
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 * @param tau_ref Reference tau value:\n
 *                all computations which involve tau, is relative to this value
 * @param tau_couplers All couplings associated with this tau system
 * (please note that this is a pointer to vector of pointer to TauCoupling data structure, for flexibility and re-usability)
 */
TauSystem::TauSystem(double tau_base_init, RealTimeAssertor* real_time_assertor,
                     double tau_ref, std::vector<TauCoupling*>* tau_couplers):
    tau_base(tau_base_init), tau_reference(tau_ref), tau_coupling(tau_couplers),
    rt_assertor(real_time_assertor)
{}

/**
 * Checks whether this tau system is valid or not.
 *
 * @return Tau system is valid (true) or tau system is invalid (false)
 */
bool TauSystem::isValid()
{
    if (rt_assert(tau_base >= MIN_TAU) == false)
    {
        return false;
    }
    if (rt_assert(tau_reference >= MIN_TAU) == false)
    {
        return false;
    }
    return true;
}

/**
 * @param tau_base_new New basic tau value (without any addition of coupling terms)\n
 *                     (usually this is the time-length of the trajectory)
 * @return Success or failure
 */
bool TauSystem::setTauBase(double tau_base_new)
{
    // input checking
    if (rt_assert(tau_base_new >= MIN_TAU) == false)    // new tau_base is too small/does NOT make sense
    {
        return false;
    }

    tau_base        = tau_base_new;
    return true;
}

/**
 * @param tau_base Basic tau value (without any addition of coupling terms, to be returned)\n
 *                 (usually this is the time-length of the trajectory)
 * @return Success or failure
 */
bool TauSystem::getTauBase(double& tau_base_return)
{
    // pre-condition(s) checking
    if (rt_assert(TauSystem::isValid()) == false)
    {
        return false;
    }

    tau_base_return = tau_base;
    return true;
}

/**
 * @param tau_relative Relative tau value (to be computed and returned by this function),\n
 *                     i.e. time parameter that influences the speed of the execution,\n
 *                     relative to a reference tau (which is usually 0.5 seconds)
 * @return Success or failure
 */
bool TauSystem::getTauRelative(double& tau_relative)
{
    // pre-condition(s) checking
    if (rt_assert(TauSystem::isValid()) == false)
    {
        return false;
    }

    double C_tau    = 0.0;
    if (rt_assert(TauSystem::getCouplingTerm(C_tau)) == false)
    {
        return false;
    }

    tau_relative    = (1.0 + C_tau) * (tau_base / tau_reference);
    if (rt_assert((tau_relative * tau_reference) >= MIN_TAU) == false) // too small/does NOT make sense
    {
        tau_relative= 0.0;
        return false;
    }

    return true;
}

/**
 * If there are coupling terms on the tau system,
 * this function will iterate over each coupling term, and returns the accumulated coupling term.
 * Otherwise it will return 0.0.
 *
 * @param accumulated_ctau The accumulated tau system coupling term value (to be returned)
 * @return Success or failure
 */
bool TauSystem::getCouplingTerm(double& accumulated_ctau)
{
    accumulated_ctau                = 0.0;

    if (tau_coupling)
    {
        for (int i=0; i<tau_coupling->size(); ++i)
        {
            if ((*tau_coupling)[i] == NULL)
            {
                continue;
            }

            double ctau             = 0.0;
            if (rt_assert((*tau_coupling)[i]->getValue(ctau)) == false)
            {
                return false;
            }
            if (rt_assert(isnan(ctau) == 0) == false)
            {
                return false;
            }

            accumulated_ctau        += ctau;
        }
    }

    return true;
}

TauSystem::~TauSystem()
{}

}
