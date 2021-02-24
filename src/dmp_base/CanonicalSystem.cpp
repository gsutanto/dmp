#include "dmp/dmp_base/CanonicalSystem.h"

namespace dmp
{

    CanonicalSystem::CanonicalSystem():
        tau_sys(NULL), canonical_coupling(NULL), is_started(false), rt_assertor(NULL)
    {}

    /**
     * @param tau_system Tau system that computes the tau parameter for this canonical system
     * @param canonical_couplers All temporal couplings associated with the temporal DMP development \n
     * (please note that this is a pointer to vector of pointer to CanonicalCoupling data structure, for flexibility and re-usability)
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    CanonicalSystem::CanonicalSystem(TauSystem* tau_system,
                                     RealTimeAssertor* real_time_assertor,
                                     std::vector<CanonicalCoupling*>* canonical_couplers):
        tau_sys(tau_system), canonical_coupling(canonical_couplers), is_started(false), rt_assertor(real_time_assertor)
    {}

    /**
     * Checks whether this canonical system is valid or not.
     *
     * @return Canonical system is valid (true) or canonical system is invalid (false)
     */
    bool CanonicalSystem::isValid()
    {
        if (rt_assert(tau_sys != NULL) == false)
        {
            return false;
        }
        if (rt_assert(tau_sys->isValid()) == false)
        {
            return false;
        }
        return true;
    }

    /**
     * If there are coupling terms on the canonical system,
     * this function will iterate over each coupling term, and returns the accumulated coupling term.
     * Otherwise it will return 0.0.
     *
     * @param accumulated_cc The accumulated canonical system coupling term value (to be returned)
     * @return Success or failure
     */
    bool CanonicalSystem::getCouplingTerm(double& accumulated_cc)
    {
        accumulated_cc                  = 0.0;

        if (canonical_coupling)
        {
            for (uint i=0; i<canonical_coupling->size(); ++i)
            {
                if ((*canonical_coupling)[i] == NULL)
                {
                    continue;
                }

                double cc               = 0.0;
                if (rt_assert((*canonical_coupling)[i]->getValue(cc)) == false)
                {
                    return false;
                }
                if (rt_assert(std::isnan(cc) == 0) == false)
                {
                    return false;
                }

                accumulated_cc          += cc;
            }
        }

        return true;
    }

    /**
     * Returns the tau system pointer.
     */
    TauSystem* CanonicalSystem::getTauSystemPointer()
    {
        return	tau_sys;
    }

    CanonicalSystem::~CanonicalSystem()
    {}
}
