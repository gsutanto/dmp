/*
 * CanonicalSystem.h
 *
 *  Abstract class that contains a canonical system that drives a DMP.
 *
 *  Created on: Dec 12, 2014
 *  Author: Yevgen Chebotar and Giovanni Sutanto
 */

#ifndef CANONICAL_SYSTEM_H
#define CANONICAL_SYSTEM_H

#include <stdlib.h>
#include <stddef.h>
#include <vector>
#include <limits>
#include "dmp/utility/RealTimeAssertor.h"
#include "dmp/dmp_param/TauSystem.h"
#include "dmp/dmp_state/DMPState.h"
#include "dmp/dmp_coupling/base/CanonicalCoupling.h"

namespace dmp
{

class CanonicalSystem
{

protected:

    // The following field(s) are written as RAW pointers because we just want to point
    // to other data structure(s) that might outlive the instance of this class:
    TauSystem*                          tau_sys;
    std::vector<CanonicalCoupling*>*    canonical_coupling;
    RealTimeAssertor*                   rt_assertor;

    bool                                is_started;

public:

    CanonicalSystem();

    /**
     * @param tau_system Tau system that computes the tau parameter for this canonical system
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     * @param canonical_couplers All temporal couplings associated with the temporal DMP development \n
     * (please note that this is a pointer to vector of pointer to CanonicalCoupling data structure, for flexibility and re-usability)
     */
    CanonicalSystem(TauSystem* tau_system,
                    RealTimeAssertor* real_time_assertor,
                    std::vector<CanonicalCoupling*>* canonical_couplers=NULL);

    /**
     * Checks whether this canonical system is valid or not.
     *
     * @return Canonical system is valid (true) or canonical system is invalid (false)
     */
    virtual bool isValid();

    /**
     * Resets canonical system and sets needed parameters.
     * Call this function before running any DMPs.
     *
     * @return Success or failure
     */
    virtual bool start() = 0;

    /**
     * Returns canonical state position.
     */
    virtual double getCanonicalPosition() = 0;

    /**
     * Returns the canonical multiplier of the system (especially relevant in CanonicalSystemDiscrete child class).
     */
    virtual double getCanonicalMultiplier() = 0;

    /**
     * Returns the state (position, velocity, and acceleration) of the canonical system.
     */
    virtual Vector3 getCanonicalState() = 0;

    /**
     * Runs one time-step ahead of the canonical system and updates the current canonical state
     * based on the time difference dt from the previous execution.
     *
     * @param dt Time difference between the current and the previous execution
     */
    virtual bool updateCanonicalState(double dt) = 0;

    /**
     * If there are coupling terms on the canonical system,
     * this function will iterate over each coupling term, and returns the accumulated coupling term.
     * Otherwise it will return 0.0.
     *
     * @param accumulated_cc The accumulated canonical system coupling term value (to be returned)
     * @return Success or failure
     */
    virtual bool getCouplingTerm(double& accumulated_cc);

    /**
     * Returns the tau system pointer.
     */
    virtual TauSystem* getTauSystemPointer();

    ~CanonicalSystem();

private:

};
}
#endif
