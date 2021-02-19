/*
 * TauSystem.h
 *
 *  Class defining computation of tau parameter for canonical system, transformation system, and goal changer, which sometime is coupled with an external variable
 *
 *  Created on: Jul 10, 2015
 *  Author: Giovanni Sutanto
 */

#ifndef TAU_SYSTEM_H
#define TAU_SYSTEM_H

#include <stdlib.h>
#include <stddef.h>
#include <vector>
#include <memory>

#include "amd_clmc_dmp/utility/RealTimeAssertor.h"
#include "amd_clmc_dmp/dmp_coupling/base/TauCoupling.h"

namespace dmp
{

class TauSystem
{

private:

    double                      tau_base;
    double                      tau_reference;

    // The following field(s) are written as RAW pointers because we just want to point
    // to other data structure(s) that might outlive the instance of this class:
    std::vector<TauCoupling*>*  tau_coupling;
    RealTimeAssertor*           rt_assertor;

public:

    TauSystem();

    /**
     * @param tau_base_init Basic tau value (without any addition of coupling terms)
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     * @param tau_ref Reference tau value:\n
     *                all computations which involve tau, is relative to this value
     * @param tau_couplers All couplings associated with this tau system
     * (please note that this is a pointer to vector of pointer to TauCoupling data structure, for flexibility and re-usability)
     */
    TauSystem(double tau_base_init, RealTimeAssertor* real_time_assertor,
              double tau_ref=0.5, std::vector<TauCoupling*>* tau_couplers=NULL);

    /**
     * Checks whether this tau system is valid or not.
     *
     * @return Tau system is valid (true) or tau system is invalid (false)
     */
    bool isValid();

    /**
     * @param tau_base_new New basic tau value (without any addition of coupling terms)\n
     *                     (usually this is the time-length of the trajectory)
     * @return Success or failure
     */
    bool setTauBase(double tau_base_new);

    /**
     * @param tau_base Basic tau value (without any addition of coupling terms, to be returned)\n
     *                 (usually this is the time-length of the trajectory)
     * @return Success or failure
     */
    bool getTauBase(double& tau_base_return);

    /**
     * @param tau_relative Relative tau value (to be computed and returned by this function),\n
     *                     i.e. time parameter that influences the speed of the execution,\n
     *                     relative to a reference tau (which is usually 0.5 seconds)
     * @return Success or failure
     */
    bool getTauRelative(double& tau_relative);

    /**
     * If there are coupling terms on the tau system,
     * this function will iterate over each coupling term, and returns the accumulated coupling term.
     * Otherwise it will return 0.0.
     *
     * @param accumulated_ctau The accumulated tau system coupling term value (to be returned)
     * @return Success or failure
     */
    bool getCouplingTerm(double& accumulated_ctau);

    double getTauReference();

    ~TauSystem();

protected:

};
}
#endif
