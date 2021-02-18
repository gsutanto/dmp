/*
 * CanonicalSystemDiscrete.h
 *
 *  Canonical system of a DMP for a discrete movement (non-rhythmic)
 *
 *  See A.J. Ijspeert, J. Nakanishi, H. Hoffmann, P. Pastor, and S. Schaal;
 *      Dynamical movement primitives: Learning attractor models for motor behaviors;
 *      Neural Comput. 25, 2 (February 2013), 328-373
 *
 *  Created on: Dec 12, 2014
 *  Author: Yevgen Chebotar and Giovanni Sutanto
 */

#ifndef CANONICAL_SYSTEM_DISCRETE_H
#define CANONICAL_SYSTEM_DISCRETE_H

#include "amd_clmc_dmp/dmp_base/CanonicalSystem.h"

namespace dmp
{

class CanonicalSystemDiscrete : public CanonicalSystem
{

protected:

    uint    order;
    double	alpha;
    double	beta;
    double	x;
    double	xd;
    double	xdd;
    double  v;
    double  vd;

public:

    CanonicalSystemDiscrete();

    /**
     * @param tau_system Tau system that computes the tau parameter for this canonical system
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     * @param cs_order The order of the dynamical system describing the canonical system
     * @param canonical_couplers All temporal couplings associated with the temporal DMP development
     * (please note that this is a pointer to vector of pointer to CanonicalCoupling data structure, for flexibility and re-usability)
     */
    CanonicalSystemDiscrete(TauSystem* tau_system, RealTimeAssertor* real_time_assertor, uint cs_order=2,
                            std::vector<CanonicalCoupling*>* canonical_couplers=NULL);

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
    CanonicalSystemDiscrete(TauSystem* tau_system, uint cs_order, double cs_alpha,
                            RealTimeAssertor* real_time_assertor,
                            std::vector<CanonicalCoupling*>* canonical_couplers=NULL);

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
    CanonicalSystemDiscrete(TauSystem* tau_system, uint cs_order,
                            double cs_alpha, double cs_beta,
                            RealTimeAssertor* real_time_assertor,
                            std::vector<CanonicalCoupling*>* canonical_couplers=NULL);

    /**
     * Checks whether this discrete canonical system is valid or not.
     *
     * @return Discrete canonical system is valid (true) or discrete canonical system is invalid (false)
     */
    bool isValid();

    /**
     * Resets canonical system and sets needed parameters.
     * Call this function before running any DMPs.
     *
     * @return Success or failure
     */
    bool start();

    /**
     * Returns the order of the canonical system (1st or 2nd order).
     */
    uint getOrder();

    /**
     * Returns the alpha parameter of the canonical system.
     */
    double getAlpha();

    /**
     * Returns the beta parameter of the canonical system.
     */
    double getBeta();

    /**
     * Returns the current canonical state position.
     */
    double getX();

    /**
     * Returns the current canonical state velocity.
     */
    double getXd();

    /**
     * Returns the current canonical state acceleration.
     */
    double getXdd();

    /**
     * Returns canonical state position.
     */
    double getCanonicalPosition();

    /**
     * Returns the canonical multiplier of the system:\n
     * if order==1, returns the canonical state position;\n
     * if order==2, returns the canonical state velocity.
     */
    double getCanonicalMultiplier();

    /**
     * Returns the state (position, velocity, and acceleration) of the canonical system.
     */
    Vector3 getCanonicalState();

    /**
     * Runs one time-step ahead of the canonical system and updates the current canonical state
     * based on the time difference dt from the previous execution.
     *
     * @param dt Time difference between the current and the previous execution
     * @return Success or failure
     */
    bool updateCanonicalState(double dt);

    ~CanonicalSystemDiscrete();

};
}
#endif
