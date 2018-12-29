#include "amd_clmc_dmp/cart_dmp/quat_dmp/TransformSystemQuaternion.h"

namespace dmp
{

    TransformSystemQuaternion::TransformSystemQuaternion():
        TransformSystemDiscrete(), start_quat_state(QuaternionDMPState()), current_quat_state(QuaternionDMPState()),
        current_angular_velocity_state(DMPState()), quat_goal_sys(QuaternionGoalSystem())
    {}

    /**
     * Original formulation by Schaal, AMAM 2003 (default formulation type) is selected.
     *
     * @param canonical_system_discrete Discrete canonical system that drives this discrete transformation system
     * @param func_approximator_discrete Discrete function approximator that produces forcing term for this discrete transformation system
     * @param is_using_scaling_init Is transformation system using scaling (true/false)?
     * @param logged_dmp_discrete_vars (Pointer to the) buffer for logged discrete DMP variables
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     * @param transform_couplers All spatial couplings associated with the transformation system \n
     * (please note that this is a pointer to vector of pointer to TransformCoupling data structure, for flexibility and re-usability)
     * @param ts_alpha Spring parameter\n
     *        Best practice: set it to 25
     * @param ts_beta Damping parameter\n
     *        Best practice: set it to alpha / 4 for a critically damped system
     */
    TransformSystemQuaternion::TransformSystemQuaternion(CanonicalSystemDiscrete* canonical_system_discrete,
                                                         FuncApproximatorDiscrete* func_approximator_discrete,
                                                         std::vector<bool> is_using_scaling_init,
                                                         LoggedDMPDiscreteVariables* logged_dmp_discrete_vars,
                                                         RealTimeAssertor* real_time_assertor,
                                                         std::vector<TransformCoupling*>* transform_couplers,
                                                         double ts_alpha, double ts_beta):
        TransformSystemDiscrete(3, canonical_system_discrete, func_approximator_discrete,
                                is_using_scaling_init, ts_alpha, ts_beta, _SCHAAL_DMP_,
                                logged_dmp_discrete_vars, real_time_assertor,
                                &start_quat_state, &current_quat_state, &current_angular_velocity_state,
                                &quat_goal_sys, transform_couplers),
        start_quat_state(QuaternionDMPState(real_time_assertor)), current_quat_state(QuaternionDMPState(real_time_assertor)),
        current_angular_velocity_state(DMPState(3, real_time_assertor)),
        quat_goal_sys(QuaternionGoalSystem(tau_sys, real_time_assertor))
    {}

    /**
     * Checks whether this transformation system is valid or not.
     *
     * @return Transformation system is valid (true) or transformation system is invalid (false)
     */
    bool TransformSystemQuaternion::isValid()
    {
        if (rt_assert(TransformSystemDiscrete::isValid()) == false)
        {
            return false;
        }
        if (rt_assert(dmp_num_dimensions == 3) == false)
        {
            return false;
        }
        if (rt_assert(formulation_type == _SCHAAL_DMP_) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(start_quat_state.isValid()))    &&
                      (rt_assert(current_quat_state.isValid()))  &&
                      (rt_assert(current_angular_velocity_state.isValid()))) == false)
        {
            return false;
        }
        if (rt_assert(quat_goal_sys.isValid()) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(start_state            == &start_quat_state))               &&
                      (rt_assert(current_state          == &current_quat_state))             &&
                      (rt_assert(current_velocity_state == &current_angular_velocity_state)) &&
                      (rt_assert(goal_sys               == &quat_goal_sys))) == false)
        {
            return false;
        }
        return true;
    }

    /**
     * Start the transformation system, by setting the states properly.
     *
     * @param start_state_init Initialization of start state
     * @param goal_state_init Initialization of goal state
     * @return Success or failure
     */
    bool TransformSystemQuaternion::startTransformSystemQuaternion(const QuaternionDMPState& start_state_init, const QuaternionDMPState& goal_state_init)
    {
        // pre-conditions checking
        if (rt_assert(this->isValid()) == false)
        {
            return false;
        }
        // input checking
        if (rt_assert((rt_assert(start_state_init.isValid())) &&
                      (rt_assert(goal_state_init.isValid()))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(start_state_init.getDMPNumDimensions() == dmp_num_dimensions)) &&
                      (rt_assert(goal_state_init.getDMPNumDimensions()  == dmp_num_dimensions))) == false)
        {
            return false;
        }

        start_quat_state    = start_state_init;
        if (rt_assert(TransformSystemQuaternion::setCurrentQuaternionState(start_state_init)) == false)
        {
            return false;
        }

        Vector4     QG_init = goal_state_init.getQ();
        QuaternionDMPState  current_goal_state_init(rt_assertor);
        if (((CanonicalSystemDiscrete*) canonical_sys)->getOrder() == 2)
        {
            // Best option for Schaal's DMP Model using 2nd order canonical system:
            // Using goal evolution system initialized with the start position (state) as goal position (state),
            // which over time evolves toward a steady-state goal position (state).
            // The main purpose is to avoid discontinuous initial acceleration (see the math formula
            // of Schaal's DMP Model for more details).
            // Please also refer the initialization described in paper:
            // B. Nemec and A. Ude, “Action sequencing using dynamic movement
            // primitives,” Robotica, vol. 30, no. 05, pp. 837–846, 2012.
            Vector4 Q0      = start_state_init.getQ();
            Vector3 omega0  = start_state_init.getOmega();
            Vector3 omegad0 = start_state_init.getOmegad();

            double  tau;
            if (rt_assert(tau_sys->getTauRelative(tau)) == false)
            {
                return false;
            }

            Vector3 log_quat_diff_Qg_and_Q  = ((((tau*tau*omegad0)/alpha) + (tau*omega0))/(2.0 * beta));
            Vector4 quat_diff_Qg_and_Q      = ZeroVector4;
            if (rt_assert(computeExpMap_so3_to_SO3(log_quat_diff_Qg_and_Q, quat_diff_Qg_and_Q)) == false)
            {
                return false;
            }
            Vector4 Qg0                     = computeQuaternionComposition(quat_diff_Qg_and_Q, Q0);
            if (rt_assert(current_goal_state_init.setQ(Qg0)) == false)
            {
                return false;
            }
            if (rt_assert(current_goal_state_init.computeQdAndQdd()) == false)
            {
                return false;
            }
        }
        else
        {
            // Best option for Schaal's DMP Model using 1st order canonical system:
            // goal position is static, no evolution
            current_goal_state_init     = goal_state_init;
        }

        if (rt_assert(quat_goal_sys.startQuaternionGoalSystem(current_goal_state_init, QG_init)) == false)
        {
            return false;
        }

        is_started  = true;

        // post-conditions checking
        return (rt_assert(this->isValid()));
    }

    /**
     * Returns the next state (e.g. position, velocity, and acceleration) of the DMP.
     * [IMPORTANT]: Before calling this function, the TransformationSystem's start_state,
     * current_state, current_velocity_state, and goal_sys need to be initialized first using
     * TransformationSystem::start() function, using the desired start and goal states.
     *
     * @param dt Time difference between current and previous steps
     * @param next_state Current DMP state (that will be updated with the computed next state)
     * @param forcing_term (optional) If you also want to know the forcing term value of the next state, \n
     *                     please put the pointer here
     * @param coupling_term_acc (optional) If you also want to know the acceleration-level \n
     *                          coupling term value of the next state, please put the pointer here
     * @param coupling_term_vel (optional) If you also want to know the velocity-level \n
     *                          coupling term value of the next state, please put the pointer here
     * @param basis_functions_out (optional) (Pointer to) vector of basis functions constructing the forcing term
     * @param normalized_basis_func_vector_mult_phase_multiplier (optional) If also interested in \n
     *              the <normalized basis function vector multiplied by the canonical phase multiplier (phase position or phase velocity)> value, \n
     *              put its pointer here (return variable)
     * @return Success or failure
     */
    bool TransformSystemQuaternion::getNextQuaternionState(double dt, QuaternionDMPState& next_state,
                                                           VectorN* forcing_term,
                                                           VectorN* coupling_term_acc,
                                                           VectorN* coupling_term_vel,
                                                           VectorM* basis_functions_out,
                                                           VectorM* normalized_basis_func_vector_mult_phase_multiplier)
    {
        // pre-conditions checking
        if (rt_assert(is_started == true) == false)
        {
            return false;
        }
        if (rt_assert(this->isValid()) == false)
        {
            return false;
        }
        // input checking
        if (rt_assert(next_state.isValid()) == false)
        {
            return false;
        }
        if (rt_assert(next_state.getDMPNumDimensions() == dmp_num_dimensions) == false)
        {
            return false;
        }
        if (rt_assert(dt > 0.0) == false)   // dt does NOT make sense
        {
            return false;
        }

        double      tau;
        if (rt_assert(tau_sys->getTauRelative(tau)) == false)
        {
            return false;
        }

        VectorN     f(dmp_num_dimensions);
        VectorM     basis_functions(func_approx->getModelSize());
        if (rt_assert(func_approx->getForcingTerm(f, &basis_functions,
                                                  normalized_basis_func_vector_mult_phase_multiplier)) == false)
        {
            return false;
        }
        if (forcing_term)
        {
            (*forcing_term)         = f;
        }
        if (basis_functions_out)
        {
            (*basis_functions_out)  = basis_functions;
        }

        VectorN     ct_acc(dmp_num_dimensions);
        VectorN     ct_vel(dmp_num_dimensions);
        ct_acc      = ZeroVectorN(dmp_num_dimensions);
        ct_vel      = ZeroVectorN(dmp_num_dimensions);  // NOT used yet
        if (rt_assert(TransformationSystem::getCouplingTerm(ct_acc, ct_vel)) == false)
        {
            return false;
        }
        for (uint d=0; d<dmp_num_dimensions; ++d)
        {
            if (is_using_coupling_term_at_dimension[d] == false)
            {
                ct_acc[d]   = 0;
                ct_vel[d]   = 0;
            }
        }
        if (coupling_term_acc)
        {
            (*coupling_term_acc)    = ct_acc;
        }
        if (coupling_term_vel)
        {
            (*coupling_term_vel)    = ct_vel;
        }

        double      time    = current_quat_state.getTime();
        Vector4     Q0      = start_quat_state.getQ();
        Vector4     Q       = current_quat_state.getQ();
        Vector3     omega   = current_quat_state.getOmega();
        Vector3     omegad  = current_quat_state.getOmegad();

        Vector3     etha    = current_angular_velocity_state.getX();
        Vector3     ethad   = current_angular_velocity_state.getXd();

        Vector4     QG      = quat_goal_sys.getSteadyStateGoalPosition();
        Vector4     Qg      = quat_goal_sys.getCurrentQuaternionGoalState().getQ();

        if (rt_assert(integrateQuaternion(Q, omega, dt, Q)) == false)
        {
            return false;
        }
        omega               = omega + (omegad * dt);
        etha                = tau * omega;  // ct_vel is NOT used yet!

        // compute scaling factor for the forcing term:
        Vector3     A       = ZeroVector3;
        if (rt_assert(computeTwiceLogQuaternionDifference(QG, Q0, A)) == false)
        {
            return false;
        }

        for (uint n = 0; n < dmp_num_dimensions; ++n)
        {
            // The below hard-coded IF statement might still need improvement...
            if (is_using_scaling[n])
            {
                if (fabs(A_learn[n]) < MIN_FABS_AMPLITUDE)    // if goal state position is too close from the start state position
                {
                    A[n]    = 1.0;
                }
                else
                {
                    A[n]    = A[n]/A_learn[n];
                }
            }
            else
            {
                A[n]        = 1.0;
            }
        }

        // original formulation by Schaal, AMAM 2003
        Vector3     twice_log_quat_diff_Qg_and_Q    = ZeroVector3;
        if (rt_assert(computeTwiceLogQuaternionDifference(Qg, Q, twice_log_quat_diff_Qg_and_Q)) == false)
        {
            return false;
        }
        ethad   = ((alpha * ((beta * twice_log_quat_diff_Qg_and_Q) - etha)) + ((f).asDiagonal() * A) + ct_acc) / tau;

        if (rt_assert(containsNaN(ethad) == false) == false)
        {
            return false;
        }

        omegad                  = ethad / tau;

        time                    = time + dt;

        current_quat_state              = QuaternionDMPState(Q, omega, omegad, time, rt_assertor);
        if (rt_assert(current_quat_state.computeQdAndQdd()) == false)
        {
            return false;
        }
        current_angular_velocity_state  = DMPState(etha, ethad, ZeroVectorN(dmp_num_dimensions), time, rt_assertor);
        next_state                      = current_quat_state;

        /**************** For Trajectory Data Logging Purposes (Start) ****************/
//        logged_dmp_discrete_variables->tau                              = tau;
//        logged_dmp_discrete_variables->transform_sys_state_local        = current_state;
//        logged_dmp_discrete_variables->canonical_sys_state[0]           = ((CanonicalSystemDiscrete*) canonical_sys)->getX();
//        logged_dmp_discrete_variables->canonical_sys_state[1]           = ((CanonicalSystemDiscrete*) canonical_sys)->getXd();
//        logged_dmp_discrete_variables->canonical_sys_state[2]           = ((CanonicalSystemDiscrete*) canonical_sys)->getXdd();
//        logged_dmp_discrete_variables->goal_state_local                 = goal_sys->getCurrentGoalState();
//        logged_dmp_discrete_variables->steady_state_goal_position_local = G;
//        logged_dmp_discrete_variables->forcing_term                     = f;
//        logged_dmp_discrete_variables->basis_functions                  = basis_functions;
//        logged_dmp_discrete_variables->transform_sys_ct_acc             = ct_acc;
        /**************** For Trajectory Data Logging Purposes (End) ****************/

        // post-conditions checking
        return (rt_assert(this->isValid()));
    }

    /**
     * Based on the start state, current state, evolving goal state and steady-state goal position,
     * compute the target value of the forcing term.
     * Needed for learning the forcing function based on a given sequence of DMP states (trajectory).
     * [IMPORTANT]: Before calling this function, the TransformationSystem's start_state, current_state, and
     * goal_sys need to be initialized first using TransformationSystem::start() function,
     * using the demonstrated trajectory.
     *
     * @param current_state_demo_local Demonstrated current DMP state in the local coordinate system \n
     *                                 (if there are coordinate transformations)
     * @param f_target Target value of the forcing term at the demonstrated current state,\n
     *                 given the demonstrated start state, evolving goal state, and \n
     *                 demonstrated steady-state goal position \n
     *                 (to be computed and returned by this function)
     * @return Success or failure
     */
    bool TransformSystemQuaternion::getTargetQuaternionForcingTerm(const QuaternionDMPState& current_state_demo_local,
                                                                   VectorN& f_target)
    {
        // pre-conditions checking
        if (rt_assert(is_started == true) == false)
        {
            return false;
        }
        if (rt_assert(this->isValid()) == false)
        {
            return false;
        }
        // input checking
        if (rt_assert(current_state_demo_local.isValid()) == false)
        {
            return false;
        }
        if (rt_assert(current_state_demo_local.getDMPNumDimensions() == dmp_num_dimensions) == false)
        {
            return false;
        }
        if (rt_assert(f_target.rows() == dmp_num_dimensions) == false)
        {
            return false;
        }

        double  tau_demo;
        if (rt_assert(tau_sys->getTauRelative(tau_demo)) == false)
        {
            return false;
        }

        Vector4     Q0_demo     = start_quat_state.getQ();
        Vector4     Q_demo      = current_state_demo_local.getQ();
        Vector3     omega_demo  = current_state_demo_local.getOmega();
        Vector3     omegad_demo = current_state_demo_local.getOmegad();

        Vector4     Qg_demo     = quat_goal_sys.getCurrentQuaternionGoalState().getQ();

        // during learning, QG = QG_learn and Q0 = Q0_learn,
        // thus amplitude   = (computeTwiceLogQuaternionDifference(QG, Q0)/computeTwiceLogQuaternionDifference(QG_learn, Q0_learn))
        //                  = (computeTwiceLogQuaternionDifference(QG, Q0)/A_learn) = OnesVectorN(3),
        // as follows (commented out for computational effiency):
        //A_demo      = OnesVectorN(3);

        // original formulation by Schaal, AMAM 2003
        Vector3     twice_log_quat_diff_Qg_demo_and_Q_demo  = ZeroVector3;
        if (rt_assert(computeTwiceLogQuaternionDifference(Qg_demo, Q_demo, twice_log_quat_diff_Qg_demo_and_Q_demo)) == false)
        {
            return false;
        }
        f_target    = ((tau_demo * tau_demo * omegad_demo) -
                       (alpha * ((beta * twice_log_quat_diff_Qg_demo_and_Q_demo) - (tau_demo * omega_demo))));
        /*for (uint n = 0; n < dmp_num_dimensions; ++n)
        {
            f_target[n] /= A_demo[n];
        }*/

        return true;
    }

    /**
     * Based on the start state, current state, evolving goal state and steady-state goal position,
     * compute the target value of the coupling term.
     * Needed for learning the coupling term function based on a given sequence of DMP states (trajectory).
     * [IMPORTANT]: Before calling this function, the TransformationSystem's start_state, current_state, and
     * goal_sys need to be initialized first using TransformationSystem::start() function,
     * and the forcing term weights need to have already been learned beforehand,
     * using the demonstrated trajectory.
     *
     * @param current_state_demo_local Demonstrated current DMP state in the local coordinate system \n
     *                                 (if there are coordinate transformations)
     * @param ct_acc_target Target value of the acceleration-level coupling term at
     *                      the demonstrated current state,\n
     *                      given the demonstrated start state, evolving goal state, \n
     *                      demonstrated steady-state goal position, and
     *                      the learned forcing term weights \n
     *                      (to be computed and returned by this function)
     * @return Success or failure
     */
    bool TransformSystemQuaternion::getTargetQuaternionCouplingTerm(const QuaternionDMPState& current_state_demo_local,
                                                                    VectorN& ct_acc_target)
    {
        // pre-conditions checking
        if (rt_assert(is_started == true) == false)
        {
            return false;
        }
        if (rt_assert(this->isValid()) == false)
        {
            return false;
        }
        // input checking
        if (rt_assert(current_state_demo_local.isValid()) == false)
        {
            return false;
        }
        if (rt_assert(current_state_demo_local.getDMPNumDimensions() == dmp_num_dimensions) == false)
        {
            return false;
        }
        if (rt_assert(ct_acc_target.rows() == dmp_num_dimensions) == false)
        {
            return false;
        }

        double  tau_demo;
        if (rt_assert(tau_sys->getTauRelative(tau_demo)) == false)
        {
            return false;
        }

        VectorN     f(dmp_num_dimensions);
        if (rt_assert(func_approx->getForcingTerm(f)) == false)
        {
            return false;
        }

        Vector4     Q0_demo     = start_quat_state.getQ();
        Vector4     Q_demo      = current_state_demo_local.getQ();
        Vector3     omega_demo  = current_state_demo_local.getOmega();
        Vector3     omegad_demo = current_state_demo_local.getOmegad();

        Vector4     QG_demo     = quat_goal_sys.getSteadyStateGoalPosition();
        Vector4     Qg_demo     = quat_goal_sys.getCurrentQuaternionGoalState().getQ();

        // compute scaling factor for the forcing term:
        Vector3     A_demo      = ZeroVector3;
        if (rt_assert(computeTwiceLogQuaternionDifference(QG_demo, Q0_demo, A_demo)) == false)
        {
            return false;
        }

        for (uint n = 0; n < dmp_num_dimensions; ++n)
        {
            // The below hard-coded IF statement might still need improvement...
            if (is_using_scaling[n])
            {
                if (fabs(A_learn[n]) < MIN_FABS_AMPLITUDE)    // if goal state position is too close from the start state position
                {
                    A_demo[n]   = 1.0;
                }
                else
                {
                    A_demo[n]   = A_demo[n]/A_learn[n];
                }
            }
            else
            {
                A_demo[n]       = 1.0;
            }
        }

        // original formulation by Schaal, AMAM 2003
        Vector3     twice_log_quat_diff_Qg_demo_and_Q_demo  = ZeroVector3;
        if (rt_assert(computeTwiceLogQuaternionDifference(Qg_demo, Q_demo, twice_log_quat_diff_Qg_demo_and_Q_demo)) == false)
        {
            return false;
        }
        ct_acc_target   = ((tau_demo * tau_demo * omegad_demo) -
                           (alpha * ((beta * twice_log_quat_diff_Qg_demo_and_Q_demo) - (tau_demo * omega_demo))) -
                           ((f).asDiagonal() * A_demo));

        for (uint d=0; d<dmp_num_dimensions; ++d)
        {
            if (is_using_coupling_term_at_dimension[d] == false)
            {
                ct_acc_target[d]    = 0;
            }
        }

        return true;
    }

    /**
     * Returns transformation system's start state.
     *
     * @return Transformation system's start state (DMPState)
     */
    QuaternionDMPState TransformSystemQuaternion::getQuaternionStartState()
    {
        return start_quat_state;
    }

    /**
     * Sets new start state.
     *
     * @param new_start_state New start state to be set
     * @return Success or failure
     */
    bool TransformSystemQuaternion::setQuaternionStartState(const QuaternionDMPState& new_start_state)
    {
        // pre-conditions checking
        if (rt_assert(this->isValid()) == false)
        {
            return false;
        }
        // input checking
        if (rt_assert(new_start_state.isValid()) == false)
        {
            return false;
        }
        if (rt_assert(start_quat_state.getDMPNumDimensions() == new_start_state.getDMPNumDimensions()) == false)
        {
            return false;
        }

        start_quat_state    = new_start_state;

        // post-condition(s) checking
        return (rt_assert(this->isValid()));
    }

    /**
     * Returns current transformation system state.
     *
     * @return Current transformation system state (DMPState)
     */
    QuaternionDMPState TransformSystemQuaternion::getCurrentQuaternionState()
    {
        return current_quat_state;
    }

    /**
     * Sets new current state.
     *
     * @param new_current_state New current state to be set
     * @return Success or failure
     */
    bool TransformSystemQuaternion::setCurrentQuaternionState(const QuaternionDMPState& new_current_state)
    {
        // pre-conditions checking
        if (rt_assert(this->isValid()) == false)
        {
            return false;
        }
        // input checking
        if (rt_assert(new_current_state.isValid()) == false)
        {
            return false;
        }
        if (rt_assert(current_quat_state.getDMPNumDimensions() == new_current_state.getDMPNumDimensions()) == false)
        {
            return false;
        }

        current_quat_state  = new_current_state;

        if (rt_assert(this->updateCurrentVelocityStateFromCurrentState()) == false)
        {
            return false;
        }

        // post-condition(s) checking
        return (rt_assert(this->isValid()));
    }

    /**
     * Update current velocity state from current (position) state
     *
     * @return Success or failure
     */
    bool TransformSystemQuaternion::updateCurrentVelocityStateFromCurrentState()
    {
        // pre-conditions checking
        if (rt_assert(this->isValid()) == false)
        {
            return false;
        }

        double  tau;
        if (rt_assert(tau_sys->getTauRelative(tau)) == false)
        {
            return false;
        }

        Vector3 omega   = current_quat_state.getOmega();
        Vector3 omegad  = current_quat_state.getOmegad();

        Vector3 etha    = tau * omega;
        Vector3 ethad   = tau * omegad;

        if (rt_assert(current_angular_velocity_state.setX(etha)) == false)
        {
            return false;
        }
        if (rt_assert(current_angular_velocity_state.setXd(ethad)) == false)
        {
            return false;
        }

        // post-conditions checking
        return (rt_assert(this->isValid()));
    }

    /**
     * Returns the current transient/non-steady-state goal state vector.
     *
     * @return The current transient/non-steady-state goal state vector
     */
    QuaternionDMPState TransformSystemQuaternion::getCurrentQuaternionGoalState()
    {
        return (quat_goal_sys.getCurrentQuaternionGoalState());
    }

    /**
     * Sets new current goal state.
     *
     * @param new_current_goal_state New current goal state to be set
     * @return Success or failure
     */
    bool TransformSystemQuaternion::setCurrentQuaternionGoalState(const QuaternionDMPState& new_current_goal_state)
    {
        return (rt_assert(quat_goal_sys.setCurrentQuaternionGoalState(new_current_goal_state)));
    }

    TransformSystemQuaternion::~TransformSystemQuaternion()
    {}

}
