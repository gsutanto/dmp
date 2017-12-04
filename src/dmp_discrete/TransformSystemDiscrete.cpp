#include "amd_clmc_dmp/dmp_discrete/TransformSystemDiscrete.h"

namespace dmp
{

TransformSystemDiscrete::TransformSystemDiscrete():
    TransformationSystem(), formulation_type(_SCHAAL_DMP_), alpha(25.0), beta(25.0/4.0),
    is_using_scaling(std::vector<bool>()), A_learn(OnesVectorN(MAX_DMP_NUM_DIMENSIONS)),
    logged_dmp_discrete_variables(NULL)
{
    A_learn.resize(dmp_num_dimensions);
}

/**
 * Original formulation by Schaal, AMAM 2003 (default formulation_type) is selected.\n
 * alpha 	= 25	(default)\n
 * beta		= 25/4	(default)
 *
 * @param dmp_num_dimensions_init Number of DMP dimensions (N) to use
 * @param canonical_system_discrete Discrete canonical system that drives this discrete transformation system
 * @param func_approximator_discrete Discrete function approximator that produces forcing term for this discrete transformation system
 * @param is_using_scaling_init Is transformation system using scaling (true/false)?
 * @param logged_dmp_discrete_vars (Pointer to the) buffer for logged discrete DMP variables
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 * @param start_dmpstate_discrete (Pointer to the) start state of the discrete DMP's transformation system
 * @param current_dmpstate_discrete (Pointer to the) current state of the discrete DMP's transformation system
 * @param current_velocity_dmpstate_discrete (Pointer to the) current velocity state of the discrete DMP's transformation system
 * @param goal_system_discrete (Pointer to the) goal system for discrete DMP's transformation system
 * @param transform_couplers All spatial couplings associated with the transformation system
 * (please note that this is a pointer to vector of pointer to TransformCoupling data structure, for flexibility and re-usability)
 */
TransformSystemDiscrete::TransformSystemDiscrete(uint dmp_num_dimensions_init,
                                                 CanonicalSystemDiscrete* canonical_system_discrete,
                                                 FuncApproximatorDiscrete* func_approximator_discrete,
                                                 std::vector<bool> is_using_scaling_init,
                                                 LoggedDMPDiscreteVariables* logged_dmp_discrete_vars,
                                                 RealTimeAssertor* real_time_assertor,
                                                 DMPState* start_dmpstate_discrete, DMPState* current_dmpstate_discrete,
                                                 DMPState* current_velocity_dmpstate_discrete, GoalSystem* goal_system_discrete,
                                                 std::vector<TransformCoupling*>* transform_couplers):
    TransformationSystem(dmp_num_dimensions_init, canonical_system_discrete, func_approximator_discrete,
                         logged_dmp_discrete_vars, real_time_assertor,
                         start_dmpstate_discrete, current_dmpstate_discrete, current_velocity_dmpstate_discrete,
                         goal_system_discrete, transform_couplers),
    formulation_type(_SCHAAL_DMP_), alpha(25.0), beta(25.0/4.0), is_using_scaling(is_using_scaling_init),
    A_learn(OnesVectorN(MAX_DMP_NUM_DIMENSIONS)),
    logged_dmp_discrete_variables(logged_dmp_discrete_vars)
{
    A_learn.resize(dmp_num_dimensions);
}

/**
 * Original formulation by Schaal, AMAM 2003 (default formulation_type) is selected.
 *
 * @param dmp_num_dimensions_init Number of DMP dimensions (N) to use
 * @param canonical_system_discrete Discrete canonical system that drives this discrete transformation system
 * @param func_approximator_discrete Discrete function approximator that produces forcing term for this discrete transformation system
 * @param is_using_scaling_init Is transformation system using scaling (true/false)?
 * @param ts_alpha Spring parameter\n
 *        Best practice: set it to 25
 * @param ts_beta Damping parameter\n
 *        Best practice: set it to alpha / 4 for a critically damped system
 * @param logged_dmp_discrete_vars (Pointer to the) buffer for logged discrete DMP variables
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 * @param start_dmpstate_discrete (Pointer to the) start state of the discrete DMP's transformation system
 * @param current_dmpstate_discrete (Pointer to the) current state of the discrete DMP's transformation system
 * @param current_velocity_dmpstate_discrete (Pointer to the) current velocity state of the discrete DMP's transformation system
 * @param goal_system_discrete (Pointer to the) goal system for discrete DMP's transformation system
 * @param transform_couplers All spatial couplings associated with the transformation system
 * (please note that this is a pointer to vector of pointer to TransformCoupling data structure, for flexibility and re-usability)
 */
TransformSystemDiscrete::TransformSystemDiscrete(uint dmp_num_dimensions_init,
                                                 CanonicalSystemDiscrete* canonical_system_discrete,
                                                 FuncApproximatorDiscrete* func_approximator_discrete,
                                                 std::vector<bool> is_using_scaling_init,
                                                 double ts_alpha, double ts_beta,
                                                 LoggedDMPDiscreteVariables* logged_dmp_discrete_vars,
                                                 RealTimeAssertor* real_time_assertor,
                                                 DMPState* start_dmpstate_discrete, DMPState* current_dmpstate_discrete,
                                                 DMPState* current_velocity_dmpstate_discrete, GoalSystem* goal_system_discrete,
                                                 std::vector<TransformCoupling*>* transform_couplers):
    TransformationSystem(dmp_num_dimensions_init, canonical_system_discrete, func_approximator_discrete,
                         logged_dmp_discrete_vars, real_time_assertor,
                         start_dmpstate_discrete, current_dmpstate_discrete, current_velocity_dmpstate_discrete,
                         goal_system_discrete, transform_couplers),
    formulation_type(_SCHAAL_DMP_), alpha(ts_alpha), beta(ts_beta), is_using_scaling(is_using_scaling_init),
    A_learn(OnesVectorN(MAX_DMP_NUM_DIMENSIONS)), logged_dmp_discrete_variables(logged_dmp_discrete_vars)
{
    A_learn.resize(dmp_num_dimensions);
}

/**
 * alpha    = 25	(default)\n
 * beta		= 25/4	(default)
 *
 * @param dmp_num_dimensions_init Number of DMP dimensions (N) to use
 * @param canonical_system_discrete Discrete canonical system that drives this discrete transformation system
 * @param func_approximator_discrete Discrete function approximator that produces forcing term for this discrete transformation system
 * @param is_using_scaling_init Is transformation system using scaling (true/false)?
 * @param ts_formulation_type Chosen formulation\n
 *        _SCHAAL_DMP_: original formulation by Schaal, AMAM 2003 (default formulation_type)\n
 *        _HOFFMANN_DMP_: formulation by Hoffmann et al., ICRA 2009
 * @param logged_dmp_discrete_vars (Pointer to the) buffer for logged discrete DMP variables
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 * @param start_dmpstate_discrete (Pointer to the) start state of the discrete DMP's transformation system
 * @param current_dmpstate_discrete (Pointer to the) current state of the discrete DMP's transformation system
 * @param current_velocity_dmpstate_discrete (Pointer to the) current velocity state of the discrete DMP's transformation system
 * @param goal_system_discrete (Pointer to the) goal system for discrete DMP's transformation system
 * @param transform_couplers All spatial couplings associated with the transformation system
 * (please note that this is a pointer to vector of pointer to TransformCoupling data structure, for flexibility and re-usability)
 */
TransformSystemDiscrete::TransformSystemDiscrete(uint dmp_num_dimensions_init,
                                                 CanonicalSystemDiscrete* canonical_system_discrete,
                                                 FuncApproximatorDiscrete* func_approximator_discrete,
                                                 std::vector<bool> is_using_scaling_init, uint ts_formulation_type,
                                                 LoggedDMPDiscreteVariables* logged_dmp_discrete_vars,
                                                 RealTimeAssertor* real_time_assertor,
                                                 DMPState* start_dmpstate_discrete, DMPState* current_dmpstate_discrete,
                                                 DMPState* current_velocity_dmpstate_discrete, GoalSystem* goal_system_discrete,
                                                 std::vector<TransformCoupling*>* transform_couplers):
    TransformationSystem(dmp_num_dimensions_init, canonical_system_discrete, func_approximator_discrete,
                         logged_dmp_discrete_vars, real_time_assertor,
                         start_dmpstate_discrete, current_dmpstate_discrete, current_velocity_dmpstate_discrete,
                         goal_system_discrete, transform_couplers),
    formulation_type(ts_formulation_type), alpha(25.0), beta(25.0/4.0), is_using_scaling(is_using_scaling_init),
    A_learn(OnesVectorN(MAX_DMP_NUM_DIMENSIONS)), logged_dmp_discrete_variables(logged_dmp_discrete_vars)
{
    A_learn.resize(dmp_num_dimensions);
}

/**
 * @param dmp_num_dimensions_init Number of DMP dimensions (N) to use
 * @param canonical_system_discrete Discrete canonical system that drives this discrete transformation system
 * @param func_approximator_discrete Discrete function approximator that produces forcing term for this discrete transformation system
 * @param is_using_scaling_init Is transformation system using scaling (true/false)?
 * @param ts_alpha Spring parameter\n
 *        Best practice: set it to 25
 * @param ts_beta Damping parameter\n
 *        Best practice: set it to alpha / 4 for a critically damped system
 * @param ts_formulation_type Chosen formulation\n
 *        _SCHAAL_DMP_: original formulation by Schaal, AMAM 2003 (default formulation_type)\n
 *        _HOFFMANN_DMP_: formulation by Hoffmann et al., ICRA 2009
 * @param logged_dmp_discrete_vars (Pointer to the) buffer for logged discrete DMP variables
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 * @param start_dmpstate_discrete (Pointer to the) start state of the discrete DMP's transformation system
 * @param current_dmpstate_discrete (Pointer to the) current state of the discrete DMP's transformation system
 * @param current_velocity_dmpstate_discrete (Pointer to the) current velocity state of the discrete DMP's transformation system
 * @param goal_system_discrete (Pointer to the) goal system for discrete DMP's transformation system
 * @param transform_couplers All spatial couplings associated with the transformation system
 * (please note that this is a pointer to vector of pointer to TransformCoupling data structure, for flexibility and re-usability)
 */
TransformSystemDiscrete::TransformSystemDiscrete(uint dmp_num_dimensions_init,
                                                 CanonicalSystemDiscrete* canonical_system_discrete,
                                                 FuncApproximatorDiscrete* func_approximator_discrete,
                                                 std::vector<bool> is_using_scaling_init,
                                                 double ts_alpha, double ts_beta, uint ts_formulation_type,
                                                 LoggedDMPDiscreteVariables* logged_dmp_discrete_vars,
                                                 RealTimeAssertor* real_time_assertor,
                                                 DMPState* start_dmpstate_discrete, DMPState* current_dmpstate_discrete,
                                                 DMPState* current_velocity_dmpstate_discrete, GoalSystem* goal_system_discrete,
                                                 std::vector<TransformCoupling*>* transform_couplers):
    TransformationSystem(dmp_num_dimensions_init, canonical_system_discrete,
                         func_approximator_discrete, logged_dmp_discrete_vars, real_time_assertor,
                         start_dmpstate_discrete, current_dmpstate_discrete, current_velocity_dmpstate_discrete,
                         goal_system_discrete, transform_couplers),
    formulation_type(ts_formulation_type), alpha(ts_alpha), beta(ts_beta), is_using_scaling(is_using_scaling_init),
    A_learn(OnesVectorN(MAX_DMP_NUM_DIMENSIONS)), logged_dmp_discrete_variables(logged_dmp_discrete_vars)
{
    A_learn.resize(dmp_num_dimensions);
}

/**
 * Checks whether this transformation system is valid or not.
 *
 * @return Transformation system is valid (true) or transformation system is invalid (false)
 */
bool TransformSystemDiscrete::isValid()
{
    if (rt_assert(TransformationSystem::isValid()) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(((CanonicalSystemDiscrete*) canonical_sys)->isValid())) &&
                  (rt_assert(((FuncApproximatorDiscrete*) func_approx)->isValid()))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(formulation_type >= _SCHAAL_DMP_)) &&
                  (rt_assert(formulation_type <= _HOFFMANN_DMP_))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(alpha > 0.0)) && (rt_assert(beta > 0.0))) == false)
    {
        return false;
    }
    if (rt_assert(is_using_scaling.size() == dmp_num_dimensions) == false)
    {
        return false;
    }
    if (rt_assert(A_learn.rows() == dmp_num_dimensions) == false)
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
bool TransformSystemDiscrete::start(const DMPState& start_state_init, const DMPState& goal_state_init)
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

    *start_state    = start_state_init;
    if (rt_assert(TransformSystemDiscrete::setCurrentState(start_state_init)) == false)
    {
        return false;
    }

    VectorN     G_init  = goal_state_init.getX();
    DMPState    current_goal_state_init(dmp_num_dimensions, rt_assertor);
    if ((formulation_type == _SCHAAL_DMP_) && (((CanonicalSystemDiscrete*) canonical_sys)->getOrder() == 2))
    {
        // Best option for Schaal's DMP Model using 2nd order canonical system:
        // Using goal evolution system initialized with the start position (state) as goal position (state),
        // which over time evolves toward a steady-state goal position (state).
        // The main purpose is to avoid discontinuous initial acceleration (see the math formula
        // of Schaal's DMP Model for more details).
        VectorN x0      = start_state_init.getX();
        VectorN xd0     = start_state_init.getXd();
        VectorN xdd0    = start_state_init.getXdd();

        double  tau;
        if (rt_assert(tau_sys->getTauRelative(tau)) == false)
        {
            return false;
        }

        VectorN g0  = ((((tau*tau*xdd0)/alpha) + (tau*xd0))/beta) + x0;

        current_goal_state_init.setX(g0);
    }
    else
    {
        // Best option for Schaal's DMP Model using 1st order canonical system:
        // goal position is static, no evolution
        current_goal_state_init     = goal_state_init;
    }

    if (rt_assert(goal_sys->start(current_goal_state_init, G_init)) == false)
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
bool TransformSystemDiscrete::getNextState(double dt, DMPState& next_state,
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
    if (rt_assert(TransformSystemDiscrete::isValid()) == false)
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
    ct_vel      = ZeroVectorN(dmp_num_dimensions);
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

    double      time= current_state->getTime();
    VectorN     x0  = start_state->getX();
    VectorN     x   = current_state->getX();
    VectorN     xd  = current_state->getXd();
    VectorN     xdd = current_state->getXdd();

    VectorN     v   = current_velocity_state->getX();
    VectorN     vd  = current_velocity_state->getXd();

    double      p   = canonical_sys->getCanonicalPosition();

    VectorN     G   = goal_sys->getSteadyStateGoalPosition();
    VectorN     g   = goal_sys->getCurrentGoalState().getX();

    x               = x + (xd * dt);

    if (formulation_type == _HOFFMANN_DMP_)
    {
        // formulation by Hoffmann et al., ICRA 2009
        double	K	= alpha * beta;
        double	D	= alpha;

        if (rt_assert(K >= MIN_K) == false) // K is too small/does NOT make sense (will have problem when computing f_target (target forcing term value))
        {
            return false;
        }

        vd          = ((K * ((g - x) - ((g - x0) * p) + (f + ct_acc))) - (D * v)) / tau;
    }
    else if (formulation_type == _SCHAAL_DMP_)
    {
        // compute scaling factor for the forcing term:
        VectorN     A(dmp_num_dimensions);
        A           = G - x0;

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
        vd    = ((alpha * ((beta * (g - x)) - v)) + ((f).asDiagonal() * A) + ct_acc) / tau;
    }

    if (rt_assert(containsNaN(vd) == false) == false)
    {
        return false;
    }

    xdd                     = vd / tau;
    xd                      = (v + ct_vel) / tau;

    v                       = v + (vd * dt);

    time                    = time + dt;

    *current_state          = DMPState(x, xd, xdd, time, rt_assertor);
    *current_velocity_state = DMPState(v, vd, ZeroVectorN(dmp_num_dimensions), time, rt_assertor);
    next_state              = *current_state;

    /**************** For Trajectory Data Logging Purposes (Start) ****************/
    logged_dmp_discrete_variables->tau                              = tau;
    logged_dmp_discrete_variables->transform_sys_state_local        = *current_state;
    logged_dmp_discrete_variables->canonical_sys_state[0]           = ((CanonicalSystemDiscrete*) canonical_sys)->getX();
    logged_dmp_discrete_variables->canonical_sys_state[1]           = ((CanonicalSystemDiscrete*) canonical_sys)->getXd();
    logged_dmp_discrete_variables->canonical_sys_state[2]           = ((CanonicalSystemDiscrete*) canonical_sys)->getXdd();
    logged_dmp_discrete_variables->goal_state_local                 = goal_sys->getCurrentGoalState();
    logged_dmp_discrete_variables->steady_state_goal_position_local = G;
    logged_dmp_discrete_variables->forcing_term                     = f;
    logged_dmp_discrete_variables->basis_functions                  = basis_functions;
    logged_dmp_discrete_variables->transform_sys_ct_acc             = ct_acc;
    /**************** For Trajectory Data Logging Purposes (End) ****************/

    return true;
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
bool TransformSystemDiscrete::getTargetForcingTerm(const DMPState& current_state_demo_local,
                                                   VectorN& f_target)
{
    // pre-conditions checking
    if (rt_assert(is_started == true) == false)
    {
        return false;
    }
    if (rt_assert(TransformSystemDiscrete::isValid()) == false)
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

    VectorN     x0_demo     = start_state->getX();
    VectorN     x_demo      = current_state_demo_local.getX();
    VectorN     xd_demo     = current_state_demo_local.getXd();
    VectorN     xdd_demo    = current_state_demo_local.getXdd();

    double      p_demo      = canonical_sys->getCanonicalPosition();

    VectorN     g_demo      = goal_sys->getCurrentGoalState().getX();

    if (formulation_type == _HOFFMANN_DMP_)
    {
        // formulation by Hoffmann et al., ICRA 2009
        double	K   = alpha * beta;
        double	D   = alpha;

        if (rt_assert(K >= MIN_K) == false) // K is too small/does NOT make sense
        {
            return false;
        }

        f_target    = (((tau_demo * tau_demo * xdd_demo) + (D * tau_demo * xd_demo)) / K)
                      - (g_demo - x_demo) + ((g_demo - x0_demo) * p_demo);
    }
    else if (formulation_type == _SCHAAL_DMP_)
    {
        // during learning, G = G_learn and x0 = x0_learn,
        // thus amplitude = ((G - x0)/(G_learn - x0_learn)) = ((G - x0)/A_learn) = 1.0,
        // as follows (commented out for computational effiency):
        //A_demo      = OnesVectorN(dmp_num_dimensions);

    	// original formulation by Schaal, AMAM 2003
        f_target    = ((tau_demo * tau_demo * xdd_demo) -
                       (alpha * ((beta * (g_demo - x_demo)) - (tau_demo * xd_demo))));
        /*for (uint n = 0; n < dmp_num_dimensions; ++n)
        {
            f_target[n] /= A_demo[n];
        }*/
    }

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
bool TransformSystemDiscrete::getTargetCouplingTerm(const DMPState& current_state_demo_local,
                                                    VectorN& ct_acc_target)
{
    // pre-conditions checking
    if (rt_assert(is_started == true) == false)
    {
        return false;
    }
    if (rt_assert(TransformSystemDiscrete::isValid()) == false)
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

    VectorN     x0_demo     = start_state->getX();
    VectorN     x_demo      = current_state_demo_local.getX();
    VectorN     xd_demo     = current_state_demo_local.getXd();
    VectorN     xdd_demo    = current_state_demo_local.getXdd();

    double      p_demo      = canonical_sys->getCanonicalPosition();

    VectorN     G_demo      = goal_sys->getSteadyStateGoalPosition();
    VectorN     g_demo      = goal_sys->getCurrentGoalState().getX();

    if (formulation_type == _HOFFMANN_DMP_)
    {
        // formulation by Hoffmann et al., ICRA 2009
        double	K   = alpha * beta;
        double	D   = alpha;

        if (rt_assert(K >= MIN_K) == false) // K is too small/does NOT make sense
        {
            return false;
        }

        ct_acc_target   = (((tau_demo * tau_demo * xdd_demo) + (D * tau_demo * xd_demo)) / K)
                          - (g_demo - x_demo) + ((g_demo - x0_demo) * p_demo) - f;
    }
    else if (formulation_type == _SCHAAL_DMP_)
    {
        // compute scaling factor for the forcing term:
        VectorN         A_demo(dmp_num_dimensions);
        A_demo          = G_demo - x0_demo;

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
        ct_acc_target   = ((tau_demo * tau_demo * xdd_demo) -
                           (alpha * ((beta * (g_demo - x_demo)) - (tau_demo * xd_demo))) -
                           ((f).asDiagonal() * A_demo));
    }

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
 * Set the scaling usage in each DMP dimension.
 *
 * @param Scaling usage (boolean) vector, each component corresponds to each DMP dimension
 */
bool TransformSystemDiscrete::setScalingUsage(const std::vector<bool>& is_using_scaling_init)
{
    // pre-conditions checking
    if (rt_assert(this->isValid()) == false)
    {
        return false;
    }
    // input checking
    if (rt_assert(is_using_scaling_init.size() == dmp_num_dimensions) == false)
    {
        return false;
    }

    is_using_scaling = is_using_scaling_init;

    return true;
}

/**
 * Get the learning amplitude (A_learn) = G_learn - y_0_learn in each DMP dimension.
 *
 * @return Current learning amplitude (A_learn)
 */
VectorN TransformSystemDiscrete::getLearningAmplitude()
{
    return A_learn;
}

/**
 * Set a new learning amplitude (A_learn) = G_learn - y_0_learn in each DMP dimension.
 * This will be called during learning from a demonstrated trajectory.
 *
 * @param new_A_learn New learning amplitude (A_learn)
 * @return Success or failure
 */
bool TransformSystemDiscrete::setLearningAmplitude(const VectorN& new_A_learn)
{
    // pre-conditions checking
    if (rt_assert(this->isValid()) == false)
    {
        return false;
    }
    // input checking
    if (rt_assert(new_A_learn.rows() == dmp_num_dimensions) == false)
    {
        return false;
    }

    A_learn = new_A_learn;

    return true;
}

/**
 * @return The chosen formulation for the discrete transformation system\n
 *         0: original formulation by Schaal, AMAM 2003 (default formulation_type)\n
 *         1: formulation by Hoffman et al., ICRA 2009
 */
uint TransformSystemDiscrete::getFormulationType()
{
    return (formulation_type);
}

TransformSystemDiscrete::~TransformSystemDiscrete()
{}

}
