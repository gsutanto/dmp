#include "amd_clmc_dmp/dmp_base/TransformationSystem.h"

namespace dmp
{

    TransformationSystem::TransformationSystem():
        dmp_num_dimensions(0), is_started(false), is_using_coupling_term_at_dimension(std::vector<bool>()),
        tau_sys(NULL), canonical_sys(NULL), func_approx(NULL), transform_coupling(NULL),
        start_state(NULL), current_state(NULL), current_velocity_state(NULL),
        goal_sys(NULL), logged_dmp_variables(NULL), rt_assertor(NULL),
        is_start_state_obj_created_here(false), is_current_state_obj_created_here(false),
        is_current_velocity_state_obj_created_here(false), is_goal_sys_obj_created_here(false)
    {}

    /**
     * @param dmp_num_dimensions_init Number of DMP dimensions (N) to use
     * @param tau_system Tau system that computes the tau parameter for this transformation system
     * @param canonical_system Canonical system that drives this transformation system
     * @param function_approximator Function approximator that produces forcing term for this transformation system
     * @param logged_dmp_vars (Pointer to the) buffer for logged DMP variables
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     * @param start_dmpstate (Pointer to the) start state of the DMP's transformation system
     * @param current_dmpstate (Pointer to the) current state of the DMP's transformation system
     * @param current_velocity_dmpstate (Pointer to the) current velocity state of the DMP's transformation system
     * @param goal_system (Pointer to the) goal system for DMP's transformation system
     * @param transform_couplers All spatial couplings associated with the transformation system
     * (please note that this is a pointer to vector of pointer to TransformCoupling data structure, for flexibility and re-usability)
     */
    TransformationSystem::TransformationSystem(uint dmp_num_dimensions_init, CanonicalSystem* canonical_system,
                                               FunctionApproximator* function_approximator,
                                               LoggedDMPVariables* logged_dmp_vars,
                                               RealTimeAssertor* real_time_assertor,
                                               DMPState* start_dmpstate, DMPState* current_dmpstate,
                                               DMPState* current_velocity_dmpstate, GoalSystem* goal_system,
                                               std::vector<TransformCoupling*>* transform_couplers):
        dmp_num_dimensions(dmp_num_dimensions_init), is_started(false),
        is_using_coupling_term_at_dimension(std::vector<bool>(dmp_num_dimensions_init, true)),
        tau_sys(canonical_system->getTauSystemPointer()),
        canonical_sys(canonical_system), func_approx(function_approximator),
        logged_dmp_variables(logged_dmp_vars), rt_assertor(real_time_assertor),
        transform_coupling(transform_couplers)
    {
        if (start_dmpstate == NULL)
        {
            start_state = new DMPState(dmp_num_dimensions_init, real_time_assertor);
            is_start_state_obj_created_here = true;
        }
        else
        {
            start_state = start_dmpstate;
            is_start_state_obj_created_here = false;
        }

        if (current_dmpstate == NULL)
        {
            current_state   = new DMPState(dmp_num_dimensions_init, real_time_assertor);
            is_current_state_obj_created_here   = true;
        }
        else
        {
            current_state   = current_dmpstate;
            is_current_state_obj_created_here   = false;
        }

        if (current_velocity_dmpstate == NULL)
        {
            current_velocity_state  = new DMPState(dmp_num_dimensions_init, real_time_assertor);
            is_current_velocity_state_obj_created_here  = true;
        }
        else
        {
            current_velocity_state  = current_velocity_dmpstate;
            is_current_velocity_state_obj_created_here  = false;
        }

        if (goal_system == NULL)
        {
            goal_sys    = new GoalSystem(dmp_num_dimensions_init, canonical_system->getTauSystemPointer(),
                                         real_time_assertor);
            is_goal_sys_obj_created_here    = true;
        }
        else
        {
            goal_sys    = goal_system;
            is_goal_sys_obj_created_here    = false;
        }
    }

    /**
     * Checks whether the transformation system is valid or not.
     *
     * @return Transformation system is valid (true) or transformation system is invalid (false)
     */
    bool TransformationSystem::isValid()
    {
        if (rt_assert((rt_assert(dmp_num_dimensions > 0)) &&
                      (rt_assert(dmp_num_dimensions <= MAX_DMP_NUM_DIMENSIONS)) &&
                      (rt_assert(is_using_coupling_term_at_dimension.size() == dmp_num_dimensions))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(tau_sys        != NULL)) &&
                      (rt_assert(canonical_sys  != NULL)) &&
                      (rt_assert(func_approx    != NULL)) &&
                      (rt_assert(goal_sys       != NULL))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(tau_sys->isValid())) &&
                      (rt_assert(canonical_sys->isValid())) &&
                      (rt_assert(func_approx->isValid())) &&
                      (rt_assert(goal_sys->isValid()))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(tau_sys == canonical_sys->getTauSystemPointer())) &&
                      (rt_assert(tau_sys == goal_sys->getTauSystemPointer()))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(start_state            != NULL)) &&
                      (rt_assert(current_state          != NULL)) &&
                      (rt_assert(current_velocity_state != NULL))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(start_state->isValid()))    &&
                      (rt_assert(current_state->isValid()))  &&
                      (rt_assert(current_velocity_state->isValid()))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(func_approx->getDMPNumDimensions()             == dmp_num_dimensions)) &&
                      (rt_assert(goal_sys->getDMPNumDimensions()                == dmp_num_dimensions)) &&
                      (rt_assert(start_state->getDMPNumDimensions()             == dmp_num_dimensions)) &&
                      (rt_assert(current_state->getDMPNumDimensions()           == dmp_num_dimensions)) &&
                      (rt_assert(current_velocity_state->getDMPNumDimensions()  == dmp_num_dimensions))) == false)
        {
            return false;
        }
        return true;
    }

    /**
     * If there are coupling terms on the transformation system,
     * this function will iterate over each coupling term, and returns the accumulated coupling term.
     * Otherwise it will return a zero VectorN.
     *
     * @param accumulated_ct_acc The accumulated transformation system coupling term value (to be returned)
     * @param accumulated_ct_vel The accumulated velocity-level transformation system coupling term value (to be returned)
     * @return Success or failure
     */
    bool TransformationSystem::getCouplingTerm(VectorN& accumulated_ct_acc,
                                               VectorN& accumulated_ct_vel)
    {
        // input checking
        if (rt_assert(accumulated_ct_acc.rows() == dmp_num_dimensions) == false)
        {
            return false;
        }

        accumulated_ct_acc          = ZeroVectorN(dmp_num_dimensions);

        if (transform_coupling)
        {
            for (uint i=0; i<(*transform_coupling).size(); ++i)
            {
                if ((*transform_coupling)[i] == NULL)
                {
                    continue;
                }

                VectorN             ct_acc(dmp_num_dimensions);
                VectorN             ct_vel(dmp_num_dimensions);
                ct_acc              = ZeroVectorN(dmp_num_dimensions);
                ct_vel              = ZeroVectorN(dmp_num_dimensions);
                if (rt_assert((*transform_coupling)[i]->getValue(ct_acc, ct_vel)) == false)
                {
                    return false;
                }
                if (rt_assert((containsNaN(ct_acc) || containsNaN(ct_vel)) == false) == false)
                {
                    return false;
                }

                accumulated_ct_acc  += ct_acc;
                accumulated_ct_vel  += ct_vel;
            }
        }

        return true;
    }

    /**
     * Returns the number of dimensions in the DMP.
     *
     * @return Number of dimensions in the DMP
     */
    uint TransformationSystem::getDMPNumDimensions()
    {
        return dmp_num_dimensions;
    }

    /**
     * Returns transformation system's start state.
     *
     * @return Transformation system's start state (DMPState)
     */
    DMPState TransformationSystem::getStartState()
    {
        return (*start_state);
    }

    /**
     * Sets new start state.
     *
     * @param new_start_state New start state to be set
     * @return Success or failure
     */
    bool TransformationSystem::setStartState(const DMPState& new_start_state)
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
        if (rt_assert(start_state->getDMPNumDimensions() == new_start_state.getDMPNumDimensions()) == false)
        {
            return false;
        }

        *start_state    = new_start_state;

        // post-condition(s) checking
        return (rt_assert(this->isValid()));
    }

    /**
     * Returns current transformation system state.
     *
     * @return Current transformation system state (DMPState)
     */
    DMPState TransformationSystem::getCurrentState()
    {
        return (*current_state);
    }

    /**
     * Sets new current state.
     *
     * @param new_current_state New current state to be set
     * @return Success or failure
     */
    bool TransformationSystem::setCurrentState(const DMPState& new_current_state)
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
        if (rt_assert(current_state->getDMPNumDimensions() == new_current_state.getDMPNumDimensions()) == false)
        {
            return false;
        }

        *current_state  = new_current_state;

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
    bool TransformationSystem::updateCurrentVelocityStateFromCurrentState()
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

        VectorN xd  = current_state->getXd();
        VectorN xdd = current_state->getXdd();

        VectorN v   = tau * xd;
        VectorN vd  = tau * xdd;

        if (rt_assert(current_velocity_state->setX(v)) == false)
        {
            return false;
        }
        if (rt_assert(current_velocity_state->setXd(vd)) == false)
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
    DMPState TransformationSystem::getCurrentGoalState()
    {
        return (goal_sys->getCurrentGoalState());
    }

    /**
     * Sets new current goal state.
     *
     * @param new_current_goal_state New current goal state to be set
     * @return Success or failure
     */
    bool TransformationSystem::setCurrentGoalState(const DMPState& new_current_goal_state)
    {
        return (rt_assert(goal_sys->setCurrentGoalState(new_current_goal_state)));
    }

    /**
     * Runs one time-step ahead of the goal state system and updates the current goal state vector
     * based on the time difference dt from the previous execution.
     *
     * @param dt Time difference between the current and the previous execution
     * @return Success or failure
     */
    bool TransformationSystem::updateCurrentGoalState(const double& dt)
    {
        return (rt_assert(goal_sys->updateCurrentGoalState(dt)));
    }

    /**
     * Returns steady-state goal position vector (G).
     *
     * @return Steady-state goal position vector (G)
     */
    VectorN TransformationSystem::getSteadyStateGoalPosition()
    {
        return (goal_sys->getSteadyStateGoalPosition());
    }

    /**
     * Sets new steady-state goal position vector (G).
     *
     * @param new_G New steady-state goal position vector (G) to be set
     * @return Success or failure
     */
    bool TransformationSystem::setSteadyStateGoalPosition(const VectorN& new_G)
    {
        return (rt_assert(goal_sys->setSteadyStateGoalPosition(new_G)));
    }

    /**
     * Returns the tau system pointer.
     */
    TauSystem* TransformationSystem::getTauSystemPointer()
    {
        return tau_sys;
    }

    /**
     * Returns the canonical system pointer.
     */
    CanonicalSystem* TransformationSystem::getCanonicalSystemPointer()
    {
        return canonical_sys;
    }

    /**
     * Returns the function approximator pointer.
     */
    FunctionApproximator* TransformationSystem::getFuncApproxPointer()
    {
        return func_approx;
    }

    bool TransformationSystem::setCouplingTermUsagePerDimensions(const std::vector<bool>& is_using_coupling_term_at_dimension_init)
    {
        if (rt_assert(is_using_coupling_term_at_dimension_init.size() == dmp_num_dimensions) == false)
        {
            return false;
        }

        is_using_coupling_term_at_dimension     = is_using_coupling_term_at_dimension_init;

        return true;
    }

    TransformationSystem::~TransformationSystem()
    {
        if (is_start_state_obj_created_here == true)
        {
            delete start_state;
        }

        if (is_current_state_obj_created_here == true)
        {
            delete current_state;
        }

        if (is_current_velocity_state_obj_created_here == true)
        {
            delete current_velocity_state;
        }

        if (is_goal_sys_obj_created_here == true)
        {
            delete goal_sys;
        }
    }

}