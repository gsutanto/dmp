#include "dmp/dmp_goal_system/GoalSystem.h"

namespace dmp
{

GoalSystem::GoalSystem():
    dmp_num_dimensions(0), alpha(25.0/2.0), current_goal_state(NULL),
    G(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)), is_started(false), tau_sys(NULL), rt_assertor(NULL),
    is_current_goal_state_obj_created_here(false)
{
    G.resize(0);
}

/**
 * @param dmp_num_dimensions_init Initialization of the number of dimensions that will be used in the DMP
 * @param tau_system Tau system that computes the tau parameter for this goal-changing system
 * @param alpha_init Spring parameter (default value = 25/2)
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 */
GoalSystem::GoalSystem(uint dmp_num_dimensions_init, TauSystem* tau_system,
                       RealTimeAssertor* real_time_assertor,
                       double alpha_init):
    dmp_num_dimensions(dmp_num_dimensions_init), current_goal_state(new DMPState(dmp_num_dimensions_init, real_time_assertor)),
    is_current_goal_state_obj_created_here(true), G(ZeroVectorN(dmp_num_dimensions_init)), is_started(false),
    tau_sys(tau_system),  alpha(alpha_init), rt_assertor(real_time_assertor)
{}

/**
 * @param dmp_num_dimensions_init Initialization of the number of dimensions that will be used in the DMP
 * @param current_goal_state_ptr Pointer to DMPState that will play the role of current_goal_state
 * @param goal_num_dimensions_init Initialization of the number of dimensions of the goal position
 * @param tau_system Tau system that computes the tau parameter for this goal-changing system
 * @param alpha_init Spring parameter (default value = 25/2)
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 */
GoalSystem::GoalSystem(uint dmp_num_dimensions_init, DMPState* current_goal_state_ptr, uint goal_num_dimensions_init,
                       TauSystem* tau_system, RealTimeAssertor* real_time_assertor,
                       double alpha_init):
    dmp_num_dimensions(dmp_num_dimensions_init), current_goal_state(current_goal_state_ptr),
    is_current_goal_state_obj_created_here(false), G(ZeroVectorN(goal_num_dimensions_init)), is_started(false),
    tau_sys(tau_system),  alpha(alpha_init), rt_assertor(real_time_assertor)
{}

/**
 * Checks whether this tau system is valid or not.
 *
 * @return Tau system is valid (true) or tau system is invalid (false)
 */
bool GoalSystem::isValid()
{
    if (rt_assert((rt_assert(dmp_num_dimensions > 0)) &&
                  (rt_assert(dmp_num_dimensions <= MAX_DMP_NUM_DIMENSIONS))) == false)
    {
        return false;
    }
    if (rt_assert(current_goal_state != NULL) == false)
    {
        return false;
    }
    if (rt_assert(current_goal_state->isValid()) == false)
    {
        return false;
    }
    if (rt_assert(current_goal_state->getDMPNumDimensions() == dmp_num_dimensions) == false)
    {
        return false;
    }
    if (rt_assert(G.rows() == current_goal_state->getX().rows()) == false)
    {
        return false;
    }
    if (rt_assert(tau_sys != NULL) == false)
    {
        return false;
    }
    if (rt_assert(tau_sys->isValid()) == false)
    {
        return false;
    }
    if (rt_assert(alpha > 0.0) == false)
    {
        return false;
    }
    return true;
}

/**
 * Start the goal (evolution) system, by setting the states properly.
 *
 * @param current_goal_state_init Initialization value for the current goal state
 * @param G_init Initialization value for steady-state goal position vector (G)
 * @return Success or failure
 */
bool GoalSystem::start(const DMPState& current_goal_state_init, const VectorN& G_init)
{
    // pre-condition(s) checking
    if (rt_assert(this->isValid()) == false)
    {
        return false;
    }

    if (rt_assert(this->setCurrentGoalState(current_goal_state_init)) == false)
    {
        return false;
    }
    if (rt_assert(this->setSteadyStateGoalPosition(G_init)) == false)
    {
        return false;
    }

    is_started  = true;

    // post-condition(s) checking
    return (rt_assert(this->isValid()));
}

/**
 * Returns the number of dimensions in the DMP.
 *
 * @return Number of dimensions in the DMP
 */
uint GoalSystem::getDMPNumDimensions()
{
    return dmp_num_dimensions;
}

/**
 * Returns steady-state goal position vector (G).
 *
 * @return Steady-state goal position vector (G)
 */
VectorN GoalSystem::getSteadyStateGoalPosition()
{
    return G;
}

/**
 * Sets new steady-state goal position vector (G).
 *
 * @param new_G New steady-state goal position vector (G) to be set
 * @return Success or failure
 */
bool GoalSystem::setSteadyStateGoalPosition(const VectorN& new_G)
{
    // pre-condition(s) checking
    if (rt_assert(this->isValid()) == false)
    {
        return false;
    }

    if (rt_assert(new_G.rows() == G.rows()))
    {
        G       = new_G;
    }
    else
    {
        return false;
    }

    // post-condition(s) checking
    return (rt_assert(this->isValid()));
}

/**
 * Returns the current transient/non-steady-state goal state vector.
 *
 * @return The current transient/non-steady-state goal state vector
 */
DMPState GoalSystem::getCurrentGoalState()
{
    return (*current_goal_state);
}

/**
 * Sets new current (evolving) goal state.
 *
 * @param new_current_goal_state New current (evolving) goal state to be set
 * @return Success or failure
 */
bool GoalSystem::setCurrentGoalState(const DMPState& new_current_goal_state)
{
    // pre-condition(s) checking
    if (rt_assert(this->isValid()) == false)
    {
        return false;
    }
    // input checking
    if (rt_assert(new_current_goal_state.isValid()) == false)
    {
        return false;
    }
    if (rt_assert(new_current_goal_state.getDMPNumDimensions() == dmp_num_dimensions) &&
        rt_assert(new_current_goal_state.getX().rows() == current_goal_state->getX().rows()))
    {
        *current_goal_state = new_current_goal_state;
    }
    else
    {
        return false;
    }

    // post-condition(s) checking
    return (rt_assert(this->isValid()));
}

/**
 * Runs one time-step ahead of the goal state system and updates the current goal state vector
 * based on the time difference dt from the previous execution.
 *
 * @param dt Time difference between the current and the previous execution
 * @return Success or failure
 */
bool GoalSystem::updateCurrentGoalState(const double& dt)
{
    // pre-condition(s) checking
    if (rt_assert(is_started == true) == false)
    {
        return false;
    }
    if (rt_assert(this->isValid()) == false)
    {
        return false;
    }
    // input checking
    if (rt_assert(dt > 0.0) == false)   // dt does NOT make sense
    {
        return false;
    }

    double  tau;
    if (rt_assert(tau_sys->getTauRelative(tau)) == false)
    {
        return false;
    }

    VectorN g   = current_goal_state->getX();
    VectorN gd  = alpha * (G - g) / tau;
    g           = g + (gd * dt);

    if (rt_assert(current_goal_state->setX(g)) == false)
    {
        return false;
    }
    if (rt_assert(current_goal_state->setXd(gd)) == false)
    {
        return false;
    }
    if (rt_assert(current_goal_state->setTime(current_goal_state->getTime() + dt)) == false)
    {
        return false;
    }

    // post-condition(s) checking
    return (rt_assert(this->isValid()));
}

/**
 * Returns the tau system pointer.
 */
TauSystem* GoalSystem::getTauSystemPointer()
{
    return tau_sys;
}

GoalSystem::~GoalSystem()
{
    if (is_current_goal_state_obj_created_here == true)
    {
        delete current_goal_state;
    }
}

}
