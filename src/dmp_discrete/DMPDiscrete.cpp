#include "amd_clmc_dmp/dmp_discrete/DMPDiscrete.h"

namespace dmp
{

DMPDiscrete::DMPDiscrete():
    DMP(), canonical_sys_discrete(NULL), func_approx_discrete(FuncApproximatorDiscrete()),
    transform_sys_discrete(NULL),
    learning_sys_discrete(LearningSystemDiscrete()),
    data_logger_loader_discrete(DMPDataIODiscrete()),
    logged_dmp_discrete_variables(LoggedDMPDiscreteVariables())
{}

/**
 * @param dmp_num_dimensions_init Number of trajectory dimensions employed in this DMP
 * @param model_size_init Size of the model (M: number of basis functions or others) used to represent the function
 * @param canonical_system_discrete (Pointer to) discrete canonical system that drives this DMP
 * @param transform_system_discrete (Pointer to) discrete transformation system that drives this DMP
 * @param learning_method Learning method that will be used to learn the DMP basis functions' weights\n
 *        0=_SCHAAL_LWR_METHOD_: Stefan Schaal's Locally-Weighted Regression (LWR): Ijspeert et al., Neural Comput. 25, 2 (Feb 2013), p328-373\n
 *        1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_: Jan Peters' Global Least Squares method
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 * @param opt_data_directory_path [optional] Full directory path where to save/log all the data
 */
DMPDiscrete::DMPDiscrete(uint dmp_num_dimensions_init, uint model_size_init,
                         CanonicalSystemDiscrete* canonical_system_discrete,
                         TransformSystemDiscrete* transform_system_discrete,
                         uint learning_method, RealTimeAssertor* real_time_assertor,
                         const char* opt_data_directory_path):
    DMP(dmp_num_dimensions_init, model_size_init, canonical_system_discrete, &func_approx_discrete,
        transform_system_discrete, &learning_sys_discrete, &data_logger_loader_discrete,
        &logged_dmp_discrete_variables, real_time_assertor, opt_data_directory_path),
    canonical_sys_discrete(canonical_system_discrete),
    transform_sys_discrete(transform_system_discrete),
    func_approx_discrete(FuncApproximatorDiscrete(dmp_num_dimensions_init, model_size_init,
                                                  canonical_system_discrete, real_time_assertor)),
    learning_sys_discrete(LearningSystemDiscrete(dmp_num_dimensions_init, model_size_init,
                                                 transform_system_discrete,
                                                 &data_logger_loader_discrete, learning_method,
                                                 real_time_assertor, opt_data_directory_path)),
    data_logger_loader_discrete(DMPDataIODiscrete(transform_system_discrete, &func_approx_discrete,
                                                  real_time_assertor)),
    logged_dmp_discrete_variables(LoggedDMPDiscreteVariables(dmp_num_dimensions_init, model_size_init,
                                                             real_time_assertor))
{}

/**
 * Checks whether this Discrete DMP is valid or not.
 *
 * @return Discrete DMP is valid (true) or Discrete DMP is invalid (false)
 */
bool DMPDiscrete::isValid()
{
    if (rt_assert(DMP::isValid()) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(canonical_sys_discrete != NULL)) &&
                  (rt_assert(transform_sys_discrete != NULL))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(canonical_sys_discrete->isValid())) &&
                  (rt_assert(transform_sys_discrete->isValid())) &&
                  (rt_assert(func_approx_discrete.isValid())) &&
                  (rt_assert(learning_sys_discrete.isValid())) &&
                  (rt_assert(data_logger_loader_discrete.isValid()))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(transform_sys_discrete == data_logger_loader_discrete.getTransformSysDiscretePointer())) &&
                  (rt_assert(&func_approx_discrete   == data_logger_loader_discrete.getFuncApproxDiscretePointer()))) == false)
    {
        return false;
    }
    return true;
}

/**
 * NON-REAL-TIME!!!\n
 * Learns from a given set of trajectories.
 *
 * @param trajectory_set Set of trajectories to learn from\n
 *        (assuming all trajectories in this set are coming from the same trajectory distribution)
 * @param robot_task_servo_rate Robot's task servo rate
 * @return Success or failure
 */
bool DMPDiscrete::learn(const TrajectorySet& trajectory_set, double robot_task_servo_rate)
{
    bool is_real_time   = false;

    // pre-conditions checking
    if (rt_assert(DMPDiscrete::isValid()) == false)
    {
        return false;
    }
    // input checking:
    if (rt_assert((rt_assert(robot_task_servo_rate > 0.0)) &&
                  (rt_assert(robot_task_servo_rate <= MAX_TASK_SERVO_RATE))) == false)
    {
        return false;
    }

    TrajectorySetPtr    trajectory_set_preprocessed;
    if (rt_assert(allocateMemoryIfNonRealTime(is_real_time,
                                              trajectory_set_preprocessed)) == false)
    {
        return (-1);
    }
    (*trajectory_set_preprocessed)  = trajectory_set;

    // pre-process trajectories
    if (rt_assert(DMPDiscrete::preprocess(*trajectory_set_preprocessed)) == false)
    {
        return false;
    }

    if (rt_assert(learning_sys_discrete.learnApproximator(*trajectory_set_preprocessed,
                                                          robot_task_servo_rate,
                                                          mean_tau)) == false)
    {
        return false;
    }

    if (rt_assert(mean_tau >= MIN_TAU) == false)
    {
        return false;
    }

    return true;
}

/**
 * Resets DMP and sets all needed parameters (including the reference coordinate frame) to start execution.
 * Call this function before beginning any DMP execution.
 *
 * @param critical_states Critical states defined for DMP trajectory reproduction/unrolling.\n
 *                        Start state is the first state in this trajectory/vector of DMPStates.\n
 *                        Goal state is the last state in this trajectory/vector of DMPStates.
 * @param tau_init Initialization for tau (time parameter that influences the speed of the execution; usually this is the time-length of the trajectory)
 * @return Success or failure
 */
bool DMPDiscrete::start(const Trajectory& critical_states, double tau_init)
{
    // pre-conditions checking
    if (rt_assert(this->isValid()) == false)
    {
        return false;
    }
    // input checking
    uint traj_size      = critical_states.size();
    if (rt_assert((rt_assert(traj_size >= 2)) && (rt_assert(traj_size <= MAX_TRAJ_SIZE))) == false)
    {
        return false;
    }
    DMPState    start_state_init    = critical_states[0];
    DMPState    goal_state_init     = critical_states[traj_size-1];
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
    if (rt_assert(tau_init >= MIN_TAU) == false)    // tau_init is too small/does NOT make sense
    {
        return false;
    }

    if (rt_assert(tau_sys->setTauBase(tau_init)) == false)
    {
        return false;
    }
    if (rt_assert(canonical_sys_discrete->start()) == false)
    {
        return false;
    }
    if (rt_assert(transform_sys_discrete->start(start_state_init, goal_state_init)) == false)
    {
        return false;
    }

    is_started  = true;

    // post-conditions checking
    return (rt_assert(this->isValid()));
}

/**
 * Runs one step of the DMP based on the time step dt.
 *
 * @param dt Time difference between the previous and the current step
 * @param update_canonical_state Set to true if the canonical state should be updated here.\n
 *        Otherwise you are responsible for updating it yourself.
 * @param next_state Computed next DMPState
 * @param transform_sys_forcing_term (optional) If you also want to know \n
 *                                   the transformation system's forcing term value of the next state, \n
 *                                   please put the pointer here
 * @param transform_sys_coupling_term_acc (optional) If you also want to know \n
 *                                        the acceleration-level coupling term value \n
 *                                        of the next state, please put the pointer here
 * @param transform_sys_coupling_term_vel (optional) If you also want to know \n
 *                                        the velocity-level coupling term value \n
 *                                        of the next state, please put the pointer here
 * @param func_approx_basis_functions (optional) (Pointer to) vector of basis functions constructing the forcing term
 * @param func_approx_normalized_basis_func_vector_mult_phase_multiplier (optional) If also interested in \n
 *              the <normalized basis function vector multiplied by the canonical phase multiplier (phase position or phase velocity)> value, \n
 *              put its pointer here (return variable)
 * @return Success or failure
 */
bool DMPDiscrete::getNextState(double dt, bool update_canonical_state,
                               DMPState& next_state, VectorN* transform_sys_forcing_term,
                               VectorN* transform_sys_coupling_term_acc,
                               VectorN* transform_sys_coupling_term_vel,
                               VectorM* func_approx_basis_functions,
                               VectorM* func_approx_normalized_basis_func_vector_mult_phase_multiplier)
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

    DMPState    result_dmp_state(dmp_num_dimensions, rt_assertor);
    if (rt_assert(transform_sys_discrete->getNextState(dt, result_dmp_state,
                                                       transform_sys_forcing_term,
                                                       transform_sys_coupling_term_acc,
                                                       transform_sys_coupling_term_vel,
                                                       func_approx_basis_functions,
                                                       func_approx_normalized_basis_func_vector_mult_phase_multiplier)) == false)
    {
        return false;
    }

    if (update_canonical_state)
    {
        if (rt_assert(canonical_sys_discrete->updateCanonicalState(dt)) == false)
        {
            return false;
        }
    }

    if (rt_assert(transform_sys_discrete->updateCurrentGoalState(dt)) == false)
    {
        return false;
    }

    next_state  = result_dmp_state;

    /**************** For Trajectory Data Logging Purposes (Start) ****************/
    logged_dmp_discrete_variables.transform_sys_state_global        = logged_dmp_discrete_variables.transform_sys_state_local;
    logged_dmp_discrete_variables.goal_state_global                 = logged_dmp_discrete_variables.goal_state_local;
    logged_dmp_discrete_variables.steady_state_goal_position_global = logged_dmp_discrete_variables.steady_state_goal_position_local;

    if (is_collecting_trajectory_data)
    {
        if (rt_assert(data_logger_loader_discrete.collectTrajectoryDataSet(logged_dmp_discrete_variables)) == false)
        {
            return false;
        }
    }
    /**************** For Trajectory Data Logging Purposes (End) ****************/
    // post-conditions checking
    return (rt_assert(this->isValid()));
}

/**
 * Returns current DMP state.
 */
DMPState DMPDiscrete::getCurrentState()
{
    return (transform_sys_discrete->getCurrentState());
}

/**
 * Returns the value of the approximated function at current canonical position and \n
 * current canonical multiplier, for each DMP dimensions (as a vector).
 *
 * @param result_f Computed forcing term vector at current canonical position and \n
 *                 current canonical multiplier, each vector component for each DMP dimension (return variable)
 * @return Success or failure
 */
bool DMPDiscrete::getForcingTerm(VectorN& result_f, VectorM* basis_function_vector)
{
    return (rt_assert(func_approx_discrete.getForcingTerm(result_f, basis_function_vector)));
}

/**
 * Returns current goal position.
 *
 * @param current_goal Current goal position (to be returned)
 * @return Success or failure
 */
bool DMPDiscrete::getCurrentGoalPosition(VectorN& current_goal)
{
    // pre-conditions checking
    if (rt_assert(DMPDiscrete::isValid()) == false)
    {
        return false;
    }

    current_goal    = transform_sys_discrete->getCurrentGoalState().getX();
    return true;
}

/**
 * Returns steady-state goal position vector (G).
 *
 * @param Steady-state goal position vector (G) (to be returned)
 * @return Success or failure
 */
bool DMPDiscrete::getSteadyStateGoalPosition(VectorN& steady_state_goal)
{
    // pre-conditions checking
    if (rt_assert(DMPDiscrete::isValid()) == false)
    {
        return false;
    }

    steady_state_goal   = transform_sys_discrete->getSteadyStateGoalPosition();
    return true;
}

/**
 * Sets new steady-state goal position (G).
 *
 * @param new_G New steady-state goal position (G) to be set
 * @return Success or failure
 */
bool DMPDiscrete::setNewSteadyStateGoalPosition(const VectorN& new_G)
{
    // pre-conditions checking
    if (rt_assert(DMPDiscrete::isValid()) == false)
    {
        return false;
    }

    return (rt_assert(transform_sys_discrete->setSteadyStateGoalPosition(new_G)));
}

/**
 * Set the scaling usage in each DMP dimension.
 *
 * @param Scaling usage (boolean) vector, each component corresponds to each DMP dimension
 */
bool DMPDiscrete::setScalingUsage(const std::vector<bool>& is_using_scaling_init)
{
    return (rt_assert(transform_sys_discrete->setScalingUsage(is_using_scaling_init)));
}

/**
 * Get the current parameters of this discrete DMP (weights and learning amplitude (A_learn))
 *
 * @param weights_buffer Weights buffer, to which the current weights will be copied onto (to be returned)
 * @param A_learn_buffer A_learn vector buffer, to which the current learning amplitude will be copied onto (to be returned)
 * @return Success or failure
 */
bool DMPDiscrete::getParams(MatrixNxM& weights_buffer, VectorN& A_learn_buffer)
{
    A_learn_buffer  = transform_sys_discrete->getLearningAmplitude();

    return (rt_assert(DMPDiscrete::getWeights(weights_buffer)));
}

/**
 * Set the new parameters of this discrete DMP (weights and learning amplitude (A_learn))
 *
 * @param new_weights New weights to be set
 * @param new_A_learn New learning amplitude (A_learn) to be set
 * @return Success or failure
 */
bool DMPDiscrete::setParams(const MatrixNxM& new_weights, const VectorN& new_A_learn)
{
    if (rt_assert(transform_sys_discrete->setLearningAmplitude(new_A_learn)) == false)
    {
        return false;
    }

    return (rt_assert(DMPDiscrete::setWeights(new_weights)));
}

/**
 * NEVER RUN IN REAL-TIME\n
 * Save the current parameters of this discrete DMP (weights and learning amplitude (A_learn)) into files.
 *
 * @param dir_path Directory path containing the files
 * @param file_name_weights Name of the file onto which the weights will be saved
 * @param file_name_A_learn Name of the file onto which the A_learn vector buffer will be saved
 * @param file_name_mean_start_position Name of the file onto which the mean start position will be saved
 * @param file_name_mean_goal_position Name of the file onto which the mean goal position will be saved
 * @param file_name_mean_tau Name of the file onto which the mean tau will be saved
 * @return Success or failure
 */
bool DMPDiscrete::saveParams(const char* dir_path,
                             const char* file_name_weights,
                             const char* file_name_A_learn,
                             const char* file_name_mean_start_position,
                             const char* file_name_mean_goal_position,
                             const char* file_name_mean_tau)
{
    return (rt_assert(data_logger_loader_discrete.saveWeights(dir_path, file_name_weights)) &&
            rt_assert(data_logger_loader_discrete.saveALearn(dir_path, file_name_A_learn))  &&
            rt_assert(data_logger_loader_discrete.writeMatrixToFile(dir_path, file_name_mean_start_position, mean_start_position)) &&
            rt_assert(data_logger_loader_discrete.writeMatrixToFile(dir_path, file_name_mean_goal_position, mean_goal_position))   &&
            rt_assert(data_logger_loader_discrete.writeMatrixToFile(dir_path, file_name_mean_tau, mean_tau)));
}

/**
 * NEVER RUN IN REAL-TIME\n
 * Load the new parameters of this discrete DMP (weights and learning amplitude (A_learn)) from files
 *
 * @param dir_path Directory path containing the files
 * @param file_name_weights Name of the file containing the new weights to be set
 * @param file_name_A_learn Name of the file containing the new learning amplitude (A_learn) to be set
 * @param file_name_mean_start_position Name of the file containing the new mean start position to be set
 * @param file_name_mean_goal_position Name of the file containing the new mean goal position to be set
 * @param file_name_mean_tau Name of the file containing the new mean tau to be set
 * @return Success or failure
 */
bool DMPDiscrete::loadParams(const char* dir_path,
                             const char* file_name_weights,
                             const char* file_name_A_learn,
                             const char* file_name_mean_start_position,
                             const char* file_name_mean_goal_position,
                             const char* file_name_mean_tau)
{
    return (rt_assert(data_logger_loader_discrete.loadWeights(dir_path, file_name_weights)) &&
            rt_assert(data_logger_loader_discrete.loadALearn(dir_path, file_name_A_learn))  &&
            rt_assert(data_logger_loader_discrete.readMatrixFromFile(dir_path, file_name_mean_start_position, mean_start_position)) &&
            rt_assert(data_logger_loader_discrete.readMatrixFromFile(dir_path, file_name_mean_goal_position, mean_goal_position))   &&
            rt_assert(data_logger_loader_discrete.readMatrixFromFile(dir_path, file_name_mean_tau, mean_tau)));
}

/**
 * NEVER RUN IN REAL-TIME\n
 * Save basis functions' values over variation of canonical state position into a file.
 *
 * @param file_path
 * @return Success or failure
 */
bool DMPDiscrete::saveBasisFunctions(const char* file_path)
{
    return (rt_assert(data_logger_loader_discrete.saveBasisFunctions(file_path)));
}

/**
 * Get the current weights of the basis functions.
 *
 * @param weights_buffer Weights buffer, to which the current weights will be copied onto (to be returned)
 * @return Success or failure
 */
bool DMPDiscrete::getWeights(MatrixNxM& weights_buffer)
{
    return (rt_assert(func_approx_discrete.getWeights(weights_buffer)));
}

/**
 * Set the new weights of the basis functions.
 *
 * @param new_weights New weights to be set
 * @return Success or failure
 */
bool DMPDiscrete::setWeights(const MatrixNxM& new_weights)
{
    return (rt_assert(func_approx_discrete.setWeights(new_weights)));
}

bool DMPDiscrete::startCollectingTrajectoryDataSet()
{
    if (rt_assert(data_logger_loader_discrete.reset()) == false)
    {
        return false;
    }

    is_collecting_trajectory_data       = true;

    return true;
}

/**
 * NON-REAL-TIME!!!\n
 * If you want to maintain real-time performance,
 * do/call this function from a separate (non-real-time) thread!!!\n
 * This function saves the trajectory data set there.
 *
 * @param new_data_dir_path New data directory path \n
 *                          (if we want to save the data in a directory path other than
 *                           the data_directory_path)
 * @return Success or failure
 */
bool DMPDiscrete::saveTrajectoryDataSet(const char* new_data_dir_path)
{
    // pre-conditions checking
    if (rt_assert(DMPDiscrete::isValid()) == false)
    {
        return false;
    }

    if (strcmp(new_data_dir_path, "") == 0)
    {
        return (rt_assert(data_logger_loader_discrete.saveTrajectoryDataSet(data_directory_path)));
    }
    else
    {
        return (rt_assert(data_logger_loader_discrete.saveTrajectoryDataSet(new_data_dir_path)));
    }
}

/**
 * @return The order of the canonical system (1st or 2nd order).
 */
uint DMPDiscrete::getCanonicalSysOrder()
{
    return (canonical_sys_discrete->getOrder());
}

/**
 * Returns model size used to represent the function.
 *
 * @return Model size used to represent the function
 */
uint DMPDiscrete::getFuncApproxModelSize()
{
    return (func_approx_discrete.getModelSize());
}

/**
 * @return The chosen formulation for the discrete transformation system\n
 *         0: original formulation by Schaal, AMAM 2003 (default formulation_type)\n
 *         1: formulation by Hoffman et al., ICRA 2009
 */
uint DMPDiscrete::getTransformSysFormulationType()
{
    return (transform_sys_discrete->getFormulationType());
}

/**
 * @return The learning method that is used to learn the DMP basis functions' weights\n
 *         0=_SCHAAL_LWR_METHOD_: Stefan Schaal's Locally-Weighted Regression (LWR): Ijspeert et al., Neural Comput. 25, 2 (Feb 2013), p328-373\n
 *         1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_: Jan Peters' Global Least Squares method
 */
uint DMPDiscrete::getLearningSysMethod()
{
    return (learning_sys_discrete.getLearningMethod());
}

DMPDiscrete::~DMPDiscrete()
{}

}
