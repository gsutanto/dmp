#include "amd_clmc_dmp/dmp_base/DMP.h"

namespace dmp
{

    DMP::DMP():
        dmp_num_dimensions(0), model_size(0), is_started(false), tau_sys(NULL), canonical_sys(NULL), func_approx(NULL),
        transform_sys(NULL), learning_sys(NULL), data_logger_loader(NULL), logged_dmp_variables(NULL),
        mean_start_position(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
        mean_goal_position(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
        mean_tau(0.0), is_collecting_trajectory_data(false), rt_assertor(NULL)
    {
        mean_start_position.resize(dmp_num_dimensions);
        mean_goal_position.resize(dmp_num_dimensions);
        strcpy(data_directory_path, "");
    }

    /**
     * @param dmp_num_dimensions_init Number of trajectory dimensions employed in this DMP
     * @param model_size_init Size of the model (M: number of basis functions or others) used to represent the function
     * @param canonical_system Canonical system that drives this DMP
     * @param function_approximator Non-linear function approximator that approximates the forcing term of this DMP
     * @param transform_system Transformation system used in this DMP
     * @param learning_system Learning system for learning the function approximator's parameters
     * @param dmp_data_io Data I/O (or logger/loader) for saving and loading into/from file(s)
     * @param logged_dmp_vars (Pointer to the) buffer for logged DMP variables
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     * @param opt_data_directory_path [optional] Full directory path where to save/log all the data
     */
    DMP::DMP(uint dmp_num_dimensions_init, uint model_size_init,
             CanonicalSystem* canonical_system, FunctionApproximator* function_approximator,
             TransformationSystem* transform_system, LearningSystem* learning_system, DMPDataIO* dmp_data_io,
             LoggedDMPVariables* logged_dmp_vars, RealTimeAssertor* real_time_assertor,
             const char* opt_data_directory_path):
        dmp_num_dimensions(dmp_num_dimensions_init), model_size(model_size_init), is_started(false),
        tau_sys(canonical_system->getTauSystemPointer()), canonical_sys(canonical_system),
        func_approx(function_approximator), transform_sys(transform_system),
        learning_sys(learning_system), data_logger_loader(dmp_data_io),
        logged_dmp_variables(logged_dmp_vars),
        mean_start_position(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
        mean_goal_position(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
        mean_tau(0.0), is_collecting_trajectory_data(false), rt_assertor(real_time_assertor)
    {
        mean_start_position.resize(dmp_num_dimensions);
        mean_goal_position.resize(dmp_num_dimensions);
        strcpy(data_directory_path, opt_data_directory_path);
    }

    /**
     * Checks whether the DMP is valid or not.
     *
     * @return DMP is valid (true) or DMP is invalid (false)
     */
    bool DMP::isValid()
    {
        if (rt_assert((rt_assert(dmp_num_dimensions > 0)) &&
                      (rt_assert(dmp_num_dimensions <= MAX_DMP_NUM_DIMENSIONS))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(model_size > 0)) && (rt_assert(model_size <= MAX_MODEL_SIZE))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(tau_sys            != NULL)) &&
                      (rt_assert(canonical_sys      != NULL)) &&
                      (rt_assert(func_approx        != NULL)) &&
                      (rt_assert(transform_sys      != NULL)) &&
                      (rt_assert(learning_sys       != NULL)) &&
                      (rt_assert(data_logger_loader != NULL))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(tau_sys->isValid())) &&
                      (rt_assert(canonical_sys->isValid())) &&
                      (rt_assert(func_approx->isValid())) &&
                      (rt_assert(transform_sys->isValid())) &&
                      (rt_assert(learning_sys->isValid())) &&
                      (rt_assert(data_logger_loader->isValid()))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(func_approx->getDMPNumDimensions()     == dmp_num_dimensions)) &&
                      (rt_assert(transform_sys->getDMPNumDimensions()   == dmp_num_dimensions)) &&
                      (rt_assert(learning_sys->getDMPNumDimensions()    == dmp_num_dimensions))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(model_size == func_approx->getModelSize())) &&
                      (rt_assert(model_size == learning_sys->getModelSize()))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(tau_sys == canonical_sys->getTauSystemPointer())) &&
                      (rt_assert(tau_sys == transform_sys->getTauSystemPointer()))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(canonical_sys == func_approx->getCanonicalSystemPointer())) &&
                      (rt_assert(canonical_sys == transform_sys->getCanonicalSystemPointer()))) == false)
        {
            return false;
        }
        if (rt_assert(func_approx == transform_sys->getFuncApproxPointer()) == false)
        {
            return false;
        }
        if (rt_assert(transform_sys == learning_sys->getTransformationSystemPointer()) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(mean_start_position.rows() == transform_sys->getCurrentGoalState().getX().rows())) &&
                      (rt_assert(mean_goal_position.rows()  == transform_sys->getCurrentGoalState().getX().rows()))) == false)
        {
            return false;
        }
        if (rt_assert(mean_tau >= 0.0) == false)
        {
            return false;
        }
        return true;
    }

    /**
     * Pre-process demonstrated trajectories before learning from it.
     * To be called before learning, for example to convert the trajectory representation to be
     * with respect to its reference (local) coordinate frame (in Cartesian Coordinate DMP), etc.
     * By default there is no pre-processing. If a specific DMP type (such as CartesianCoordDMP)
     * requires pre-processing, then just override this method/member function.
     *
     * @param trajectory_set Set of raw demonstrated trajectories to be pre-processed (in-place pre-processing)
     * @return Success or failure
     */
    bool DMP::preprocess(TrajectorySet& trajectory_set)
    {
        // pre-conditions checking
        if (rt_assert(DMP::isValid()) == false)
        {
            return false;
        }

        int N_traj                      = trajectory_set.size();

        // computing average start and goal positions across trajectories in the set:
        mean_start_position             = ZeroVectorN(dmp_num_dimensions);
        mean_goal_position              = ZeroVectorN(dmp_num_dimensions);
        for (uint i=0; i<N_traj; ++i)
        {
            int traj_size               = trajectory_set[i].size();
            if (rt_assert((rt_assert(traj_size > 0)) && (rt_assert(traj_size <= MAX_TRAJ_SIZE))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(trajectory_set[i][0].getDMPNumDimensions()           == dmp_num_dimensions)) &&
                          (rt_assert(trajectory_set[i][traj_size-1].getDMPNumDimensions() == dmp_num_dimensions))) == false)
            {
                return false;
            }
            mean_start_position         += trajectory_set[i][0].getX().block(0,0,dmp_num_dimensions,1);
            mean_goal_position          += trajectory_set[i][traj_size-1].getX().block(0,0,dmp_num_dimensions,1);
        }
        mean_start_position             = mean_start_position/(N_traj*1.0);
        mean_goal_position              = mean_goal_position/(N_traj*1.0);

        return true;
    }

    /**
     * NON-REAL-TIME!!!\n
     * Learns DMP's forcing term from trajectory(ies) in the specified file (or directory).
     *
     * @param training_data_dir_or_file_path Directory or file path where all training data is stored
     * @param robot_task_servo_rate Robot's task servo rate
     * @param tau_learn (optional) If you also want to know \n
     *                  the tau (movement duration) used during learning, \n
     *                  please put the pointer here
     * @param critical_states_learn (optional) If you also want to know \n
     *                              the critical states (start position, goal position, etc.) \n
     *                              used during learning, please put the pointer here
     * @return Success or failure
     */
    bool DMP::learn(const char* training_data_dir_or_file_path,
                    double robot_task_servo_rate,
                    double* tau_learn,
                    Trajectory* critical_states_learn)
    {
        bool is_real_time   = false;

        // Load sample input trajectory(ies):
        TrajectorySetPtr    set_traj_input;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time,
                                                  set_traj_input)) == false)
        {
            return false;
        }

        if (rt_assert(this->extractSetTrajectories(training_data_dir_or_file_path,
                                                   *set_traj_input, false)) == false)
        {
            return false;
        }

        // Learn the trajectory
        if (rt_assert(this->learn(*set_traj_input, robot_task_servo_rate)) == false)
        {
            return false;
        }

        if (tau_learn != NULL)
        {
            *tau_learn  = mean_tau;
        }

        if (critical_states_learn != NULL)
        {
            if (critical_states_learn->size() >= 2)
            {
                uint    end_idx                     = critical_states_learn->size() - 1;

                (*critical_states_learn)[0]         = DMPState(mean_start_position, rt_assertor);
                (*critical_states_learn)[end_idx]   = DMPState(mean_goal_position, rt_assertor);
            }
        }

        return true;
    }

    /**
     * Resets DMP and sets all needed parameters (including the reference coordinate frame) to start execution.
     * Call this function before beginning any DMP execution.
     *
     * @param dmp_unroll_init_parameters Parameter value definitions required to unroll a DMP.
     * @return Success or failure
     */
    bool DMP::start(const DMPUnrollInitParams& dmp_unroll_init_parameters)
    {
        // input checking
        if (rt_assert(dmp_unroll_init_parameters.isValid()) == false)
        {
            return false;
        }

        return (rt_assert(start(dmp_unroll_init_parameters.critical_states,
                                dmp_unroll_init_parameters.tau)));
    }

    /**
     * Based on the start state, current state, evolving goal state and steady-state goal position,
     * compute the target value of the coupling term, and then update the canonical state and goal state.
     * Needed for learning the coupling term function based on a given sequence of DMP states (trajectory).
     * [IMPORTANT]: Before calling this function, the TransformationSystem's start_state, current_state, and
     * goal_sys need to be initialized first using TransformationSystem::start() function,
     * and the forcing term weights need to have already been learned beforehand,
     * using the demonstrated trajectory.
     *
     * @param current_state_demo_local Demonstrated current DMP state in the local coordinate system \n
     *                                 (if there are coordinate transformations)
     * @param dt Time difference between the current and the previous execution
     * @param ct_acc_target Target value of the acceleration-level coupling term at
     *                      the demonstrated current state,\n
     *                      given the demonstrated start state, evolving goal state, \n
     *                      demonstrated steady-state goal position, and
     *                      the learned forcing term weights \n
     *                      (to be computed and returned by this function)
     * @return Success or failure
     */
    bool DMP::getTargetCouplingTermAndUpdateStates(const DMPState& current_state_demo_local,
                                                   const double& dt,
                                                   VectorN& ct_acc_target)
    {
        if (rt_assert(is_started == true) == false)
        {
            return false;
        }
        if (rt_assert(transform_sys->getTargetCouplingTerm(current_state_demo_local,
                                                           ct_acc_target)) == false)
        {
            return false;
        }
        if (rt_assert(canonical_sys->updateCanonicalState(dt)) == false)
        {
            return false;
        }
        if (rt_assert(transform_sys->updateCurrentGoalState(dt)) == false)
        {
            return false;
        }

        return true;
    }

    /**
     * Returns the average start position of the training set of trajectories.
     */
    VectorN DMP::getMeanStartPosition()
    {
        return (mean_start_position);
    }

    /**
     * Returns the average goal position of the training set of trajectories.
     */
    VectorN DMP::getMeanGoalPosition()
    {
        return (mean_goal_position);
    }

    /**
     * Returns the average tau of the training set of trajectories.
     */
    double DMP::getMeanTau()
    {
        return (mean_tau);
    }

    bool DMP::setDataDirectory(const char* new_data_dir_path)
    {
        if (rt_assert(createDirIfNotExistYet(new_data_dir_path)) == false)
        {
            return false;
        }

        strcpy(data_directory_path, new_data_dir_path);

        return true;
    }

    bool DMP::startCollectingTrajectoryDataSet()
    {
        if (rt_assert(data_logger_loader->reset()) == false)
        {
            return false;
        }

        is_collecting_trajectory_data       = true;

        return true;
    }

    void DMP::stopCollectingTrajectoryDataSet()
    {
        is_collecting_trajectory_data       = false;
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
    bool DMP::saveTrajectoryDataSet(const char* new_data_dir_path)
    {
        // pre-conditions checking
        if (rt_assert(DMP::isValid()) == false)
        {
            return false;
        }

        if (strcmp(new_data_dir_path, "") == 0)
        {
            return (rt_assert(data_logger_loader->saveTrajectoryDataSet(data_directory_path)));
        }
        else
        {
            return (rt_assert(data_logger_loader->saveTrajectoryDataSet(new_data_dir_path)));
        }
    }

    bool DMP::setTransformSystemCouplingTermUsagePerDimensions(const std::vector<bool>& is_using_transform_sys_coupling_term_at_dimension_init)
    {
        return (rt_assert(transform_sys->setCouplingTermUsagePerDimensions(is_using_transform_sys_coupling_term_at_dimension_init)));
    }

    FunctionApproximator* DMP::getFunctionApproximatorPointer()
    {
        return func_approx;
    }

    DMP::~DMP()
    {}

}
