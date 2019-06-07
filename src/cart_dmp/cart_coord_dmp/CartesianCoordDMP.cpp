#include "amd_clmc_dmp/cart_dmp/cart_coord_dmp/CartesianCoordDMP.h"

namespace dmp
{

/**
 * NEVER RUN IN REAL-TIME, ONLY RUN IN INIT ROUTINE
 */
CartesianCoordDMP::CartesianCoordDMP():
    transform_sys_discrete_cart_coord(TransformSystemDiscrete(3, NULL,
                                                              &func_approx_discrete, std::vector<bool>(3, true),
                                                              _SCHAAL_DMP_, &logged_dmp_discrete_variables,
                                                              NULL, NULL, NULL, NULL, NULL, NULL)),
    DMPDiscrete(3, 0, NULL, &transform_sys_discrete_cart_coord,
                _SCHAAL_LWR_METHOD_, NULL, ""),
    cart_coord_transformer(CartesianCoordTransformer()),
    ctraj_local_coord_selection(2)
{
    ctraj_hmg_transform_local_to_global_matrix  = ZeroMatrix4x4;
    ctraj_hmg_transform_global_to_local_matrix  = ZeroMatrix4x4;

    ctraj_critical_states_global_coord.resize(5);
    ctraj_critical_states_local_coord.resize(5);
}

/**
 * NEVER RUN IN REAL-TIME, ONLY RUN IN INIT ROUTINE
 *
 * @param canonical_system_discrete (Pointer to) discrete canonical system that drives this DMP
 * @param model_size_init Size of the model (M: number of basis functions or others) used to represent the function
 * @param ts_formulation_type Chosen formulation for the discrete transformation system\n
 *        _SCHAAL_DMP_: original formulation by Schaal, AMAM 2003 (default formulation_type)\n
 *        _HOFFMANN_DMP_: formulation by Hoffmann et al., ICRA 2009
 * @param learning_method Learning method that will be used to learn the DMP basis functions' weights\n
 *        0=_SCHAAL_LWR_METHOD_: Stefan Schaal's Locally-Weighted Regression (LWR): Ijspeert et al., Neural Comput. 25, 2 (Feb 2013), p328-373\n
 *        1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_: Jan Peters' Global Least Squares method
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 * @param ctraj_local_coordinate_frame_selection Trajectory local coordinate frame that will be used as\n
 *                                               the reference frame. Possible values are:\n
 *        0: No transformation, means local coordinate frame's axes is exactly in the same direction as\n
 *           the global coordinate frame's axes\n
 *        1: G. Sutanto's trajectory local coordinate system formulation:\n
 *           Local x-axis is the normalized vector going from start to goal position\n
 *           This method is designed to tackle the problem on \n
 *           Schaal's trajectory local coordinate system formulation (ctraj_local_coordinate_frame_selection == 2),
 *           which at certain cases might have a discontinuous local coordinate
 *           transformation as the local x-axis changes with respect to the
 *           global coordinate system, due to the use of hard-coded IF
 *           statement to tackle the situation where local x-axis gets close
 *           as being parallel to global z-axis.\n
 *           G. Sutanto's trajectory local coordinate system formulation tackle
 *           this problem by defining 5 basis vectors as candidates of anchor local
 *           z-axis at 5 different situations as defined below:\n
 *           >> theta == 0 (local_x_axis is perfectly aligned with global_z_axis)
 *              corresponds to anchor_local_z_axis = global_y_axis\n
 *           >> theta == pi/4 corresponds to
 *              anchor_local_z_axis = global_z_axis X local_x_axis\n
 *           >> theta == pi/2 (local_x_axis is perpendicular from global_z_axis)
 *              corresponds to anchor_local_z_axis = global_z_axis\n
 *           >> theta == 3*pi/4 corresponds to
 *              anchor_local_z_axis = -global_z_axis X local_x_axis\n
 *           >> theta == pi (local_x_axis is perfectly aligned with -global_z_axis)
 *              corresponds to anchor_local_z_axis = -global_y_axis\n
 *           To ensure continuity/smoothness on transitions among these 5 cases,
 *           anchor_local_z_axis are the normalized weighted linear combination of
 *           these 5 basis vectors (the weights are gaussian kernels centered at the
 *           5 different situations/cases above).\n
 *        2: Schaal's trajectory local coordinate system formulation:\n
 *           Local x-axis is the normalized vector going from start to goal position\n
 *           >> IF   local x-axis is "NOT TOO parallel" to the global z-axis\n
 *                   (checked by: if dot-product or projection length is below some threshold)\n
 *              THEN local x-axis and global z-axis (~= local z-axis) become the anchor axes,\n
 *                   local y-axis is the cross-product between them (z X x)\n
 *           >> ELSE IF local x-axis is "TOO parallel" to the global z-axis\n
 *              THEN    local x-axis and global y-axis (~= local y-axis) become the anchor axes,\n
 *                      local z-axis is the cross-product between them (x X y)\n
 *           Scaling is active only on local x-axis.\n
 *        3: Kroemer's trajectory local coordinate system formulation\n
 *           See O.B. Kroemer, R. Detry, J. Piater, and J. Peters;\n
 *               Combining active learning and reactive control for robot grasping;\n
 *               Robotics and Autonomous Systems 58 (2010), page 1111 (Figure 5)\n
 *           Local x-axis is the normalized vector approaching steady-state goal position\n
 *           "Proposed" local y-axis is the normalized vector going from start \n
 *           to a point "near" goal position (along local x-axis) such that local y-axis \n
 *           is perpendicular to the local x-axis.\n
 *           >> IF   local x-axis is "NOT TOO parallel" to the local y-axis\n
 *                   (checked by: if dot-product or projection length is below some threshold)\n
 *              THEN local x-axis and local y-axis become the anchor axes,\n
 *                   local z-axis is the cross-product between them (x X y)\n
 *           >> ELSE IF local x-axis is "TOO parallel" to the local y-axis\n
 *              THEN    (follow a similar method as Schaal's trajectory local coordinate system formulation:)\n
 *              >> IF   local x-axis is "NOT TOO parallel" to the global z-axis\n
 *                 THEN local x-axis and global z-axis (~= local z-axis) become the anchor axes,\n
 *                      local y-axis is the cross-product between them (z X x)\n
 *              >> ELSE IF local x-axis is "TOO parallel" to the global z-axis\n
 *                 THEN    local x-axis and global y-axis (~= local y-axis) become the anchor axes,\n
 *                         local z-axis is the cross-product between them (x X y)\n
 *           Scaling is active on local x-axis and local y-axis.\n
 *        Default selected value is 1.
 * @param opt_data_directory_path [optional] Full directory path where to save/log all the data
 * @param transform_couplers All spatial couplings associated with the discrete transformation system
 * (please note that this is a pointer to vector of pointer to TransformCoupling data structure, for flexibility and re-usability)
 */
CartesianCoordDMP::CartesianCoordDMP(CanonicalSystemDiscrete* canonical_system_discrete,
                                     uint model_size_init,
                                     uint ts_formulation_type, uint learning_method,
                                     RealTimeAssertor* real_time_assertor,
                                     uint ctraj_local_coordinate_frame_selection,
                                     const char* opt_data_directory_path,
                                     std::vector<TransformCoupling*>* transform_couplers):
    transform_sys_discrete_cart_coord(TransformSystemDiscrete(3, canonical_system_discrete,
                                                              &func_approx_discrete, std::vector<bool>(3, true),
                                                              ts_formulation_type, &logged_dmp_discrete_variables,
                                                              real_time_assertor, NULL, NULL, NULL, NULL,
                                                              transform_couplers)),
    DMPDiscrete(3, model_size_init, canonical_system_discrete, &transform_sys_discrete_cart_coord,
                learning_method, real_time_assertor, opt_data_directory_path),
    cart_coord_transformer(CartesianCoordTransformer(real_time_assertor)),
    ctraj_local_coord_selection(ctraj_local_coordinate_frame_selection)
{
    // if using Schaal's DMP model:
    if (ts_formulation_type == _SCHAAL_DMP_)
    {
        // only (local) x-axis DMP is using scaling, (local) y and (local) z-axes do NOT use scaling
        std::vector<bool>   is_using_scaling_init    = std::vector<bool>(3, true);
        if ((ctraj_local_coord_selection == _GSUTANTO_LOCAL_COORD_FRAME_) || (ctraj_local_coord_selection == _SCHAAL_LOCAL_COORD_FRAME_))
        {
            is_using_scaling_init[1]                 = false;    // turn-off scaling on y-axis
            is_using_scaling_init[2]                 = false;    // turn-off scaling on z-axis
        }
        else if (ctraj_local_coord_selection == _KROEMER_LOCAL_COORD_FRAME_)
        {
            is_using_scaling_init[2]                 = false;    // turn-off scaling on z-axis
        }
        DMPDiscrete::setScalingUsage(is_using_scaling_init);
        // (let me know if there is a better way of initializing std::vector<bool> than this)
    }

    ctraj_hmg_transform_local_to_global_matrix  = ZeroMatrix4x4;
    ctraj_hmg_transform_global_to_local_matrix  = ZeroMatrix4x4;

    ctraj_critical_states_global_coord.resize(5);
    ctraj_critical_states_local_coord.resize(5);
}

/**
 * Checks whether this Cartesian Coordinate DMP is valid or not.
 *
 * @return Cartesian Coordinate DMP is valid (true) or Cartesian Coordinate DMP is invalid (false)
 */
bool CartesianCoordDMP::isValid()
{
    if (rt_assert(transform_sys_discrete_cart_coord.isValid()) == false)
    {
        return false;
    }
    if (rt_assert(DMPDiscrete::isValid()) == false)
    {
        return false;
    }
    if (rt_assert(transform_sys_discrete == &transform_sys_discrete_cart_coord) == false)
    {
        return false;
    }
    if (rt_assert(dmp_num_dimensions == 3) == false)
    {
        return false;
    }
    if (rt_assert(cart_coord_transformer.isValid()) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(ctraj_critical_states_global_coord.size()  == 5)) &&
                  (rt_assert(ctraj_critical_states_local_coord.size()   == 5))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(ctraj_local_coord_selection >= MIN_CTRAJ_LOCAL_COORD_OPTION_NO)) &&
                  (rt_assert(ctraj_local_coord_selection <= MAX_CTRAJ_LOCAL_COORD_OPTION_NO))) == false)
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
bool CartesianCoordDMP::preprocess(TrajectorySet& trajectory_set)
{
    if (rt_assert(DMP::preprocess(trajectory_set)) == false)
    {
        return false;
    }

    mean_start_global_position  = mean_start_position.block(0,0,3,1);
    mean_goal_global_position   = mean_goal_position.block(0,0,3,1);

    int N_traj                  = trajectory_set.size();  // size of Cartesian coordinate trajectory
    for (uint i=0; i<N_traj; ++i)
    {
        // computing the transformation to local coordinate system:
        if (rt_assert(cart_coord_transformer.computeCTrajCoordTransforms(trajectory_set[i],
                                                                         ctraj_hmg_transform_local_to_global_matrix,
                                                                         ctraj_hmg_transform_global_to_local_matrix,
                                                                         ctraj_local_coord_selection)) == false)
        {
            return false;
        }

        // computing trajectory representation in the local coordinate system:
        if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(trajectory_set[i],
                                                                            ctraj_hmg_transform_global_to_local_matrix,
                                                                            trajectory_set[i])) == false)
        {
            return false;
        }
    }

    if (rt_assert(cart_coord_transformer.computeCPosAtNewCoordSys(mean_start_global_position,
                                                                  ctraj_hmg_transform_global_to_local_matrix,
                                                                  mean_start_local_position)) == false)
    {
        return false;
    }

    if (rt_assert(cart_coord_transformer.computeCPosAtNewCoordSys(mean_goal_global_position,
                                                                  ctraj_hmg_transform_global_to_local_matrix,
                                                                  mean_goal_local_position)) == false)
    {
        return false;
    }

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
bool CartesianCoordDMP::learn(const char* training_data_dir_or_file_path,
                              double robot_task_servo_rate,
                              double* tau_learn,
                              Trajectory* critical_states_learn)
{
    return(rt_assert(DMP::learn(training_data_dir_or_file_path, robot_task_servo_rate,
                                tau_learn, critical_states_learn)));
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
bool CartesianCoordDMP::learn(const TrajectorySet& trajectory_set, double robot_task_servo_rate)
{
    bool is_real_time   = false;

    // pre-conditions checking
    if (rt_assert(CartesianCoordDMP::isValid()) == false)
    {
        return false;
    }
    // input checking:
    if (rt_assert((rt_assert(robot_task_servo_rate >  0.0)) &&
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
    if (rt_assert(CartesianCoordDMP::preprocess(*trajectory_set_preprocessed)) == false)
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
 * NON-REAL-TIME!!!\n
 * Learn multiple CartesianCoordDMP movement primitives at once.
 *
 * @param in_data_dir_path Input data directory path, containing sample trajectories, grouped by folder.\n
 *                         A single CartesianCoordDMP movement primitive will be learnt from each folder.
 * @param out_data_dir_path Output data directory path, where \n
 *                          the learnt CartesianCoordDMP movement primitives parameters \n
 *                          (weights and learning amplitude) will be saved into.\n
 *                          If (save_unroll_traj_dataset==true), the unrolled trajectories will also\n
 *                          saved in this directory path.
 * @param robot_task_servo_rate Robot's task servo rate
 * @param in_data_subdir_subpath If each folder UNDER in_data_dir_path containing \n
 *                               another sub-directories where the sample trajectories set are actually saved,\n
 *                               specify the sub-path here.
 * @param save_unroll_traj_dataset Choose whether to try unrolling \n
 *                                 the learnt CartesianCoordDMP movement primitives (true) or not (false).
 * @param tau_unroll Tau for movement primitives unrolling
 * @return Success or failure
 */
bool CartesianCoordDMP::learnMultiCartCoordDMPs(const char* in_data_dir_path,
                                                const char* out_data_dir_path,
                                                double robot_task_servo_rate,
                                                const char* in_data_subdir_subpath,
                                                bool save_unroll_traj_dataset, double tau_unroll)
{
    bool is_real_time       = false;

    // pre-conditions checking
    if (rt_assert(CartesianCoordDMP::isValid()) == false)
    {
        return false;
    }
    // input checking
    if (rt_assert(file_type(in_data_dir_path) == _DIR_) == false)   // the in_data_dir_path directory must have existed!!!
    {
        return false;
    }

    if (rt_assert(createDirIfNotExistYet(out_data_dir_path)) == false)
    {
        return false;
    }

    boost::filesystem::path                 in_data_dir(in_data_dir_path);
    boost::filesystem::directory_iterator   null_directory_iterator;
    char                                    in_data_subdir_path[1000];
    char                                    out_data_subdir_path[1000];

    for (boost::filesystem::directory_iterator dir_iter(in_data_dir); dir_iter != null_directory_iterator; ++dir_iter)
    {
        char                                subdir_name[1000];
        strcpy(subdir_name, dir_iter->path().filename().string().c_str());
        sprintf(in_data_subdir_path, "%s/%s/", dir_iter->path().string().c_str(), in_data_subdir_subpath);
        sprintf(out_data_subdir_path, "%s/%s/", out_data_dir_path, subdir_name);

        // Create out_data_subdir_path directory if it doesn't exist yet:
        if (rt_assert(createDirIfNotExistYet(out_data_subdir_path)) == false)
        {
            return false;
        }

        TrajectorySetPtr                    set_ctraj_input;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time,
                                                  set_ctraj_input, 0)) == false)
        {
            return false;
        }

        // Load the set of demonstrated Cartesian coordinate trajectories:
        if (rt_assert(CartesianCoordDMP::extractSetTrajectories(in_data_subdir_path,
                                                                *set_ctraj_input,
                                                                false)) == false)
        {
            return false;
        }

        // Learn the CartesianCoordDMP movement primitive from the demonstrated Cartesian coordinate trajectories set:
        if (rt_assert(CartesianCoordDMP::learn(*set_ctraj_input, robot_task_servo_rate)) == false)
        {
            return false;
        }

        // Save the learned parameters of the CartesianCoordDMP movement primitive:
        if (rt_assert(CartesianCoordDMP::saveParams(out_data_subdir_path)) == false)
        {
            return false;
        }

        if (save_unroll_traj_dataset)
        {
            // Set tau for movement reproduction:
            double  dt                      = 1.0/robot_task_servo_rate;

            // total number of states to be reproduced/unrolled:
            uint    N_reprod_traj_states    = round(tau_unroll*robot_task_servo_rate) + 1;

            // Set the critical states (start position and goal position):
            Trajectory          critical_states(2);
            critical_states[0]  = DMPState(CartesianCoordDMP::getMeanStartPosition(), rt_assertor);
            critical_states[1]  = DMPState(CartesianCoordDMP::getMeanGoalPosition(), rt_assertor);

            // Start the BASELINE (w/o obstacle) DMP (including the computation of the coordinate transformation):
            if (rt_assert(CartesianCoordDMP::start(critical_states, tau_unroll)) == false)
            {
                return false;
            }

            // Reproduce/unroll the learned CartesianCoordDMP movement primitive trajectory:
            std::cout << "Unrolling CartesianCoordDMP movement primitive #" << subdir_name << std::endl;
            if (rt_assert(CartesianCoordDMP::startCollectingTrajectoryDataSet()) == false)
            {
                return false;
            }
            for (uint i=0; i<N_reprod_traj_states; ++i)
            {
                // Get the next state of the Cartesian DMP
                DMPState        endeff_cart_state_global(rt_assertor);

                if (rt_assert(CartesianCoordDMP::getNextState(dt, true,
                                                              endeff_cart_state_global)) == false)
                {
                    return false;
                }
            }
            CartesianCoordDMP::stopCollectingTrajectoryDataSet();
            if (rt_assert(CartesianCoordDMP::saveTrajectoryDataSet(out_data_subdir_path)) == false)
            {
                return false;
            }
        }
    }

    return true;
}

/**
 * Resets DMP and sets all needed parameters (including the reference coordinate frame) to start execution.
 * Call this function before beginning any DMP execution.
 *
 * @param critical_states Critical states defined for DMP trajectory reproduction/unrolling.\n
 *                        Start state is the first state in this trajectory/vector of DMPStates.\n
 *                        Goal state is the last state in this trajectory/vector of DMPStates.\n
 *                        All are defined with respect to the global coordinate system.
 * @param tau_init Time parameter that influences the speed of the execution (usually this is the time-length of the trajectory)
 * @return Success or failure
 */
bool CartesianCoordDMP::start(const Trajectory& critical_states,
                              double tau_init)
{
    // pre-conditions checking
    if (rt_assert(CartesianCoordDMP::isValid()) == false)
    {
        return false;
    }
    // input checking
    if (rt_assert((rt_assert(critical_states.size() >= 2)) &&
                  (rt_assert(critical_states.size() <= MAX_TRAJ_SIZE))) == false)
    {
        return false;
    }
    if (rt_assert(((ctraj_local_coord_selection == 3) && (critical_states.size() < 3)) == false) == false)
    {
        return false;
    }
    DMPState    start_state_global_init                 = critical_states[0];
    DMPState    approaching_ss_goal_state_global_init   = critical_states[critical_states.size()-2];    // approaching-steady-state goal state in the global coordinate frame
    DMPState    ss_goal_state_global_init               = critical_states[critical_states.size()-1];    // steady-state goal state in the global coordinate frame
    if (rt_assert((rt_assert(start_state_global_init.isValid())) &&
                  (rt_assert(approaching_ss_goal_state_global_init.isValid())) &&
                  (rt_assert(ss_goal_state_global_init.isValid()))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(start_state_global_init.getDMPNumDimensions()                  == 3)) &&
                  (rt_assert(approaching_ss_goal_state_global_init.getDMPNumDimensions()    == 3)) &&
                  (rt_assert(ss_goal_state_global_init.getDMPNumDimensions()                == 3))) == false)
    {
        return false;
    }
    if (rt_assert(tau_init >= MIN_TAU) == false)    // tau_init is too small/does NOT make sense
    {
        return false;
    }

    // Compute critical states on the new local coordinate system:
    DMPState    current_state_global_init               = start_state_global_init;
    DMPState    current_goal_state_global_init;
    if ((transform_sys_discrete_cart_coord.getFormulationType() == _SCHAAL_DMP_) &&
        (canonical_sys_discrete->getOrder() == 2))
    {
        current_goal_state_global_init                  = start_state_global_init;
    }
    else
    {
        current_goal_state_global_init                  = ss_goal_state_global_init;
    }
    ctraj_critical_states_global_coord[0]                                           = start_state_global_init;
    ctraj_critical_states_global_coord[1]                                           = current_state_global_init;
    ctraj_critical_states_global_coord[2]                                           = current_goal_state_global_init;
    ctraj_critical_states_global_coord[ctraj_critical_states_global_coord.size()-2] = approaching_ss_goal_state_global_init;
    ctraj_critical_states_global_coord[ctraj_critical_states_global_coord.size()-1] = ss_goal_state_global_init;

    mean_start_global_position  = start_state_global_init.getX();
    mean_goal_global_position   = ss_goal_state_global_init.getX();

    // Compute new coordinate system transformation (representing new local coordinate system) for trajectory reproduction:
    if (rt_assert(cart_coord_transformer.computeCTrajCoordTransforms(ctraj_critical_states_global_coord,
                                                                     ctraj_hmg_transform_local_to_global_matrix,
                                                                     ctraj_hmg_transform_global_to_local_matrix,
                                                                     ctraj_local_coord_selection)) == false)
    {
        return false;
    }

    if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(ctraj_critical_states_global_coord,
                                                                        ctraj_hmg_transform_global_to_local_matrix,
                                                                        ctraj_critical_states_local_coord)) == false)
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
    if (rt_assert(transform_sys_discrete_cart_coord.start(ctraj_critical_states_local_coord[0],
                                                          ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-1])) == false)
    {
        return false;
    }

    mean_start_local_position   = ctraj_critical_states_local_coord[0].getX();
    mean_goal_local_position    = ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-1].getX();

    is_started  = true;

    return true;
}

/**
 * Runs one step of the DMP based on the time step dt.
 *
 * @param dt Time difference between the previous and the current step
 * @param update_canonical_state Set to true if the canonical state should be updated here. Otherwise you are responsible for updating it yourself.
 * @param result_dmp_state_global Computed next DMPState on the global coordinate system (to be computed/returned)
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
bool CartesianCoordDMP::getNextState(double dt, bool update_canonical_state,
                                     DMPState& result_dmp_state_global,
                                     VectorN* transform_sys_forcing_term,
                                     VectorN* transform_sys_coupling_term_acc,
                                     VectorN* transform_sys_coupling_term_vel,
                                     VectorM* func_approx_basis_functions,
                                     VectorM* func_approx_normalized_basis_func_vector_mult_phase_multiplier)
{
    DMPState    result_dmp_state_local(3, rt_assertor);

    return (rt_assert(CartesianCoordDMP::getNextState(dt, update_canonical_state, result_dmp_state_global,
                                                      result_dmp_state_local,
                                                      transform_sys_forcing_term,
                                                      transform_sys_coupling_term_acc,
                                                      transform_sys_coupling_term_vel,
                                                      func_approx_basis_functions,
                                                      func_approx_normalized_basis_func_vector_mult_phase_multiplier)));
}

/**
 * Runs one step of the DMP based on the time step dt.
 *
 * @param dt Time difference between the previous and the current step
 * @param update_canonical_state Set to true if the canonical state should be updated here. Otherwise you are responsible for updating it yourself.
 * @param result_dmp_state_global Computed next DMPState on the global coordinate system (to be computed/returned)
 * @param result_dmp_state_local Computed next DMPState on the local coordinate system (to be computed/returned)
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
bool CartesianCoordDMP::getNextState(double dt, bool update_canonical_state,
                                     DMPState& result_dmp_state_global,
                                     DMPState& result_dmp_state_local,
                                     VectorN* transform_sys_forcing_term,
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
    if (rt_assert(CartesianCoordDMP::isValid()) == false)
    {
        return false;
    }

    if (rt_assert(transform_sys_discrete_cart_coord.getNextState(dt, result_dmp_state_local,
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

    if (rt_assert(transform_sys_discrete_cart_coord.updateCurrentGoalState(dt)) == false)
    {
        return false;
    }

    // update critical states in local coordinate frame (only current state and current goal state changes)
    //ctraj_critical_states_local_coord[0]    = transform_sys_discrete_cart_coord.getStartState();
    ctraj_critical_states_local_coord[1]    = result_dmp_state_local;  // = transform_sys_discrete_cart_coord.getCurrentState();
    ctraj_critical_states_local_coord[2]    = transform_sys_discrete_cart_coord.getCurrentGoalState();
    //ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-2]   = ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-2];
    //ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-1]   = DMPState(transform_sys_discrete_cart_coord.getSteadyStateGoalPosition());

    // update critical states in global coordinate frame
    if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(ctraj_critical_states_local_coord,
                                                                        ctraj_hmg_transform_local_to_global_matrix,
                                                                        ctraj_critical_states_global_coord)) == false)
    {
        return false;
    }

    result_dmp_state_global                 = ctraj_critical_states_global_coord[1];

    /**************** For Trajectory Data Logging Purposes (Start) ****************/
    logged_dmp_discrete_variables.transform_sys_state_global        = result_dmp_state_global;
    logged_dmp_discrete_variables.goal_state_global                 = ctraj_critical_states_global_coord[2];
    logged_dmp_discrete_variables.steady_state_goal_position_global = ctraj_critical_states_global_coord[ctraj_critical_states_global_coord.size()-1].getX();

    if (is_collecting_trajectory_data)
    {
        if (rt_assert(data_logger_loader_discrete.collectTrajectoryDataSet(logged_dmp_discrete_variables)) == false)
        {
            return false;
        }
    }
    /**************** For Trajectory Data Logging Purposes (End) ****************/

    return true;
}

/**
 * Returns current (global) DMP state.
 */
DMPState CartesianCoordDMP::getCurrentState()
{
    // update critical states in local coordinate frame (only current state and current goal state changes)
    //ctraj_critical_states_local_coord[0]    = transform_sys_discrete_cart_coord.getStartState();
    ctraj_critical_states_local_coord[1]    = transform_sys_discrete_cart_coord.getCurrentState();
    ctraj_critical_states_local_coord[2]    = transform_sys_discrete_cart_coord.getCurrentGoalState();
    //ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-2]   = ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-2];
    //ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-1]   = DMPState(transform_sys_discrete_cart_coord.getSteadyStateGoalPosition());

    // update critical states in global coordinate frame
    rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(ctraj_critical_states_local_coord,
                                                                    ctraj_hmg_transform_local_to_global_matrix,
                                                                    ctraj_critical_states_global_coord));

    return (ctraj_critical_states_global_coord[1]);
}

/**
 * Returns current goal position on x, y, and z axes.
 *
 * @param current_goal_global Current goal position on the global coordinate system (to be returned)
 * @return Success or failure
 */
bool CartesianCoordDMP::getCurrentGoalPosition(Vector3& current_goal_global)
{
    DMPState    current_goal_state_local  = transform_sys_discrete_cart_coord.getCurrentGoalState();
    DMPState    current_goal_state_global(3, rt_assertor);

    // Compute current goal state on global coordinate system:
    if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(current_goal_state_local,
                                                                        ctraj_hmg_transform_local_to_global_matrix,
                                                                        current_goal_state_global)) == false)
    {
        return false;
    }

    current_goal_global = current_goal_state_global.getX().block(0,0,3,1);
    return true;
}

/**
 * Returns steady-state goal position vector (G) on the global coordinate system.
 *
 * @param Steady-state goal position vector (G) on the global coordinate system (to be returned)
 * @return Success or failure
 */
bool CartesianCoordDMP::getSteadyStateGoalPosition(Vector3& steady_state_goal_global)
{
    VectorN steady_state_goal_local(3);
    steady_state_goal_local                     = ZeroVector3;
    Vector4 steady_state_goal_local_H           = ZeroVector4;
    Vector4 steady_state_goal_global_H          = ZeroVector4;

    // Get steady-state goal position on the local coordinate system:
    if (rt_assert(DMPDiscrete::getSteadyStateGoalPosition(steady_state_goal_local)) == false)
    {
        return false;
    }
    steady_state_goal_local_H.block(0,0,3,1)    = steady_state_goal_local.block(0,0,3,1);
    steady_state_goal_local_H(3,0)              = 1.0;
    steady_state_goal_global_H(3,0)             = 1.0;

    // Compute steady-state goal position on the global coordinate system:
    if (rt_assert(cart_coord_transformer.computeCTrajAtNewCoordSys(&steady_state_goal_local_H, NULL, NULL,NULL,
                                                                   ctraj_hmg_transform_local_to_global_matrix,
                                                                   &steady_state_goal_global_H, NULL, NULL,NULL)) == false)
    {
        return false;
    }

    steady_state_goal_global                    = steady_state_goal_global_H.block(0,0,3,1);
    return true;
}

/**
 * Sets new steady-state goal position (G).
 *
 * @param new_G New steady-state goal position (G) to be set
 * @return Success or failure
 */
bool CartesianCoordDMP::setNewSteadyStateGoalPosition(const VectorN& new_G)
{
    return (rt_assert(CartesianCoordDMP::setNewSteadyStateGoalPosition(new_G, new_G)));
}

/**
 * Sets new steady-state Cartesian goal position (G).
 *
 * @param new_G_global New steady-state Cartesian goal position (G) on the global coordinate system
 * @param new_approaching_G_global New approaching-steady-state Cartesian goal position on \n
 *                                 the global coordinate system. Need to be specified correctly \n
 *                                 when using ctraj_local_coord_selection = 3, otherwise can be \n
 *                                 specified arbitrarily (see more definition/explanation in \n
 *                                 CartesianCoordTransformer::computeCTrajCoordTransforms)
 * @return Success or failure
 */
bool CartesianCoordDMP::setNewSteadyStateGoalPosition(const Vector3& new_G_global,
                                                      const Vector3& new_approaching_G_global)
{
    ctraj_critical_states_local_coord[0]     = transform_sys_discrete_cart_coord.getStartState();
    ctraj_critical_states_local_coord[1]     = transform_sys_discrete_cart_coord.getCurrentState();
    ctraj_critical_states_local_coord[2]     = transform_sys_discrete_cart_coord.getCurrentGoalState();
    ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-2]   = ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-2];
    ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-1]   = DMPState(transform_sys_discrete_cart_coord.getSteadyStateGoalPosition(), rt_assertor);

    // Compute critical states on the global coordinate system
    // (using the OLD coordinate transformation):
    if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(ctraj_critical_states_local_coord,
                                                                        ctraj_hmg_transform_local_to_global_matrix,
                                                                        ctraj_critical_states_global_coord)) == false)
    {
        return false;
    }

    // Compute the NEW coordinate system transformation (representing the new local coordinate system),
    // based on the original start state, the new approaching-steady-state goal,
    // the new steady-state goal (G, NOT the current/transient/non-steady-state goal),
    // both on global coordinate system, for trajectory reproduction:
    ctraj_critical_states_global_coord[ctraj_critical_states_global_coord.size()-2] = DMPState(new_approaching_G_global, rt_assertor);
    ctraj_critical_states_global_coord[ctraj_critical_states_global_coord.size()-1] = DMPState(new_G_global, rt_assertor);
    if (rt_assert(cart_coord_transformer.computeCTrajCoordTransforms(ctraj_critical_states_global_coord,
                                                                     ctraj_hmg_transform_local_to_global_matrix,
                                                                     ctraj_hmg_transform_global_to_local_matrix,
                                                                     ctraj_local_coord_selection)) == false)
    {
        return false;
    }

    // Compute the critical states on the NEW local coordinate system:
    if (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(ctraj_critical_states_global_coord,
                                                                        ctraj_hmg_transform_global_to_local_matrix,
                                                                        ctraj_critical_states_local_coord)) == false)
    {
        return false;
    }

    if (rt_assert(transform_sys_discrete_cart_coord.setStartState(ctraj_critical_states_local_coord[0])) == false)
    {
        return false;
    }
    if (rt_assert(transform_sys_discrete_cart_coord.setCurrentState(ctraj_critical_states_local_coord[1])) == false)
    {
        return false;
    }
    if (rt_assert(transform_sys_discrete_cart_coord.setCurrentGoalState(ctraj_critical_states_local_coord[2])) == false)
    {
        return false;
    }
    if (rt_assert(transform_sys_discrete_cart_coord.setSteadyStateGoalPosition(ctraj_critical_states_local_coord[ctraj_critical_states_local_coord.size()-1].getX())) == false)
    {
        return false;
    }

    return true;
}

/**
 * Convert DMPState representation from local coordinate system into global coordinate system.
 *
 * @param dmp_state_local DMPState representation in the local coordinate system
 * @param dmp_state_global DMPState representation in the global coordinate system
 * @return Success or failure
 */
bool CartesianCoordDMP::convertToGlobalDMPState(const DMPState& dmp_state_local, DMPState& dmp_state_global)
{
    return (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(dmp_state_local,
                                                                            ctraj_hmg_transform_local_to_global_matrix,
                                                                            dmp_state_global)));
}

/**
 * Convert DMPState representation from global coordinate system into local coordinate system.
 *
 * @param dmp_state_global DMPState representation in the global coordinate system
 * @param dmp_state_local DMPState representation in the local coordinate system
 * @return Success or failure
 */
bool CartesianCoordDMP::convertToLocalDMPState(const DMPState& dmp_state_global, DMPState& dmp_state_local)
{
    return (rt_assert(cart_coord_transformer.convertCTrajAtOldToNewCoordSys(dmp_state_global,
                                                                            ctraj_hmg_transform_global_to_local_matrix,
                                                                            dmp_state_local)));
}

Matrix4x4 CartesianCoordDMP::getHomogeneousTransformMatrixLocalToGlobal()
{
    return (ctraj_hmg_transform_local_to_global_matrix);
}

Matrix4x4 CartesianCoordDMP::getHomogeneousTransformMatrixGlobalToLocal()
{
    return (ctraj_hmg_transform_global_to_local_matrix);
}

Vector3 CartesianCoordDMP::getMeanStartGlobalPosition()
{
    return (mean_start_global_position);
}

Vector3 CartesianCoordDMP::getMeanGoalGlobalPosition()
{
    return (mean_goal_global_position);
}

Vector3 CartesianCoordDMP::getMeanStartLocalPosition()
{
    return (mean_start_local_position);
}

Vector3 CartesianCoordDMP::getMeanGoalLocalPosition()
{
    return (mean_goal_local_position);
}

/**
 * NON-REAL-TIME!!!\n
 * Write a Cartesian coordinate trajectory's information/content to a text file.
 *
 * @param cart_coord_trajectory Cartesian coordinate trajectory to be written to a file
 * @param file_path File path where to write the Cartesian coordinate trajectory's information into
 * @param is_comma_separated If true, Cartesian coordinate trajectory fields will be written with comma-separation;\n
 *        If false, Cartesian coordinate trajectory fields will be written with white-space-separation (default is true)
 * @return Success or failure
 */
bool CartesianCoordDMP::writeCartCoordTrajectoryToFile(const Trajectory& cart_coord_trajectory,
                                                       const char* file_path,
                                                       bool is_comma_separated)
{
    return (rt_assert(data_logger_loader_discrete.writeCartCoordTrajectoryToFile(cart_coord_trajectory,
                                                                                 file_path,
                                                                                 is_comma_separated)));
}

/**
 * NON-REAL-TIME!!!\n
 * Extract a set of trajectory(ies) from text file(s). Implementation is in derived classes.
 *
 * @param dir_or_file_path Two possibilities:\n
 *        If want to extract a set of trajectories, \n
 *        then specify the directory path containing the text files, \n
 *        each text file represents a single trajectory.
 *        If want to extract just a single trajectory, \n
 *        then specify the file path to the text file representing such trajectory.
 * @param trajectory_set (Blank/empty) set of trajectories to be filled-in
 * @param is_comma_separated If true, trajectory fields are separated by commas;\n
 *        If false, trajectory fields are separated by white-space characters (default is false)
 * @param max_num_trajs [optional] Maximum number of trajectories that could be loaded into a trajectory set (default is 500)
 * @return Success or failure
 */
bool CartesianCoordDMP::extractSetTrajectories(const char* dir_or_file_path,
                                               TrajectorySet& trajectory_set,
                                               bool is_comma_separated,
                                               uint max_num_trajs)
{
    return (rt_assert(data_logger_loader_discrete.extractSetCartCoordTrajectories(dir_or_file_path,
                                                                                  trajectory_set,
                                                                                  is_comma_separated,
                                                                                  max_num_trajs)));
}

/**
 * NON-REAL-TIME!!!\n
 * Write a set of Cartesian coordinate trajectories to text files contained in one folder/directory.
 *
 * @param cart_coord_trajectory_set Set of Cartesian coordinate trajectories to be written into the specified directory
 * @param dir_path Directory path where to write the set of Cartesian coordinate trajectories representation into
 * @param is_comma_separated If true, Cartesian coordinate trajectory fields will be written with comma-separation;\n
 *        If false, Cartesian coordinate trajectory fields will be written with white-space-separation (default is true)
 * @return Success or failure
 */
bool CartesianCoordDMP::writeSetCartCoordTrajectoriesToDir(const TrajectorySet& cart_coord_trajectory_set,
                                                           const char* dir_path,
                                                           bool is_comma_separated)
{
    return (rt_assert(data_logger_loader_discrete.writeSetCartCoordTrajectoriesToDir(cart_coord_trajectory_set,
                                                                                     dir_path,
                                                                                     is_comma_separated)));
}

/**
 * NEVER RUN IN REAL-TIME\n
 * Save the current parameters of this discrete DMP (weights and learning amplitude (A_learn)) into files.
 *
 * @param dir_path Directory path containing the files
 * @param file_name_weights Name of the file onto which the weights will be saved
 * @param file_name_A_learn Name of the file onto which the A_learn vector buffer will be saved
 * @param file_name_mean_start_position_global Name of the file onto which the mean start position w.r.t. global coordinate system will be saved
 * @param file_name_mean_goal_position_global Name of the file onto which the mean goal position w.r.t. global coordinate system will be saved
 * @param file_name_mean_tau Name of the file onto which the mean tau will be saved
 * @param file_name_mean_start_position_local Name of the file onto which the mean start position w.r.t. local coordinate system will be saved
 * @param file_name_mean_goal_position_local Name of the file onto which the mean goal position w.r.t. local coordinate system will be saved
 * @param file_name_ctraj_hmg_transform_local_to_global_matrix Name of the file onto which the homogeneous coordinate transform matrix from local to global coordinate systems will be saved
 * @param file_name_ctraj_hmg_transform_global_to_local_matrix Name of the file onto which the homogeneous coordinate transform matrix from global to local coordinate systems will be saved
 * @return Success or failure
 */
bool CartesianCoordDMP::saveParams(const char* dir_path,
                                   const char* file_name_weights,
                                   const char* file_name_A_learn,
                                   const char* file_name_mean_start_position_global,
                                   const char* file_name_mean_goal_position_global,
                                   const char* file_name_mean_tau,
                                   const char* file_name_mean_start_position_local,
                                   const char* file_name_mean_goal_position_local,
                                   const char* file_name_ctraj_hmg_transform_local_to_global_matrix,
                                   const char* file_name_ctraj_hmg_transform_global_to_local_matrix)
{
    return (rt_assert(DMPDiscrete::saveParams(dir_path,
                                              file_name_weights,
                                              file_name_A_learn,
                                              file_name_mean_start_position_global,
                                              file_name_mean_goal_position_global,
                                              file_name_mean_tau))  &&
            rt_assert(data_logger_loader_discrete.writeMatrixToFile(dir_path, file_name_mean_start_position_local, mean_start_local_position))    &&
            rt_assert(data_logger_loader_discrete.writeMatrixToFile(dir_path, file_name_mean_goal_position_local, mean_goal_local_position))      &&
            rt_assert(data_logger_loader_discrete.writeMatrixToFile(dir_path, file_name_mean_start_position_global, mean_start_global_position))    &&
            rt_assert(data_logger_loader_discrete.writeMatrixToFile(dir_path, file_name_mean_goal_position_global, mean_goal_global_position))      &&
            rt_assert(data_logger_loader_discrete.writeMatrixToFile(dir_path, file_name_ctraj_hmg_transform_local_to_global_matrix, ctraj_hmg_transform_local_to_global_matrix))    &&
            rt_assert(data_logger_loader_discrete.writeMatrixToFile(dir_path, file_name_ctraj_hmg_transform_global_to_local_matrix, ctraj_hmg_transform_global_to_local_matrix)));
}

//bool CartesianCoordDMP::loadParams(const char* dir_path,
//                                   const char* file_name_weights,
//                                   const char* file_name_A_learn,
//                                   const char* file_name_mean_start_position_global,
//                                   const char* file_name_mean_goal_position_global,
//                                   const char* file_name_mean_tau,
//                                   const char* file_name_mean_start_position_local,
//                                   const char* file_name_mean_goal_position_local,
//                                   const char* file_name_ctraj_hmg_transform_local_to_global_matrix,
//                                   const char* file_name_ctraj_hmg_transform_global_to_local_matrix)
//{
//    return (rt_assert(DMPDiscrete::loadParams(dir_path,
//                                              file_name_weights,
//                                              file_name_A_learn,
//                                              file_name_mean_start_position_global,
//                                              file_name_mean_goal_position_global,
//                                              file_name_mean_tau))  &&
//            rt_assert(data_logger_loader_discrete.readMatrixFromFile(dir_path, file_name_mean_start_position_local, mean_start_local_position))    &&
//            rt_assert(data_logger_loader_discrete.readMatrixFromFile(dir_path, file_name_mean_goal_position_local, mean_goal_local_position))      &&
//            rt_assert(data_logger_loader_discrete.readMatrixFromFile(dir_path, file_name_mean_start_position_global, mean_start_global_position))    &&
//            rt_assert(data_logger_loader_discrete.readMatrixFromFile(dir_path, file_name_mean_goal_position_global, mean_goal_global_position))      &&
//            rt_assert(data_logger_loader_discrete.readMatrixFromFile(dir_path, file_name_ctraj_hmg_transform_local_to_global_matrix, ctraj_hmg_transform_local_to_global_matrix))    &&
//            rt_assert(data_logger_loader_discrete.readMatrixFromFile(dir_path, file_name_ctraj_hmg_transform_global_to_local_matrix, ctraj_hmg_transform_global_to_local_matrix)));
//}

CartesianCoordDMP::~CartesianCoordDMP()
{}

}
