#include "dmp/dmp_coupling/learn_obs_avoid/LearnObsAvoidDataIO.h"

namespace dmp
{

LearnObsAvoidDataIO::LearnObsAvoidDataIO():
    DataIO(), feature_trajectory_buffer(new MatrixloaTxF3()),
    endeff_cart_position_local_trajectory_buffer(new MatrixTx3()),
    endeff_cart_velocity_local_trajectory_buffer(new MatrixTx3()),
    obstacle_cart_position_local_trajectory_buffer(new MatrixTx3()),
    obstacle_cart_velocity_local_trajectory_buffer(new MatrixTx3()),
    save_data_buffer(new MatrixloaTxF3())
{
    LearnObsAvoidDataIO::reset();
}

/**
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 */
LearnObsAvoidDataIO::LearnObsAvoidDataIO(RealTimeAssertor* real_time_assertor):
    DataIO(real_time_assertor), feature_trajectory_buffer(new MatrixloaTxF3()),
    endeff_cart_position_local_trajectory_buffer(new MatrixTx3()),
    endeff_cart_velocity_local_trajectory_buffer(new MatrixTx3()),
    obstacle_cart_position_local_trajectory_buffer(new MatrixTx3()),
    obstacle_cart_velocity_local_trajectory_buffer(new MatrixTx3()),
    save_data_buffer(new MatrixloaTxF3())
{
    LearnObsAvoidDataIO::reset();
}

/**
 * Initialize/reset the data buffers and counters (usually done everytime before start of data saving).
 *
 * @return Success or failure
 */
bool LearnObsAvoidDataIO::reset()
{
    if (rt_assert((rt_assert(resizeAndReset(feature_trajectory_buffer,
                                            MAX_TRAJ_SIZE,
                                            MAX_LOA_NUM_FEATURE_VECTOR_SIZE)))  &&
                  (rt_assert(resizeAndReset(endeff_cart_position_local_trajectory_buffer,
                                            MAX_TRAJ_SIZE,  3)))        &&
                  (rt_assert(resizeAndReset(endeff_cart_velocity_local_trajectory_buffer,
                                            MAX_TRAJ_SIZE, 3)))         &&
                  (rt_assert(resizeAndReset(obstacle_cart_position_local_trajectory_buffer,
                                            MAX_TRAJ_SIZE, 3)))         &&
                  (rt_assert(resizeAndReset(obstacle_cart_velocity_local_trajectory_buffer,
                                            MAX_TRAJ_SIZE, 3)))) == false)
    {
        return false;
    }

    trajectory_step_count   = 0;

    return true;
}

/**
 * Checks whether this data I/O is valid or not.
 *
 * @return LearnObsAvoid data I/O is valid (true) or invalid (false)
 */
bool LearnObsAvoidDataIO::isValid()
{
    if (rt_assert(DataIO::isValid()) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(feature_trajectory_buffer                         != NULL)) &&
                  (rt_assert(endeff_cart_position_local_trajectory_buffer      != NULL)) &&
                  (rt_assert(endeff_cart_velocity_local_trajectory_buffer      != NULL)) &&
                  (rt_assert(obstacle_cart_position_local_trajectory_buffer    != NULL)) &&
                  (rt_assert(obstacle_cart_velocity_local_trajectory_buffer    != NULL))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(feature_vector_size > 0)) &&
                  (rt_assert(feature_vector_size <= MAX_LOA_NUM_FEATURE_VECTOR_SIZE))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(trajectory_step_count >= 0)) &&
                  (rt_assert(trajectory_step_count <= MAX_TRAJ_SIZE))) == false)
    {
        return false;
    }
    return true;
}

/**
 * NON-REAL-TIME!!!\n
 * If you want to maintain real-time performance,
 * do/call this function from a separate (non-real-time) thread!!!\n
 * Extract all DMP obstacle avoidance coupling term training data information for sphere obstacles:\n
 * demonstrations, sphere centers, radius in global coordinate system.
 *
 * @param in_data_dir_path The input data directory
 * @param demo_group_set_obs_avoid_global [to-be-filled/returned] Extracted obstacle avoidance demonstrations \n
 *                                        in the Cartesian global coordinate system
 * @param obs_sph_centers_cart_position_global [to-be-filled/returned] Set of center point coordinates of \n
 *                                             the spherical obstacles, \n
 *                                             in Cartesian global coordinate system, \n
 *                                             corresponding to a set of demonstrations
 * @param obs_sph_radius [to-be-filled/returned] Set of radius of the spherical obstacles, corresponding to a set of demonstrations
 * @param N_demo_settings [to-be-filled/returned] Total number of different obstacle avoidance demonstration settings \n
 *                        (or total number of different obstacle positions)
 * @param max_num_trajs_per_setting [optional] Maximum number of trajectories that could be loaded into a trajectory set \n
 *                                  of an obstacle avoidance demonstration setting \n
 *                                  (setting = obstacle position) (default is 500)
 * @param selected_obs_avoid_setting_numbers [optional] If we are just interested in learning a subset of the obstacle avoidance demonstration settings,\n
 *                                           then specify the numbers (IDs) of the setting here (as a vector).\n
 *                                           If not specified, then all settings will be considered.
 * @param demo_group_set_dmp_unroll_init_params [optional] DMP unroll initialization parameters of each trajectory in \n
 *                                              the extracted demonstrations (to be returned)
 * @return Success or failure
 */
bool LearnObsAvoidDataIO::extractObstacleAvoidanceTrainingDataSphObs(const char* in_data_dir_path,
                                                                     DemonstrationGroupSet& demo_group_set_obs_avoid_global,
                                                                     Points& obs_sph_centers_cart_position_global,
                                                                     DoubleVector& obs_sph_radius,
                                                                     uint& N_demo_settings,
                                                                     uint max_num_trajs_per_setting,
                                                                     std::vector<uint>* selected_obs_avoid_setting_numbers,
                                                                     VecVecDMPUnrollInitParams* demo_group_set_dmp_unroll_init_params)
{
    if (rt_assert(DataIO::extractSetCartCoordDemonstrationGroups(in_data_dir_path,
                                                                 demo_group_set_obs_avoid_global,
                                                                 N_demo_settings,
                                                                 max_num_trajs_per_setting,
                                                                 selected_obs_avoid_setting_numbers,
                                                                 demo_group_set_dmp_unroll_init_params)) == false)
    {
        return false;
    }

    char    obs_setting_dir_path[1000];

    obs_sph_centers_cart_position_global.resize(N_demo_settings);
    obs_sph_radius.resize(N_demo_settings);
    if (rt_assert(rt_assert(obs_sph_centers_cart_position_global.size() == N_demo_settings) &&
                  rt_assert(obs_sph_radius.size() == N_demo_settings)) == false)
    {
        return false;
    }

    for (uint n=1; n<=N_demo_settings; n++)
    {
        if (selected_obs_avoid_setting_numbers == NULL) // use all settings
        {
            sprintf(obs_setting_dir_path, "%s/%u/", in_data_dir_path, n);
        }
        else
        {
            sprintf(obs_setting_dir_path, "%s/%u/", in_data_dir_path, (*selected_obs_avoid_setting_numbers)[n-1]);
        }
        // Load sphere obstacle parameters of the demonstration setting:
        if (rt_assert(LearnObsAvoidDataIO::readSphereObstacleParametersFromFile(obs_setting_dir_path,
                                                                                obs_sph_centers_cart_position_global[n-1],
                                                                                obs_sph_radius[n-1])) == false)
        {
            return false;
        }
    }

    return true;
}

/**
 * NON-REAL-TIME!!!\n
 * If you want to maintain real-time performance,
 * do/call this function from a separate (non-real-time) thread!!!\n
 * Extract all DMP obstacle avoidance coupling term training data information \n
 * for point obstacles represented by Vicon markers: \n
 * demonstrations and point obstacles' coordinate in the global coordinate system.
 *
 * @param in_data_dir_path The input data directory
 * @param demo_group_set_obs_avoid_global [to-be-filled/returned] Extracted obstacle avoidance demonstrations \n
 *                                        in the Cartesian global coordinate system
 * @param vector_point_obstacles_cart_position_global [to-be-filled/returned] Vector of point obstacles' coordinates \n
 *                                                    in Cartesian global coordinate system, \n
 *                                                    corresponding to a set of demonstrations
 * @param N_demo_settings [to-be-filled/returned] Total number of different obstacle avoidance demonstration settings \n
 *                        (or total number of different obstacle positions)
 * @param max_num_trajs_per_setting [optional] Maximum number of trajectories that could be loaded into a trajectory set \n
 *                                  of an obstacle avoidance demonstration setting \n
 *                                  (setting = obstacle position) (default is 500)
 * @param selected_obs_avoid_setting_numbers [optional] If we are just interested in learning a subset of the obstacle avoidance demonstration settings,\n
 *                                           then specify the numbers (IDs) of the setting here (as a vector).\n
 *                                           If not specified, then all settings will be considered.
 * @param demo_group_set_dmp_unroll_init_params [optional] DMP unroll initialization parameters of each trajectory in \n
 *                                              the extracted demonstrations (to be returned)
 * @return Success or failure
 */
bool LearnObsAvoidDataIO::extractObstacleAvoidanceTrainingDataVicon(const char* in_data_dir_path,
                                                                    DemonstrationGroupSet& demo_group_set_obs_avoid_global,
                                                                    PointsVector& vector_point_obstacles_cart_position_global,
                                                                    uint& N_demo_settings,
                                                                    uint max_num_trajs_per_setting,
                                                                    std::vector<uint>* selected_obs_avoid_setting_numbers,
                                                                    VecVecDMPUnrollInitParams* demo_group_set_dmp_unroll_init_params)
{
    if (rt_assert(DataIO::extractSetCartCoordDemonstrationGroups(in_data_dir_path,
                                                                 demo_group_set_obs_avoid_global,
                                                                 N_demo_settings,
                                                                 max_num_trajs_per_setting,
                                                                 selected_obs_avoid_setting_numbers,
                                                                 demo_group_set_dmp_unroll_init_params)) == false)
    {
        return false;
    }

    char    obs_setting_dir_path[1000];

    vector_point_obstacles_cart_position_global.resize(N_demo_settings);
    if (rt_assert(vector_point_obstacles_cart_position_global.size() == N_demo_settings) == false)
    {
        return false;
    }

    for (uint n=1; n<=N_demo_settings; n++)
    {
        if (selected_obs_avoid_setting_numbers == NULL) // use all settings
        {
            sprintf(obs_setting_dir_path, "%s/%u/", in_data_dir_path, n);
        }
        else
        {
            sprintf(obs_setting_dir_path, "%s/%u/", in_data_dir_path, (*selected_obs_avoid_setting_numbers)[n-1]);
        }
        // Load point obstacles' (global) coordinates of the demonstration setting:
        if (rt_assert(LearnObsAvoidDataIO::readPointObstaclesFromFile(obs_setting_dir_path,
                                                                      vector_point_obstacles_cart_position_global[n-1])) == false)
        {
            return false;
        }
    }

    return true;
}

uint LearnObsAvoidDataIO::getFeatureVectorSize()
{
    return (feature_vector_size);
}

bool LearnObsAvoidDataIO::setFeatureVectorSize(uint new_feature_vector_size)
{
    feature_vector_size   = new_feature_vector_size;
    feature_trajectory_buffer->resize(MAX_TRAJ_SIZE, feature_vector_size);
    return true;
}

/**
 * Copy the supplied vectors into current column of the buffers.
 *
 * @param feature_vector Feature vector that will be copied/saved
 * @param endeff_cart_state_local End-effector local Cartesian state that will be copied/saved
 * @param obstacle_cart_state_local Obstacle local Cartesian state that will be copied/saved
 * @return Success or failure
 */
bool LearnObsAvoidDataIO::collectTrajectoryDataSet(const VectorloaF3& feature_vector,
                                                   const DMPState& endeff_cart_state_local,
                                                   const DMPState& obstacle_cart_state_local)
{
    // pre-conditions checking
    if (rt_assert(LearnObsAvoidDataIO::isValid()) == false)
    {
        return false;
    }
    // trajectory_step_count must be STRICTLY less than MAX_TRAJ_SIZE,
    // otherwise segmentation fault could happen:
    if (rt_assert(trajectory_step_count < MAX_TRAJ_SIZE) == false)
    {
        return false;
    }
    // input checking
    if (rt_assert(feature_vector.rows() == feature_vector_size) == false)
    {
        return false;
    }

    feature_trajectory_buffer->block(trajectory_step_count,0,1,feature_vector_size)           = feature_vector.transpose();
    endeff_cart_position_local_trajectory_buffer->block(trajectory_step_count,0,1,3)   = endeff_cart_state_local.getX().block(0,0,3,1).transpose();
    endeff_cart_velocity_local_trajectory_buffer->block(trajectory_step_count,0,1,3)   = endeff_cart_state_local.getXd().block(0,0,3,1).transpose();
    obstacle_cart_position_local_trajectory_buffer->block(trajectory_step_count,0,1,3) = obstacle_cart_state_local.getX().block(0,0,3,1).transpose();
    obstacle_cart_velocity_local_trajectory_buffer->block(trajectory_step_count,0,1,3) = obstacle_cart_state_local.getXd().block(0,0,3,1).transpose();
    trajectory_step_count++;

    // post-conditions checking
    if (rt_assert(trajectory_step_count <= MAX_TRAJ_SIZE) == false)
    {
        return false;
    }

    return true;
}

/**
 * NON-REAL-TIME!!!\n
 * If you want to maintain real-time performance,
 * do/call this function from a separate (non-real-time) thread!!!
 *
 * @param data_directory Specified directory to save the data files
 * @return Success or failure
 */
bool LearnObsAvoidDataIO::saveTrajectoryDataSet(const char* data_directory)
{
    // pre-conditions checking
    if (rt_assert(LearnObsAvoidDataIO::isValid()) == false)
    {
        return false;
    }

    if (rt_assert(DataIO::saveTrajectoryData(data_directory, "feature_trajectory.txt",
                                             trajectory_step_count, feature_vector_size,
                                             *save_data_buffer, *feature_trajectory_buffer)) == false)
    {
        return false;
    }
    if (rt_assert(DataIO::saveTrajectoryData(data_directory, "endeff_cart_position_local_trajectory.txt",
                                             trajectory_step_count, 3, *save_data_buffer,
                                             *endeff_cart_position_local_trajectory_buffer)) == false)
    {
        return false;
    }
    if (rt_assert(DataIO::saveTrajectoryData(data_directory, "endeff_cart_velocity_local_trajectory.txt",
                                             trajectory_step_count, 3, *save_data_buffer,
                                             *endeff_cart_velocity_local_trajectory_buffer)) == false)
    {
        return false;
    }
    if (rt_assert(DataIO::saveTrajectoryData(data_directory, "obstacle_cart_position_local_trajectory.txt",
                                             trajectory_step_count, 3, *save_data_buffer,
                                             *obstacle_cart_position_local_trajectory_buffer)) == false)
    {
        return false;
    }
    if (rt_assert(DataIO::saveTrajectoryData(data_directory, "obstacle_cart_velocity_local_trajectory.txt",
                                             trajectory_step_count, 3, *save_data_buffer,
                                             *obstacle_cart_velocity_local_trajectory_buffer)) == false)
    {
        return false;
    }

    return true;
}

/**
 * NON-REAL-TIME!!!\n
 * If you want to maintain real-time performance,
 * do/call this function from a separate (non-real-time) thread!!!\n
 * Read sphere obstacle parameters from files in a specified directory.
 *
 * @param dir_path Directory path from where the data files will be read
 * @param obsctr_cart_position_global Center point coordinate of the obstacle geometry, in Cartesian global coordinate system
 * @param radius Radius of the sphere obstacle
 * @return Success or failure
 */
bool LearnObsAvoidDataIO::readSphereObstacleParametersFromFile(const char* dir_path,
                                                               Vector3& obsctr_cart_position_global,
                                                               double& radius)
{
     // Load obstacle coordinate of the demonstration setting:
    if (rt_assert(LearnObsAvoidDataIO::readMatrixFromFile(dir_path,
                                                          "obs_sph_center_coord.txt",
                                                          obsctr_cart_position_global)) == false)
    {
        return false;
    }
    // Load the sphere obstacle radius of the demonstration setting:
    if (rt_assert(LearnObsAvoidDataIO::readMatrixFromFile(dir_path,
                                                          "obs_sph_radius.txt",
                                                          radius)) == false)
    {
        return false;
    }

    return true;
}

/**
 * NON-REAL-TIME!!!\n
 * If you want to maintain real-time performance,
 * do/call this function from a separate (non-real-time) thread!!!\n
 * Read point obstacles coordinates from file(s) in a specified directory.
 *
 * @param dir_path Directory path from where the data files will be read
 * @param point_obstacles_cart_position_global Vector of point obstacles' coordinates in Cartesian global coordinate system
 * @return Success or failure
 */
bool LearnObsAvoidDataIO::readPointObstaclesFromFile(const char* dir_path,
                                                     Points& point_obstacles_cart_position_global)
{
    char    file_path[1000];
    sprintf(file_path, "%s/%s", dir_path, "obs_markers_global_coord.txt");

    // Load point obstacles' coordinates of the demonstration setting:
    if (rt_assert(LearnObsAvoidDataIO::extract3DPoints(file_path,
                                                       point_obstacles_cart_position_global,
                                                       false)) == false)
    {
        return false;
    }

    return true;
}

LearnObsAvoidDataIO::~LearnObsAvoidDataIO()
{}

}
