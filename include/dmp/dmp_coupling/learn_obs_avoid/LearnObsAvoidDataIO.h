/*
 * LearnObsAvoidDataIO.h
 *

 *  Created on: Dec 11, 2015
 *  Author: Giovanni Sutanto
 */


#ifndef LEARN_OBS_AVOID_DATA_IO_H
#define LEARN_OBS_AVOID_DATA_IO_H

#include <stdio.h>
#include <vector>
#include "dmp/utility/DataIO.h"
#include "dmp/dmp_coupling/learn_obs_avoid/DefinitionsLearnObsAvoid.h"

namespace dmp
{

class LearnObsAvoidDataIO : public DataIO
{

protected:

    // The following data structures are encapsulated in std::unique_ptr
    // to make sure this data structure is allocated in the heap
    MatrixloaTxF3Ptr    feature_trajectory_buffer;
    MatrixTx3Ptr        endeff_cart_position_local_trajectory_buffer;
    MatrixTx3Ptr        endeff_cart_velocity_local_trajectory_buffer;
    MatrixTx3Ptr        obstacle_cart_position_local_trajectory_buffer;
    MatrixTx3Ptr        obstacle_cart_velocity_local_trajectory_buffer;
    MatrixloaTxF3Ptr    save_data_buffer;

    uint                feature_vector_size;    // length of learn_obs_avoid's feature vector
    uint                trajectory_step_count;  // counting number of states so far in the trajectory

public:

    LearnObsAvoidDataIO();

    /**
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    LearnObsAvoidDataIO(RealTimeAssertor* real_time_assertor);

    /**
     * Initialize/reset the data buffers and counters (usually done everytime before start of data saving).
     *
     * @return Success or failure
     */
    bool reset();

    /**
     * Checks whether this data I/O is valid or not.
     *
     * @return LearnObsAvoid data I/O is valid (true) or invalid (false)
     */
    bool isValid();

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
    bool extractObstacleAvoidanceTrainingDataSphObs(const char* in_data_dir_path,
                                                    DemonstrationGroupSet& demo_group_set_obs_avoid_global,
                                                    Points& obs_sph_centers_cart_position_global,
                                                    DoubleVector& obs_sph_radius,
                                                    uint& N_demo_settings,
                                                    uint max_num_trajs_per_setting=500,
                                                    std::vector<uint>* selected_obs_avoid_setting_numbers=nullptr,
                                                    VecVecDMPUnrollInitParams* demo_group_set_dmp_unroll_init_params=nullptr);

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
    bool extractObstacleAvoidanceTrainingDataVicon(const char* in_data_dir_path,
                                                   DemonstrationGroupSet& demo_group_set_obs_avoid_global,
                                                   PointsVector& vector_point_obstacles_cart_position_global,
                                                   uint& N_demo_settings,
                                                   uint max_num_trajs_per_setting=500,
                                                   std::vector<uint>* selected_obs_avoid_setting_numbers=nullptr,
                                                   VecVecDMPUnrollInitParams* demo_group_set_dmp_unroll_init_params=nullptr);

    uint getFeatureVectorSize();

    bool setFeatureVectorSize(uint new_feature_vector_size);

    /**
     * Copy the supplied vectors into current column of the buffers.
     *
     * @param feature_vector Feature vector that will be copied/saved
     * @param endeff_cart_state_local End-effector local Cartesian state that will be copied/saved
     * @param obstacle_cart_state_local Obstacle local Cartesian state that will be copied/saved
     * @return Success or failure
     */
    bool collectTrajectoryDataSet(const VectorloaF3& feature_vector,
                                  const DMPState& endeff_cart_state_local,
                                  const DMPState& obstacle_cart_state_local);

    /**
     * NON-REAL-TIME!!!\n
     * If you want to maintain real-time performance,
     * do/call this function from a separate (non-real-time) thread!!!
     *
     * @param data_directory Specified directory to save the data files
     * @return Success or failure
     */
    bool saveTrajectoryDataSet(const char* data_directory);

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
    bool readSphereObstacleParametersFromFile(const char* dir_path,
                                              Vector3& obsctr_cart_position_global,
                                              double& radius);

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
    bool readPointObstaclesFromFile(const char* dir_path,
                                    Points& point_obstacles_cart_position_global);

    ~LearnObsAvoidDataIO();

};

}
#endif
