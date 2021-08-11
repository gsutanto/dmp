/*
 * TransformCouplingLearnObsAvoid.h
 *
 *  Class defining transform coupling term for obstacle avoidance from learning by demonstration on a DMP.
 *
 *  See A. Rai, F. Meier, A. Ijspeert, and S. Schaal;
 *      Learning coupling terms for obstacle avoidance;
 *      IEEE-RAS International Conference on Humanoid Robotics, pp. 512-518, November 2014
 *
 *  Created on: Aug 28, 2015
 *  Author: Giovanni Sutanto
 */

#ifndef TRANSFORM_COUPLING_LEARN_OBS_AVOID_H
#define TRANSFORM_COUPLING_LEARN_OBS_AVOID_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <Eigen/Dense>

#include "dmp/utility/utility.h"
#include "dmp/dmp_state/DMPState.h"
#include "dmp/dmp_discrete/CanonicalSystemDiscrete.h"
#include "dmp/dmp_discrete/FuncApproximatorDiscrete.h"
#include "dmp/cart_dmp/cart_coord_dmp/CartesianCoordDMP.h"
#include "dmp/cart_dmp/cart_coord_dmp/CartesianCoordTransformer.h"
#include "dmp/dmp_coupling/base/TransformCoupling.h"
#include "dmp/dmp_coupling/learn_obs_avoid/DefinitionsLearnObsAvoid.h"
#include "dmp/dmp_coupling/learn_obs_avoid/LearnObsAvoidDataIO.h"
#include "dmp/dmp_coupling/learn_obs_avoid/TCLearnObsAvoidFeatureParameter.h"

namespace dmp
{

class TransformCouplingLearnObsAvoid : public TransformCoupling
{

private:

    CartesianCoordTransformer   cart_coord_transformer;
    LearnObsAvoidDataIO         loa_data_io;

    bool                        is_logging_learning_data;
    bool                        is_collecting_trajectory_data;
    bool                        is_constraining_Rp_yd_relationship;
    char                        data_directory_path[1000];

    // The following field(s) are written as RAW pointers because we just want to point
    // to other data structure(s) that might outlive the instance of this class:
    TCLearnObsAvoidFeatureParameter*    loa_feat_param;
    TauSystem*                          tau_sys;
    DMPState*                           endeff_cartesian_state_global;
    DMPStatePtr                         endeff_cartesian_state_local;
    ObstacleStates*                     point_obstacles_cartesian_state_global;
    ObstacleStatesPtr                   point_obstacles_cartesian_state_local;
    Matrix4x4*                          ctraj_hmg_transform_global_to_local_matrix;
    Vector3*                            goal_cartesian_position_global;
    Vector3                             goal_cartesian_position_local;
    VectorTPtr                          point_obstacles_cartesian_distances_from_endeff;
    FuncApproximatorDiscrete*           func_approx_discrete;  // No ownership.

public:

    /**
     * NON-REAL-TIME!!!
     */
    TransformCouplingLearnObsAvoid();

    /**
     * NON-REAL-TIME!!!\n
     *
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     * @param opt_is_logging_learning_data [optional] Option to save/log learning data (i.e. weights matrix, Ct_target, and Ct_fit) into a file (set to 'true' to save)
     * @param opt_data_directory_path [optional] Full directory path where to save/log all the data
     */
    TransformCouplingLearnObsAvoid(TCLearnObsAvoidFeatureParameter* loa_feature_parameter,
                                   TauSystem* tau_system,
                                   DMPState* endeff_cart_state_global,
                                   ObstacleStates* point_obstacles_cart_state_global,
                                   Matrix4x4* cart_traj_homogeneous_transform_global_to_local_matrix,
                                   RealTimeAssertor* real_time_assertor,
                                   bool opt_is_constraining_Rp_yd_relationship=false,
                                   bool opt_is_logging_learning_data=false,
                                   const char* opt_data_directory_path="",
                                   Vector3* goal_position_global=NULL,
                                   FuncApproximatorDiscrete* function_approximator_discrete=nullptr);

    /**
     * Checks whether this transformation coupling term for obstacle avoidance by demonstration is valid or not.
     *
     * @return Valid (true) or invalid (false)
     */
    bool isValid();

    /**
     * NON-REAL-TIME!!!\n
     * Learn coupling term for obstacle avoidance from demonstrated trajectories \n
     * (at baseline (during the absence of an obstacle) and during the presence of an obstacle).
     *
     * @param training_data_directory Directory where all training data is stored\n
     *        This directory must contain the following:\n
     *        (1) directory "baseline" (contains demonstrated trajectories information when there is NO obstacle),\n
     *        (2) numerical directories (start from "1", each contains demonstrated trajectories information \n
     *            when there is an obstacle at a specific position/setting,\n
     *            which is different between a numbered directory and the other)
     * @param robot_task_servo_rate Robot's task servo rate
     * @param cart_dmp_baseline_ptr Pointer to the BASELINE CartesianCoordDMP
     * @param regularization_const [optional] Regularization constant for the ridge regression
     * @param is_learning_baseline [optional] Also learning the baseline DMP forcing term?
     * @param max_num_trajs_per_setting [optional] Maximum number of trajectories that could be loaded into a trajectory set \n
     *                                  of an obstacle avoidance demonstration setting \n
     *                                  (setting = obstacle position) (default is 500)
     * @param selected_obs_avoid_setting_numbers [optional] If we are just interested in learning a subset of the obstacle avoidance demonstration settings,\n
     *                                           then specify the numbers (IDs) of the setting here (as a vector).\n
     *                                           If not specified, then all settings will be considered.
     * @param max_critical_point_distance_baseline_vs_obs_avoid_demo [optional] Maximum tolerable distance (in meter) \n
     *                                                               between start position in baseline and obstacle avoidance demonstrations,\n
     *                                                               as well as between goal position in baseline and obstacle avoidance demonstrations.
     * @return Success or failure
     */
    bool learnCouplingTerm(const char* training_data_directory,
                           double robot_task_servo_rate,
                           CartesianCoordDMP* cart_dmp_baseline_ptr,
                           double regularization_const=5.0,
                           bool is_learning_baseline=true,
                           uint max_num_trajs_per_setting=500,
                           std::vector<uint>* selected_obs_avoid_setting_numbers=NULL,
                           double max_critical_point_distance_baseline_vs_obs_avoid_demo=0.075);

    /**
     * NON-REAL-TIME!!!\n
     * If you want to maintain real-time performance,
     * do/call this function from a separate (non-real-time) thread!!!\n
     * Compute the sub-feature-matrix and the sub-target DMP coupling term (as a sub-block of a matrix) from a single demonstrated obstacle avoidance trajectory.\n
     * This will later be stacked together with those from other demonstrations, to become a huge feature matrix (usually called X matrix) and \n
     * a huge target DMP coupling term (target regression variable, usually a vector).
     *
     * @param demo_obs_avoid_traj_global A demonstrated obstacle avoidance trajectory for a particular obstacle setting, in Cartesian global coordinate system
     * @param point_obstacles_cart_state_traj_global Trajectory or vector of vectors of \n
     *                                               point obstacles' Cartesian state \n
     *                                               in the global coordinate system
     * @param robot_task_servo_rate Robot's task servo rate
     * @param cart_coord_dmp_critical_states Critical states for unrolling the Cartesian Coordinate DMP
     * @param cart_coord_dmp Cartesian Coordinate DMP that will be used to extract the sub-feature-matrix and the target coupling term
     * @param feature_vector_buffer Buffer for the feature vector, to be transferred to the sub-feature-matrix (so that we do not have to allocate memory in this function)
     * @param sub_feature_matrix The resulting sub-feature-matrix (to be returned)
     * @param sub_target_coupling_term The resulting target obstacle avoidance coupling term (as a sub-block of a matrix) (to be returned)
     * @param sub_feature_matrix_Rp_yd_constrained The resulting sub-feature-matrix when Rp*yd relationship is constrained (to be returned)
     * @param sub_target_coupling_term_Rp_yd_constrained The resulting target obstacle avoidance coupling term when Rp*yd relationship is constrained (to be returned)
     * @param max_critical_point_distance_baseline_vs_obs_avoid_demo [optional] Maximum tolerable distance (in meter) \n
     *                                                               between start position in baseline and obstacle avoidance demonstrations,\n
     *                                                               as well as between goal position in baseline and obstacle avoidance demonstrations.
     * @return Success or failure
     */
    bool computeSubFeatureMatrixAndSubTargetCouplingTerm(const Trajectory& demo_obs_avoid_traj_global,
                                                         const ObstacleStatesVector& point_obstacles_cart_state_traj_global,
                                                         const double& robot_task_servo_rate,
                                                         const Trajectory& cart_coord_dmp_critical_states,
                                                         CartesianCoordDMP& cart_coord_dmp,
                                                         VectorloaF3& feature_vector_buffer,
                                                         MatrixloaTxF3& sub_feature_matrix,
                                                         Matrix3xT& sub_target_coupling_term,
                                                         Matrixloa3TxF1* sub_feature_matrix_Rp_yd_constrained=NULL,
                                                         Vectorloa3T* sub_target_coupling_term_Rp_yd_constrained=NULL,
                                                         double max_critical_point_distance_baseline_vs_obs_avoid_demo=0.075);

    /**
     * This is for computing the stacked feature vector Phi, such that: \n
     * Ct = Gamma^T * Phi , where Gamma is the weight vector.\n
     * Inputs are represented in the global coordinate system.
     *
     * @param endeff_cart_state_global The end-effector Cartesian state in the global coordinate system
     * @param point_obstacles_cart_state_global Vector of point obstacles' Cartesian state in the global coordinate system
     * @param cart_hmg_transform_global_to_local_matrix The Cartesian homogeneous transformation matrix from global to local coordinate system
     * @param feature_vector The resulting stacked feature vector (Phi) (to be returned)
     * @param feature_matrix [optional] The original (unstacked) feature matrix (to be returned)
     * @return Success or failure
     */
    bool computeFeatures(const DMPState& endeff_cart_state_global,
                         const ObstacleStates& point_obstacles_cart_state_global,
                         const Matrix4x4& cart_hmg_transform_global_to_local_matrix,
                         VectorloaF3& feature_vector,
                         MatrixloaF1x3* feature_matrix=NULL);

    /**
     * This is for computing the stacked feature vector Phi, such that: \n
     * Ct = Gamma^T * Phi , where Gamma is the weight vector.\n
     * Inputs are represented in the local coordinate system.
     *
     * @param endeff_cart_state_local The end-effector Cartesian state in the local coordinate system
     * @param point_obstacles_cart_state_local Vector of point obstacles' Cartesian state in the local coordinate system
     * @param feature_vector The resulting stacked feature vector (Phi) (to be returned)
     * @param feature_matrix [optional] The original (unstacked) feature matrix (to be returned)
     * @return Success or failure
     */
    bool computeFeatures(const DMPState& endeff_cart_state_local,
                         const ObstacleStates& point_obstacles_cart_state_local,
                         VectorloaF3& feature_vector,
                         MatrixloaF1x3* feature_matrix=NULL);

    /**
     * In Rai et al. (Humanoids 2014) paper, this is for computing \n
     * the sub-feature vector (R*yd*phi), i.e. either (R*yd*phi1) or (R_p*yd*phi2), \n
     * which is needed for constructing the stacked feature vector Phi in computeFeatures() function.\n
     * (AF_H14 = Akshara-Franzi Humanoids'14: Learning Obstacle Avoidance Coupling Term paper)
     *
     * @param endeff_cart_state_local The end-effector Cartesian state in local coordinate system
     * @param obs_evalpt_cart_state_local The Cartesian state (in local coordinate system) of an evaluation point inside/on the obstacle geometry
     * @param feature_matrix_phi1_or_phi2_per_point The computed feature matrix (R*yd*phi1) or (R*yd*phi2) for the specified obstacle point (to be returned)
     * @param feature_matrix_phi3_per_point [optional] The computed feature matrix (R*yd*phi3) for the specified obstacle point (to be returned)
     * @return Success or failure
     */
    bool computeAF_H14ObsAvoidCtFeatureMatricesPerObsPoint(const DMPState& endeff_cart_state_local,
                                                           const DMPState& obs_evalpt_cart_state_local,
                                                           Matrixloa3xG2& feature_matrix_phi1_or_phi2_per_point,
                                                           Matrixloa3xG* feature_matrix_phi3_per_point=NULL);

    /**
     * In Park et al. (Humanoids 2008) paper, this is for computing \n
     * an improved version of the dynamic obstacle avoidance model presented there (designed by Giovanni Sutanto), \n
     * i.e. the sub-feature vector, which is needed for constructing the stacked feature vector Phi \n
     * in computeFeatures() function.
     *
     * @param endeff_cart_state_local The end-effector Cartesian state in local coordinate system
     * @param obs_evalpt_cart_state_local The Cartesian state (in local coordinate system) of an evaluation point inside/on the obstacle geometry
     * @param feature_matrix_per_point The computed feature matrix for the specified obstacle point (to be returned)
     * @return Success or failure
     */
    bool computePF_DYN2ObsAvoidCtFeatureMatricesPerObsPoint(const DMPState& endeff_cart_state_local,
                                                            const DMPState& obs_evalpt_cart_state_local,
                                                            Matrixloa3xG2& feature_matrix_per_point);

    /**
     * This is for computing the kernelized general features (KGF), which is a product of \n
     * 3 kernels: 1 kernel as a function of theta_p, \n
     *            1 kernel as a function of d_p, and \n
     *            1 kernel as a function of the phase variable s (from canonical system discrete)\n
     * (designed by Giovanni Sutanto), \n
     * i.e. the sub-feature vector, which is needed for constructing the stacked feature vector Phi \n
     * in computeFeatures() function.
     *
     * @param endeff_cart_state_local The end-effector Cartesian state in local coordinate system
     * @param obs_evalpt_cart_state_local The Cartesian state (in local coordinate system) of an evaluation point inside/on the obstacle geometry
     * @param feature_matrix_per_point The computed feature matrix for the specified obstacle point (to be returned)
     * @return Success or failure
     */
    bool computeKGFObsAvoidCtFeatureMatricesPerObsPoint(const DMPState& endeff_cart_state_local,
                                                        const DMPState& obs_evalpt_cart_state_local,
                                                        Matrixloa3xG3& feature_matrix_per_point);

    bool computeNNObsAvoidCtFeature(const DMPState& endeff_cart_state_local,
                                    const ObstacleStates& point_obstacles_cart_state_local,
                                    VectorloaF3& feature_vector);

    bool computeNNInference(const VectorloaF3& feature_vector,
                            VectorloaNN_O& NN_output);

    /**
     * Returns both acceleration-level coupling value (C_t_{Acc}) and
     * velocity-level coupling value (C_t_{Vel}) that will be used in the transformation system
     * to adjust motion of a DMP.
     *
     * @param ct_acc Acceleration-level coupling value (C_t_{Acc}) that will be used to adjust the acceleration of DMP transformation system
     * @param ct_vel Velocity-level coupling value (C_t_{Vel}) that will be used to adjust the velocity of DMP transformation system
     * @return Success or failure
     */
    bool getValue(VectorN& ct_acc, VectorN& ct_vel);

    /**
     * This is a special case where we are provided with the information of the sphere obstacle's
     * center and radius, and we want to compute the points that will be considered for
     * obstacle avoidance features computation.
     *
     * @param endeff_cart_state_global The end-effector Cartesian state in the global coordinate system
     * @param obssphctr_cart_state_global The obstacle's center Cartesian state in the global coordinate system
     * @param obs_sphere_radius Radius of the sphere obstacle
     * @param point_obstacles_cart_state_global Vector of point obstacles' Cartesian state in the global coordinate system (to be returned)
     * @return Success or failure
     */
    bool computePointObstaclesCartStateGlobalFromStaticSphereObstacle(const DMPState& endeff_cart_state_global,
                                                                      const DMPState& obssphctr_cart_state_global,
                                                                      const double& obs_sphere_radius,
                                                                      ObstacleStates& point_obstacles_cart_state_global);

    /**
     * NON-REAL-TIME!!!\n
     * This is a special case as well, similar to computePointObstaclesCartStateGlobalFromStaticSphereObstacle(),
     * but this time is with regard to an evaluation trajectory (endeff_cart_state_traj_global),
     * instead of an evaluation state. Please note that even though the sphere obstacle is static,
     * the closest point on the surface of the sphere obstacle is dynamic, i.e. changing as
     * the evaluation state (a state in endeff_cart_state_traj_global) changes.
     * This function is called from inside TransformCouplingLearnObsAvoid::learnCouplingTerm() function.
     *
     * @param endeff_cart_state_traj_global The end-effector Cartesian state trajectory in the global coordinate system
     * @param obssphctr_cart_state_global The obstacle's center Cartesian state in the global coordinate system
     * @param obs_sphere_radius Radius of the sphere obstacle
     * @param point_obstacles_cart_state_traj_global Trajectory or vector of vectors of point obstacles' Cartesian state \n
     *                                               in the global coordinate system (to be returned)
     * @return Success or failure
     */
    bool computePointObstaclesCartStateGlobalTrajectoryFromStaticSphereObstacle(const Trajectory& endeff_cart_state_traj_global,
                                                                                const DMPState& obssphctr_cart_state_global,
                                                                                const double& obs_sphere_radius,
                                                                                ObstacleStatesVector& point_obstacles_cart_state_traj_global);

    /**
     * This is simply a conversion function from a vector of position (point_obstacles_cart_position_global)
     * into a vector of state (point_obstacles_cart_state_global)
     *
     * @param point_obstacles_cart_position_global The point obstacles' Cartesian coordinate in the global coordinate system
     * @param point_obstacles_cart_state_global The point obstacles' Cartesian state in the global coordinate system (to be returned)
     * @return Success or failure
     */
    bool computePointObstaclesCartStateGlobalFromStaticViconData(const Points& point_obstacles_cart_position_global,
                                                                 ObstacleStates& point_obstacles_cart_state_global);

    /**
     * NON-REAL-TIME!!!\n
     * This is a special case as well, similar to computePointObstaclesCartStateGlobalFromStaticSphereObstacle(),
     * but this time is for point obstacles processed from Vicon data.
     * Because the point obstacles are static, the point_obstacles_cart_state_traj_global only
     * has size 1 (1 state), which applies to all end-effector states in the end-effector trajectory.
     * This function is called from inside TransformCouplingLearnObsAvoid::learnCouplingTerm() function.
     *
     * @param point_obstacles_cart_position_global The point obstacles' Cartesian coordinate in the global coordinate system
     * @param point_obstacles_cart_state_traj_global Trajectory or vector of vectors of point obstacles' Cartesian state \n
     *                                               in the global coordinate system (to be returned)
     * @return Success or failure
     */
    bool computePointObstaclesCartStateGlobalTrajectoryFromStaticViconData(const Points& point_obstacles_cart_position_global,
                                                                           ObstacleStatesVector& point_obstacles_cart_state_traj_global);

    bool startCollectingTrajectoryDataSet();

    void stopCollectingTrajectoryDataSet();

    /**
     * NON-REAL-TIME!!!\n
     * If you want to maintain real-time performance,
     * do/call this function from a separate (non-real-time) thread!!!
     *
     * @param new_data_dir_path New data directory path \n
     *                          (if we want to save the data in a directory path other than
     *                           the data_directory_path)
     * @return Success or failure
     */
    bool saveTrajectoryDataSet(const char* new_data_dir_path="");

    bool saveWeights(const char* dir_path, const char* file_name);

    bool saveWeights(const char* file_path);

    bool loadWeights(const char* dir_path, const char* file_name);

    bool loadWeights(const char* file_path);

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
                                                   std::vector<uint>* selected_obs_avoid_setting_numbers=NULL,
                                                   VecVecDMPUnrollInitParams* demo_group_set_dmp_unroll_init_params=NULL);

    ~TransformCouplingLearnObsAvoid();

};
}
#endif
