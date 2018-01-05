/*
 * CartesianCoordDMP.h
 *
 *  Class definition to deal with Cartesian Coordinate System (3D) DMP.
 *
 *  Created on: June 23, 2015
 *  Author: Giovanni Sutanto
 */

#ifndef CARTESIAN_COORD_DMP_H
#define CARTESIAN_COORD_DMP_H

#include <boost/filesystem.hpp>

#include "amd_clmc_dmp/dmp_discrete/DMPDiscrete.h"
#include "amd_clmc_dmp/cart_dmp/cart_coord_dmp/CartesianCoordTransformer.h"
#include "amd_clmc_dmp/dmp_discrete/TransformSystemDiscrete.h"

namespace dmp
{

class CartesianCoordDMP : public DMPDiscrete
{

protected:

    TransformSystemDiscrete     transform_sys_discrete_cart_coord;

    CartesianCoordTransformer   cart_coord_transformer;

    // Trajectory's relative homogeneous transformation matrix:
    // (a) from local to global coordinate system (or local as seen from global coordinate system)
    Matrix4x4                   ctraj_hmg_transform_local_to_global_matrix;
    // (b) from global to local coordinate system (or global as seen from local coordinate system) (inverse of ctraj_hmg_transform_matrix_global_to_local)
    Matrix4x4                   ctraj_hmg_transform_global_to_local_matrix;

    // 5 critical states of a Cartesian coordinate trajectory (continuously being updated/kept-track-of during trajectory reproduction/
    // unrolling by DMP):
    // start state, current state, current goal state, approaching-steady-state goal state, and
    // steady-state goal state:
    // (i)  on the global coordinate system
    Trajectory                  ctraj_critical_states_global_coord;
    // (ii) on the local coordinate system
    Trajectory                  ctraj_critical_states_local_coord;

    uint                        ctraj_local_coord_selection;

    Vector3                     mean_start_global_position; // average start position of the training set of trajectories, in global coordinate system
    Vector3                     mean_goal_global_position;  // average goal position of the training set of trajectories, in global coordinate system

    Vector3                     mean_start_local_position; // average start position of the training set of trajectories, in local coordinate system
    Vector3                     mean_goal_local_position;  // average goal position of the training set of trajectories, in local coordinate system

public:

    /**
     * NEVER RUN IN REAL-TIME, ONLY RUN IN INIT ROUTINE
     */
    CartesianCoordDMP();

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
    CartesianCoordDMP(CanonicalSystemDiscrete* canonical_system_discrete, uint model_size_init,
                      uint ts_formulation_type, uint learning_method,
                      RealTimeAssertor* real_time_assertor,
                      uint ctraj_local_coordinate_frame_selection=1,
                      const char* opt_data_directory_path="",
                      std::vector<TransformCoupling*>* transform_couplers=NULL);

    /**
     * Checks whether this Cartesian Coordinate DMP is valid or not.
     *
     * @return Cartesian Coordinate DMP is valid (true) or Cartesian Coordinate DMP is invalid (false)
     */
    bool isValid();

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
    bool preprocess(TrajectorySet& trajectory_set);

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
    bool learn(const char* training_data_dir_or_file_path,
               double robot_task_servo_rate,
               double* tau_learn=NULL,
               Trajectory* critical_states_learn=NULL);

    /**
     * NON-REAL-TIME!!!\n
     * Learns from a given set of trajectories.
     *
     * @param trajectory_set Set of trajectories to learn from\n
     *        (assuming all trajectories in this set are coming from the same trajectory distribution)
     * @param robot_task_servo_rate Robot's task servo rate
     * @return Success or failure
     */
    bool learn(const TrajectorySet& trajectory_set, double robot_task_servo_rate);

    /**
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
    bool learnMultiCartCoordDMPs(const char* in_data_dir_path, const char* out_data_dir_path,
                                 double robot_task_servo_rate, const char* in_data_subdir_subpath="",
                                 bool save_unroll_traj_dataset=false, double tau_unroll=1.0);

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
    bool start(const Trajectory& critical_states, double tau_init);

    using DMP::start;

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
    bool getNextState(double dt, bool update_canonical_state,
                      DMPState& result_dmp_state_global,
                      VectorN* transform_sys_forcing_term=NULL,
                      VectorN* transform_sys_coupling_term_acc=NULL,
                      VectorN* transform_sys_coupling_term_vel=NULL,
                      VectorM* func_approx_basis_functions=NULL,
                      VectorM* func_approx_normalized_basis_func_vector_mult_phase_multiplier=NULL);

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
    bool getNextState(double dt, bool update_canonical_state,
                      DMPState& result_dmp_state_global, DMPState& result_dmp_state_local,
                      VectorN* transform_sys_forcing_term=NULL,
                      VectorN* transform_sys_coupling_term_acc=NULL,
                      VectorN* transform_sys_coupling_term_vel=NULL,
                      VectorM* func_approx_basis_functions=NULL,
                      VectorM* func_approx_normalized_basis_func_vector_mult_phase_multiplier=NULL);

    /**
     * Returns current (global) DMP state.
     */
    DMPState getCurrentState();

    /**
     * Returns current goal position on x, y, and z axes on the global coordinate system.
     *
     * @param current_goal_global Current goal position on the global coordinate system (to be returned)
     * @return Success or failure
     */
    bool getCurrentGoalPosition(Vector3& current_goal_global);

    /**
     * Returns steady-state goal position vector (G) on the global coordinate system.
     *
     * @param Steady-state goal position vector (G) on the global coordinate system (to be returned)
     * @return Success or failure
     */
    bool getSteadyStateGoalPosition(Vector3& steady_state_goal_global);

    /**
     * Sets new steady-state goal position (G).
     *
     * @param new_G New steady-state goal position (G) to be set
     * @return Success or failure
     */
    bool setNewSteadyStateGoalPosition(const VectorN& new_G);

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
    bool setNewSteadyStateGoalPosition(const Vector3& new_G_global,
                                       const Vector3& new_approaching_G_global);

    /**
     * Convert DMPState representation from local coordinate system into global coordinate system.
     *
     * @param dmp_state_local DMPState representation in the local coordinate system
     * @param dmp_state_global DMPState representation in the global coordinate system
     * @return Success or failure
     */
    bool convertToGlobalDMPState(const DMPState& dmp_state_local, DMPState& dmp_state_global);

    /**
     * Convert DMPState representation from global coordinate system into local coordinate system.
     *
     * @param dmp_state_global DMPState representation in the global coordinate system
     * @param dmp_state_local DMPState representation in the local coordinate system
     * @return Success or failure
     */
    bool convertToLocalDMPState(const DMPState& dmp_state_global, DMPState& dmp_state_local);

    Matrix4x4 getHomogeneousTransformMatrixLocalToGlobal();

    Matrix4x4 getHomogeneousTransformMatrixGlobalToLocal();

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
    bool writeCartCoordTrajectoryToFile(const Trajectory& cart_coord_trajectory,
                                        const char* file_path,
                                        bool is_comma_separated=true);

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
    bool extractSetTrajectories(const char* dir_or_file_path,
                                TrajectorySet& trajectory_set,
                                bool is_comma_separated=false,
                                uint max_num_trajs=500);

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
    bool writeSetCartCoordTrajectoriesToDir(const TrajectorySet& cart_coord_trajectory_set,
                                            const char* dir_path,
                                            bool is_comma_separated=true);

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
    bool saveParams(const char* dir_path,
                    const char* file_name_weights="f_weights_matrix.txt",
                    const char* file_name_A_learn="f_A_learn_matrix.txt",
                    const char* file_name_mean_start_position_global="mean_start_position_global.txt",
                    const char* file_name_mean_goal_position_global="mean_goal_position_global.txt",
                    const char* file_name_mean_tau="mean_tau.txt",
                    const char* file_name_mean_start_position_local="mean_start_position_local.txt",
                    const char* file_name_mean_goal_position_local="mean_goal_position_local.txt",
                    const char* file_name_ctraj_hmg_transform_local_to_global_matrix="ctraj_hmg_transform_local_to_global_matrix.txt",
                    const char* file_name_ctraj_hmg_transform_global_to_local_matrix="ctraj_hmg_transform_global_to_local_matrix.txt");

//    bool loadParams(const char* dir_path,
//                    const char* file_name_weights="f_weights_matrix.txt",
//                    const char* file_name_A_learn="f_A_learn_matrix.txt",
//                    const char* file_name_mean_start_position_global="mean_start_position_global.txt",
//                    const char* file_name_mean_goal_position_global="mean_goal_position_global.txt",
//                    const char* file_name_mean_tau="mean_tau.txt",
//                    const char* file_name_mean_start_position_local="mean_start_position_local.txt",
//                    const char* file_name_mean_goal_position_local="mean_goal_position_local.txt",
//                    const char* file_name_ctraj_hmg_transform_local_to_global_matrix="ctraj_hmg_transform_local_to_global_matrix.txt",
//                    const char* file_name_ctraj_hmg_transform_global_to_local_matrix="ctraj_hmg_transform_global_to_local_matrix.txt");

    ~CartesianCoordDMP();

};
}
#endif
