/*
 * DMP.h
 *
 *  This implements a general structure DMP that contains
 *  CanonicalSystem, FunctionApproximator, TransformationSystem, LearningSystem,
 * and external couplings
 *
 *  Created on: Dec 12, 2014
 *  Author: Yevgen Chebotar and Giovanni Sutanto
 */

#ifndef DMP_H
#define DMP_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "dmp/dmp_base/CanonicalSystem.h"
#include "dmp/dmp_base/DMPDataIO.h"
#include "dmp/dmp_base/FunctionApproximator.h"
#include "dmp/dmp_base/LearningSystem.h"
#include "dmp/dmp_base/TransformationSystem.h"
#include "dmp/dmp_param/DMPUnrollInitParams.h"
#include "dmp/dmp_param/TauSystem.h"
#include "dmp/dmp_state/DMPState.h"
#include "dmp/utility/DefinitionsDerived.h"

namespace dmp {

class DMP {
 protected:
  uint dmp_num_dimensions;
  uint model_size;

  bool is_started;

  TauSystem* tau_sys;
  CanonicalSystem* canonical_sys;
  TransformationSystem* transform_sys;
  LearningSystem* learning_sys;
  DMPDataIO* data_logger_loader;

  LoggedDMPVariables* logged_dmp_variables;

  VectorN mean_start_position;  // average start position of the training set of
                                // trajectories
  VectorN mean_goal_position;   // average goal position of the training set of
                                // trajectories
  double mean_tau;  // average tau of the training set of trajectories

  char data_directory_path[1000];
  bool is_collecting_trajectory_data;

  RealTimeAssertor* rt_assertor;

 public:
  DMP();

  /**
   * @param dmp_num_dimensions_init Number of trajectory dimensions employed in
   * this DMP
   * @param model_size_init Size of the model (M: number of basis functions or
   * others) used to represent the function
   * @param canonical_system Canonical system that drives this DMP
   * @param transform_system Transformation system used in this DMP
   * @param learning_system Learning system for learning the function
   * approximator's parameters
   * @param dmp_data_io Data I/O (or logger/loader) for saving and loading
   * into/from file(s)
   * @param logged_dmp_vars (Pointer to the) buffer for logged DMP variables
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   * @param opt_data_directory_path [optional] Full directory path where to
   * save/log all the data
   */
  DMP(uint dmp_num_dimensions_init, uint model_size_init,
      CanonicalSystem* canonical_system, TransformationSystem* transform_system,
      LearningSystem* learning_system, DMPDataIO* dmp_data_io,
      LoggedDMPVariables* logged_dmp_vars, RealTimeAssertor* real_time_assertor,
      const char* opt_data_directory_path = "");

  /**
   * Checks whether this DMP is valid or not.
   *
   * @return DMP is valid (true) or DMP is invalid (false)
   */
  virtual bool isValid();

  /**
   * NON-REAL-TIME!!!\n
   * Extract a set of trajectory(ies) from text file(s). Implementation is in
   * derived classes.
   *
   * @param dir_or_file_path Two possibilities:\n
   *        If want to extract a set of trajectories, \n
   *        then specify the directory path containing the text files, \n
   *        each text file represents a single trajectory.
   *        If want to extract just a single trajectory, \n
   *        then specify the file path to the text file representing such
   * trajectory.
   * @param trajectory_set (Blank/empty) set of trajectories to be filled-in
   * @param is_comma_separated If true, trajectory fields are separated by
   * commas;\n If false, trajectory fields are separated by white-space
   * characters (default is false)
   * @param max_num_trajs [optional] Maximum number of trajectories that could
   * be loaded into a trajectory set (default is 500)
   * @return Success or failure
   */
  virtual bool extractSetTrajectories(const char* dir_or_file_path,
                                      TrajectorySet& trajectory_set,
                                      bool is_comma_separated = false,
                                      uint max_num_trajs = 500) = 0;

  /**
   * Pre-process demonstrated trajectories before learning from it.
   * To be called before learning, for example to convert the trajectory
   * representation to be with respect to its reference (local) coordinate frame
   * (in Cartesian Coordinate DMP), etc. By default there is no pre-processing.
   * If a specific DMP type (such as CartesianCoordDMP) requires pre-processing,
   * then just override this method/member function.
   *
   * @param trajectory_set Set of raw demonstrated trajectories to be
   * pre-processed (in-place pre-processing)
   * @return Success or failure
   */
  virtual bool preprocess(TrajectorySet& trajectory_set);

  /**
   * NON-REAL-TIME!!!\n
   * Learns DMP's forcing term from trajectory(ies) in the specified file (or
   * directory).
   *
   * @param training_data_dir_or_file_path Directory or file path where all
   * training data is stored
   * @param robot_task_servo_rate Robot's task servo rate
   * @param tau_learn (optional) If you also want to know \n
   *                  the tau (movement duration) used during learning, \n
   *                  please put the pointer here
   * @param critical_states_learn (optional) If you also want to know \n
   *                              the critical states (start position, goal
   * position, etc.) \n used during learning, please put the pointer here
   * @return Success or failure
   */
  virtual bool learn(const char* training_data_dir_or_file_path,
                     double robot_task_servo_rate, double* tau_learn = NULL,
                     Trajectory* critical_states_learn = NULL);

  /**
   * NON-REAL-TIME!!!\n
   * Learns from a given set of trajectories.
   *
   * @param trajectory_set Set of trajectories to learn from\n
   *        (assuming all trajectories in this set are coming from the same
   * trajectory distribution)
   * @param robot_task_servo_rate Robot's task servo rate
   * @return Success or failure
   */
  virtual bool learn(const TrajectorySet& trajectory_set,
                     double robot_task_servo_rate) = 0;

  /**
   * Resets DMP and sets all needed parameters (including the reference
   * coordinate frame) to start execution. Call this function before beginning
   * any DMP execution.
   *
   * @param critical_states Critical states defined for DMP trajectory
   * reproduction/unrolling.\n Start state is the first state in this
   * trajectory/vector of DMPStates.\n Goal state is the last state in this
   * trajectory/vector of DMPStates.
   * @param tau_init Initialization for tau (time parameter that influences the
   * speed of the execution; usually this is the time-length of the trajectory)
   * @return Success or failure
   */
  virtual bool start(const Trajectory& critical_states, double tau_init) = 0;

  /**
   * Resets DMP and sets all needed parameters (including the reference
   * coordinate frame) to start execution. Call this function before beginning
   * any DMP execution.
   *
   * @param dmp_unroll_init_parameters Parameter value definitions required to
   * unroll a DMP.
   * @return Success or failure
   */
  virtual bool start(const DMPUnrollInitParams& dmp_unroll_init_parameters);

  /**
   * Runs one step of the DMP based on the time step dt.
   *
   * @param dt Time difference between the previous and the current step
   * @param update_canonical_state Set to true if the canonical state should be
   * updated here.\n Otherwise you are responsible for updating it yourself.
   * @param next_state Computed next DMPState
   * @param transform_sys_forcing_term (optional) If you also want to know \n
   *                                   the transformation system's forcing term
   * value of the next state, \n please put the pointer here
   * @param transform_sys_coupling_term_acc (optional) If you also want to know
   * \n the acceleration-level coupling term value \n of the next state, please
   * put the pointer here
   * @param transform_sys_coupling_term_vel (optional) If you also want to know
   * \n the velocity-level coupling term value \n of the next state, please put
   * the pointer here
   * @param func_approx_basis_functions (optional) (Pointer to) vector of basis
   * functions constructing the forcing term
   * @param func_approx_normalized_basis_func_vector_mult_phase_multiplier
   * (optional) If also interested in \n the <normalized basis function vector
   * multiplied by the canonical phase multiplier (phase position or phase
   * velocity)> value, \n put its pointer here (return variable)
   * @return Success or failure
   */
  virtual bool getNextState(
      double dt, bool update_canonical_state, DMPState& next_state,
      VectorN* transform_sys_forcing_term = NULL,
      VectorN* transform_sys_coupling_term_acc = NULL,
      VectorN* transform_sys_coupling_term_vel = NULL,
      VectorM* func_approx_basis_functions = NULL,
      VectorM* func_approx_normalized_basis_func_vector_mult_phase_multiplier =
          NULL) = 0;

  /**
   * Returns current DMP state.
   */
  virtual DMPState getCurrentState() = 0;

  /**
   * Returns the value of the approximated function at current canonical
   * position and current canonical multiplier.
   *
   * @param result_f Computed forcing term at current canonical position and
   * current canonical multiplier (return variable)
   * @return Success or failure
   */
  virtual bool getForcingTerm(VectorN& result_f,
                              VectorM* basis_function_vector = NULL) = 0;

  /**
   * Returns current goal position.
   *
   * @param current_goal Current goal position (to be returned)
   * @return Success or failure
   */
  virtual bool getCurrentGoalPosition(VectorN& current_goal) = 0;

  /**
   * Returns steady-state goal position vector (G).
   *
   * @param Steady-state goal position vector (G) (to be returned)
   * @return Success or failure
   */
  virtual bool getSteadyStateGoalPosition(VectorN& steady_state_goal) = 0;

  /**
   * Sets new steady-state goal position (G).
   *
   * @param new_G New steady-state goal position (G) to be set
   * @return Success or failure
   */
  virtual bool setNewSteadyStateGoalPosition(const VectorN& new_G) = 0;

  /**
   * Set the scaling usage in each DMP dimension.
   *
   * @param Scaling usage (boolean) vector, each component corresponds to each
   * DMP dimension
   */
  virtual bool setScalingUsage(const std::vector<bool>& is_using_scaling) = 0;

  /**
   * NEVER RUN IN REAL-TIME\n
   * Save basis functions' values over variation of canonical state into a file.
   *
   * @param file_path
   * @return Success or failure
   */
  virtual bool saveBasisFunctions(const char* file_path) = 0;

  /**
   * Based on the start state, current state, evolving goal state and
   * steady-state goal position, compute the target value of the coupling term,
   * and then update the canonical state and goal state. Needed for learning the
   * coupling term function based on a given sequence of DMP states
   * (trajectory). [IMPORTANT]: Before calling this function, the
   * TransformationSystem's start_state, current_state, and goal_sys need to be
   * initialized first using TransformationSystem::start() function, and the
   * forcing term weights need to have already been learned beforehand, using
   * the demonstrated trajectory.
   *
   * @param current_state_demo_local Demonstrated current DMP state in the local
   * coordinate system \n (if there are coordinate transformations)
   * @param dt Time difference between the current and the previous execution
   * @param ct_acc_target Target value of the acceleration-level coupling term
   * at the demonstrated current state,\n given the demonstrated start state,
   * evolving goal state, \n demonstrated steady-state goal position, and the
   * learned forcing term weights \n (to be computed and returned by this
   * function)
   * @return Success or failure
   */
  virtual bool getTargetCouplingTermAndUpdateStates(
      const DMPState& current_state_demo_local, const double& dt,
      VectorN& ct_acc_target);

  /**
   * Returns the average start position of the training set of trajectories.
   */
  virtual VectorN getMeanStartPosition();

  /**
   * Returns the average goal position of the training set of trajectories.
   */
  virtual VectorN getMeanGoalPosition();

  /**
   * Returns the average tau of the training set of trajectories.
   */
  virtual double getMeanTau();

  virtual bool setDataDirectory(const char* new_data_dir_path);

  virtual bool startCollectingTrajectoryDataSet();

  virtual void stopCollectingTrajectoryDataSet();

  /**
   * NON-REAL-TIME!!!\n
   * If you want to maintain real-time performance,
   * do/call this function from a separate (non-real-time) thread!!!\n
   * This function saves the trajectory data set there.
   *
   * @param new_data_dir_path New data directory path \n
   *                          (if we want to save the data in a directory path
   * other than the data_directory_path)
   * @return Success or failure
   */
  virtual bool saveTrajectoryDataSet(const char* new_data_dir_path = "");

  virtual bool setTransformSystemCouplingTermUsagePerDimensions(
      const std::vector<bool>&
          is_using_transform_sys_coupling_term_at_dimension_init);

  virtual FunctionApproximator* getFunctionApproximatorPointer();

  virtual ~DMP();
};

}  // namespace dmp
#endif
