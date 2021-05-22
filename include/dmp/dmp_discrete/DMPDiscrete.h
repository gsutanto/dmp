/*
 * DMPDiscrete.h
 *
 *  This implements a general structure of Discrete DMP that contains
 *  CanonicalSystem, TransformationSystem, FunctionApproximator and external
 * couplings
 *
 *  Created on: Dec 12, 2014
 *  Author: Yevgen Chebotar
 */

#ifndef DMP_DISCRETE_H
#define DMP_DISCRETE_H

#include "dmp/dmp_base/DMP.h"
#include "dmp/dmp_discrete/CanonicalSystemDiscrete.h"
#include "dmp/dmp_discrete/DMPDataIODiscrete.h"
#include "dmp/dmp_discrete/FuncApproximatorDiscrete.h"
#include "dmp/dmp_discrete/LearningSystemDiscrete.h"
#include "dmp/dmp_discrete/TransformSystemDiscrete.h"

namespace dmp {

class DMPDiscrete : public DMP {
 protected:
  CanonicalSystemDiscrete* canonical_sys_discrete;

  // The following field(s) are written as RAW pointers for extensibility in
  // Object-Oriented Programming (OOP)
  TransformSystemDiscrete* transform_sys_discrete;

  LearningSystemDiscrete learning_sys_discrete;
  DMPDataIODiscrete data_logger_loader_discrete;

  LoggedDMPDiscreteVariables logged_dmp_discrete_variables;

 private:
  /**
   * Get the current weights of the basis functions. \n
   * (Giovanni's Note: This is placed in private space to AVOID mistakes \n
   *                   by only getting/setting weights; You need to get/set both
   * \n FuncApproximatorDiscrete's weights and TransformSystemDiscrete's A_learn
   * together; \n See getParams(), setParams(), saveParams(), and loadParams(),
   * which are also \n in this same file, on public space...)
   *
   * @param weights_buffer Weights buffer, to which the current weights will be
   * copied onto (to be returned)
   * @return Success or failure
   */
  bool getWeights(MatrixNxM& weights_buffer);

  /**
   * Set the new weights of the basis functions. \n
   * (Giovanni's Note: This is placed in private space to AVOID mistakes \n
   *                   by only getting/setting weights; You need to get/set both
   * \n FuncApproximatorDiscrete's weights and TransformSystemDiscrete's A_learn
   * together; \n See getParams(), setParams(), saveParams(), and loadParams(),
   * which are also \n in this same file, on public space...)
   *
   * @param new_weights New weights to be set
   * @return Success or failure
   */
  bool setWeights(const MatrixNxM& new_weights);

 public:
  DMPDiscrete();

  /**
   * @param dmp_num_dimensions_init Number of trajectory dimensions employed in
   * this DMP
   * @param model_size_init Size of the model (M: number of basis functions or
   * others) used to represent the function
   * @param canonical_system_discrete (Pointer to) discrete canonical system
   * that drives this DMP
   * @param transform_system_discrete (Pointer to) discrete transformation
   * system that drives this DMP
   * @param learning_method Learning method that will be used to learn the DMP
   * basis functions' weights\n 0=_SCHAAL_LWR_METHOD_: Stefan Schaal's
   * Locally-Weighted Regression (LWR): Ijspeert et al., Neural Comput. 25, 2
   * (Feb 2013), p328-373\n 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_: Jan Peters'
   * Global Least Squares method
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   * @param opt_data_directory_path [optional] Full directory path where to
   * save/log all the data
   */
  DMPDiscrete(uint dmp_num_dimensions_init, uint model_size_init,
              CanonicalSystemDiscrete* canonical_system_discrete,
              TransformSystemDiscrete* transform_system_discrete,
              uint learning_method, RealTimeAssertor* real_time_assertor,
              const char* opt_data_directory_path = "");

  /**
   * Checks whether this Discrete DMP is valid or not.
   *
   * @return Discrete DMP is valid (true) or Discrete DMP is invalid (false)
   */
  bool isValid();

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
  bool learn(const TrajectorySet& trajectory_set, double robot_task_servo_rate);

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
  bool start(const Trajectory& critical_states, double tau_init);

  using DMP::start;

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
  bool getNextState(
      double dt, bool update_canonical_state, DMPState& next_state,
      VectorN* transform_sys_forcing_term = NULL,
      VectorN* transform_sys_coupling_term_acc = NULL,
      VectorN* transform_sys_coupling_term_vel = NULL,
      VectorM* func_approx_basis_functions = NULL,
      VectorM* func_approx_normalized_basis_func_vector_mult_phase_multiplier =
          NULL);

  /**
   * Returns current DMP state.
   */
  DMPState getCurrentState();

  /**
   * Returns the value of the approximated function at current canonical
   * position and \n current canonical multiplier, for each DMP dimensions (as a
   * vector).
   *
   * @param result_f Computed forcing term vector at current canonical position
   * and \n current canonical multiplier, each vector component for each DMP
   * dimension (return variable)
   * @return Success or failure
   */
  bool getForcingTerm(VectorN& result_f, VectorM* basis_function_vector = NULL);

  /**
   * Returns current goal position.
   *
   * @param current_goal Current goal position (to be returned)
   * @return Success or failure
   */
  bool getCurrentGoalPosition(VectorN& current_goal);

  /**
   * Returns steady-state goal position vector (G).
   *
   * @param Steady-state goal position vector (G) (to be returned)
   * @return Success or failure
   */
  bool getSteadyStateGoalPosition(VectorN& steady_state_goal);

  /**
   * Sets new steady-state goal position (G).
   *
   * @param new_G New steady-state goal position (G) to be set
   * @return Success or failure
   */
  bool setNewSteadyStateGoalPosition(const VectorN& new_G);

  /**
   * Set the scaling usage in each DMP dimension.
   *
   * @param Scaling usage (boolean) vector, each component corresponds to each
   * DMP dimension
   */
  bool setScalingUsage(const std::vector<bool>& is_using_scaling_init);

  /**
   * Get the current parameters of this discrete DMP (weights and learning
   * amplitude (A_learn))
   *
   * @param weights_buffer Weights buffer, to which the current weights will be
   * copied onto (to be returned)
   * @param A_learn_buffer A_learn vector buffer, to which the current learning
   * amplitude will be copied onto (to be returned)
   * @return Success or failure
   */
  bool getParams(MatrixNxM& weights_buffer, VectorN& A_learn_buffer);

  /**
   * Set the new parameters of this discrete DMP (weights and learning amplitude
   * (A_learn))
   *
   * @param new_weights New weights to be set
   * @param new_A_learn New learning amplitude (A_learn) to be set
   * @return Success or failure
   */
  bool setParams(const MatrixNxM& new_weights, const VectorN& new_A_learn);

  /**
   * NEVER RUN IN REAL-TIME\n
   * Save the current parameters of this discrete DMP (weights and learning
   * amplitude (A_learn)) into files.
   *
   * @param dir_path Directory path containing the files
   * @param file_name_weights Name of the file onto which the weights will be
   * saved
   * @param file_name_A_learn Name of the file onto which the A_learn vector
   * buffer will be saved
   * @param file_name_mean_start_position Name of the file onto which the mean
   * start position will be saved
   * @param file_name_mean_goal_position Name of the file onto which the mean
   * goal position will be saved
   * @param file_name_mean_tau Name of the file onto which the mean tau will be
   * saved
   * @return Success or failure
   */
  bool saveParams(
      const char* dir_path,
      const char* file_name_weights = "f_weights_matrix.txt",
      const char* file_name_A_learn = "f_A_learn_matrix.txt",
      const char* file_name_mean_start_position = "mean_start_position.txt",
      const char* file_name_mean_goal_position = "mean_goal_position.txt",
      const char* file_name_mean_tau = "mean_tau.txt");

  /**
   * NEVER RUN IN REAL-TIME\n
   * Load the new parameters of this discrete DMP (weights and learning
   * amplitude (A_learn)) from files
   *
   * @param dir_path Directory path containing the files
   * @param file_name_weights Name of the file containing the new weights to be
   * set
   * @param file_name_A_learn Name of the file containing the new learning
   * amplitude (A_learn) to be set
   * @param file_name_mean_start_position Name of the file containing the new
   * mean start position to be set
   * @param file_name_mean_goal_position Name of the file containing the new
   * mean goal position to be set
   * @param file_name_mean_tau Name of the file containing the new mean tau to
   * be set
   * @return Success or failure
   */
  bool loadParams(
      const char* dir_path,
      const char* file_name_weights = "f_weights_matrix.txt",
      const char* file_name_A_learn = "f_A_learn_matrix.txt",
      const char* file_name_mean_start_position = "mean_start_position.txt",
      const char* file_name_mean_goal_position = "mean_goal_position.txt",
      const char* file_name_mean_tau = "mean_tau.txt");

  /**
   * NEVER RUN IN REAL-TIME\n
   * Save basis functions' values over variation of canonical state position
   * into a file.
   *
   * @param file_path
   * @return Success or failure
   */
  bool saveBasisFunctions(const char* file_path);

  bool startCollectingTrajectoryDataSet();

  /**
   * NON-REAL-TIME!!!\n
   * If you want to maintain real-time performance,
   * do/call this function from a separate (non-real-time) thread!!!\n
   * This function saves the trajectory data set.
   *
   * @param new_data_dir_path New data directory path \n
   *                          (if we want to save the data in a directory path
   * other than the data_directory_path)
   * @return Success or failure
   */
  bool saveTrajectoryDataSet(const char* new_data_dir_path = "");

  /**
   * @return The order of the canonical system (1st or 2nd order).
   */
  uint getCanonicalSysOrder();

  /**
   * Returns model size used to represent the function.
   *
   * @return Model size used to represent the function
   */
  uint getFuncApproxModelSize();

  /**
   * @return The chosen formulation for the discrete transformation system\n
   *         0: original formulation by Schaal, AMAM 2003 (default
   * formulation_type)\n 1: formulation by Hoffman et al., ICRA 2009
   */
  uint getTransformSysFormulationType();

  /**
   * @return The learning method that is used to learn the DMP basis functions'
   * weights\n 0=_SCHAAL_LWR_METHOD_: Stefan Schaal's Locally-Weighted
   * Regression (LWR): Ijspeert et al., Neural Comput. 25, 2 (Feb 2013),
   * p328-373\n 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_: Jan Peters' Global
   * Least Squares method
   */
  uint getLearningSysMethod();

  /**
   * Returns the discrete function approximator pointer.
   */
  virtual std::shared_ptr<FuncApproximatorDiscrete>
  getFuncApproxDiscretePointer();

  ~DMPDiscrete();
};
}  // namespace dmp
#endif
