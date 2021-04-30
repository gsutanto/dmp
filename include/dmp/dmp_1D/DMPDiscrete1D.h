/*
 * DMPDiscrete1D.h
 *
 *  This implements a one-dimensional (1D) Discrete DMP.
 *
 *  Created on: October 18, 2015
 *  Author: Giovanni Sutanto
 */

#ifndef DMP_DISCRETE_1D_H
#define DMP_DISCRETE_1D_H

#include "dmp/dmp_discrete/DMPDiscrete.h"
#include "dmp/dmp_discrete/TransformSystemDiscrete.h"

namespace dmp {

class DMPDiscrete1D : public DMPDiscrete {
 protected:
  TransformSystemDiscrete transform_sys_discrete_1D;

  using DMPDiscrete::learn;

 public:
  DMPDiscrete1D();

  /**
   * @param model_size_init Size of the model (M: number of basis functions or
   * others) used to represent the function
   * @param canonical_system_discrete (Pointer to) discrete canonical system
   * that drives this DMP
   * @param ts_formulation_type Chosen formulation for the discrete
   * transformation system\n 0: original formulation by Schaal, AMAM 2003
   * (default formulation_type)\n 1: formulation by Hoffman et al., ICRA 2009
   * @param learning_method Learning method that will be used to learn the DMP
   * basis functions' weights\n 0=_SCHAAL_LWR_METHOD_: Stefan Schaal's
   * Locally-Weighted Regression (LWR): Ijspeert et al., Neural Comput. 25, 2
   * (Feb 2013), p328-373\n 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_: Jan Peters'
   * Global Least Squares method
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   * @param opt_data_directory_path [optional] Full directory path where to
   * save/log all the data
   * @param transform_couplers All spatial couplings associated with the
   * discrete transformation system (please note that this is a pointer to
   * vector of pointer to TransformCoupling data structure, for flexibility and
   * re-usability)
   */
  DMPDiscrete1D(uint model_size_init,
                CanonicalSystemDiscrete* canonical_system_discrete,
                uint ts_formulation_type, uint learning_method,
                RealTimeAssertor* real_time_assertor,
                const char* opt_data_directory_path = "",
                std::vector<TransformCoupling*>* transform_couplers = NULL);

  /**
   * Checks whether this 1D Discrete DMP is valid or not.
   *
   * @return 1D Discrete DMP is valid (true) or 1D Discrete DMP is invalid
   * (false)
   */
  bool isValid();

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
  bool extractSetTrajectories(const char* dir_or_file_path,
                              TrajectorySet& trajectory_set,
                              bool is_comma_separated = false,
                              uint max_num_trajs = 500);

  /**
   * NON-REAL-TIME!!!\n
   * Write a trajectory's information/content to a text file.
   *
   * @param trajectory Trajectory to be written to a file
   * @param file_path File path where to write the trajectory's information into
   * @param is_comma_separated If true, trajectory fields will be written with
   * comma-separation;\n If false, trajectory fields will be written with
   * white-space-separation (default is true)
   * @return Success or failure
   */
  bool write1DTrajectoryToFile(const Trajectory& trajectory,
                               const char* file_path,
                               bool is_comma_separated = true);

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
  bool learn(const char* training_data_dir_or_file_path,
             double robot_task_servo_rate, double* tau_learn = NULL,
             Trajectory* critical_states_learn = NULL);

  ~DMPDiscrete1D();
};
}  // namespace dmp
#endif
