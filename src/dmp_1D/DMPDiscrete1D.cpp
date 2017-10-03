#include "amd_clmc_dmp/dmp_1D/DMPDiscrete1D.h"

namespace dmp
{

DMPDiscrete1D::DMPDiscrete1D():
    transform_sys_discrete_1D(TransformSystemDiscrete(1, NULL,
                                                      &func_approx_discrete, std::vector<bool>(1, true),
                                                      _SCHAAL_DMP_, &logged_dmp_discrete_variables,
                                                      NULL, NULL, NULL, NULL, NULL, NULL)),
    DMPDiscrete(1, 0, NULL, &transform_sys_discrete_1D, _SCHAAL_LWR_METHOD_, NULL, "")
{}

/**
 * @param model_size_init Size of the model (M: number of basis functions or others) used to represent the function
 * @param canonical_system_discrete (Pointer to) discrete canonical system that drives this DMP
 * @param ts_formulation_type Chosen formulation for the discrete transformation system\n
 *        0: original formulation by Schaal, AMAM 2003 (default formulation_type)\n
 *        1: formulation by Hoffman et al., ICRA 2009
 * @param learning_method Learning method that will be used to learn the DMP basis functions' weights\n
 *        0=_SCHAAL_LWR_METHOD_: Stefan Schaal's Locally-Weighted Regression (LWR): Ijspeert et al., Neural Comput. 25, 2 (Feb 2013), p328-373\n
 *        1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_: Jan Peters' Global Least Squares method
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 * @param opt_data_directory_path [optional] Full directory path where to save/log all the data
 * @param transform_couplers All spatial couplings associated with the discrete transformation system
 * (please note that this is a pointer to vector of pointer to TransformCoupling data structure, for flexibility and re-usability)
 */
DMPDiscrete1D::DMPDiscrete1D(uint model_size_init, CanonicalSystemDiscrete* canonical_system_discrete,
                             uint ts_formulation_type, uint learning_method,
                             RealTimeAssertor* real_time_assertor,
                             const char* opt_data_directory_path,
                             std::vector<TransformCoupling*>* transform_couplers):
    transform_sys_discrete_1D(TransformSystemDiscrete(1, canonical_system_discrete,
                                                      &func_approx_discrete, std::vector<bool>(1, true),
                                                      ts_formulation_type, &logged_dmp_discrete_variables,
                                                      real_time_assertor, NULL, NULL, NULL, NULL, transform_couplers)),
    DMPDiscrete(1, model_size_init, canonical_system_discrete, &transform_sys_discrete_1D,
                learning_method, real_time_assertor,
                opt_data_directory_path)
{}

/**
 * Checks whether this 1D Discrete DMP is valid or not.
 *
 * @return 1D Discrete DMP is valid (true) or 1D Discrete DMP is invalid (false)
 */
bool DMPDiscrete1D::isValid()
{
    if (rt_assert(transform_sys_discrete_1D.isValid()) == false)
    {
        return false;
    }
    if (rt_assert(DMPDiscrete::isValid()) == false)
    {
        return false;
    }
    if (rt_assert(transform_sys_discrete == &transform_sys_discrete_1D) == false)
    {
        return false;
    }
    if (rt_assert(dmp_num_dimensions == 1) == false)
    {
        return false;
    }
    return true;
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
bool DMPDiscrete1D::extractSetTrajectories(const char* dir_or_file_path,
                                           TrajectorySet& trajectory_set,
                                           bool is_comma_separated,
                                           uint max_num_trajs)
{
    bool is_real_time   = false;

    if (trajectory_set.size() != 0)
    {
        trajectory_set.clear();
    }

    TrajectoryPtr       trajectory;
    if (rt_assert(allocateMemoryIfNonRealTime(is_real_time,
                                              trajectory)) == false)
    {
        return false;
    }

    if (rt_assert(data_logger_loader_discrete.extract1DTrajectory(dir_or_file_path, *trajectory,
                                                                  is_comma_separated)) == false)
    {
        return false;
    }

    trajectory_set.push_back(*trajectory);

    return true;
}

/**
 * NON-REAL-TIME!!!\n
 * Write a trajectory's information/content to a text file.
 *
 * @param trajectory Trajectory to be written to a file
 * @param file_path File path where to write the trajectory's information into
 * @param is_comma_separated If true, trajectory fields will be written with comma-separation;\n
 *        If false, trajectory fields will be written with white-space-separation (default is true)
 * @return Success or failure
 */
bool DMPDiscrete1D::write1DTrajectoryToFile(const Trajectory& trajectory,
                                            const char* file_path,
                                            bool is_comma_separated)
{
    return (rt_assert(data_logger_loader_discrete.write1DTrajectoryToFile(trajectory,
                                                                          file_path,
                                                                          is_comma_separated)));
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
bool DMPDiscrete1D::learn(const char* training_data_dir_or_file_path,
                          double robot_task_servo_rate,
                          double* tau_learn,
                          Trajectory* critical_states_learn)
{
    return(rt_assert(DMP::learn(training_data_dir_or_file_path, robot_task_servo_rate,
                                tau_learn, critical_states_learn)));
}

DMPDiscrete1D::~DMPDiscrete1D()
{}

}
