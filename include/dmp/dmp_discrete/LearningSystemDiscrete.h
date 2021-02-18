/*
 * LearningSystemDiscrete.h
 * (NON-Real-Time!!!)
 *
 *  Uses locally-weighted regression (LWR) to learn weights of the non-linear
 *  function approximator based on weighted Gaussian basis functions.
 *  At the moment all learning parts of DMP are NON-real-time.
 *  Please do NOT call any functions/routines below
 *  (or any of its derivatives) within a real-time operation!!!
 *
 *  See A.J. Ijspeert, J. Nakanishi, H. Hoffmann, P. Pastor, and S. Schaal;
 *      Dynamical movement primitives: Learning attractor models for motor behaviors;
 *      Neural Comput. 25, 2 (February 2013), 328-373
 *
 *  Created on: Dec 12, 2014
 *  Author: Yevgen Chebotar and Giovanni Sutanto
 */

#ifndef LEARNING_SYSTEM_DISCRETE_H
#define LEARNING_SYSTEM_DISCRETE_H

#include <stddef.h>

#include "amd_clmc_dmp/utility/DefinitionsDerived.h"
#include "amd_clmc_dmp/dmp_goal_system/GoalSystem.h"
#include "amd_clmc_dmp/dmp_state/DMPState.h"
#include "amd_clmc_dmp/dmp_base/LearningSystem.h"
#include "amd_clmc_dmp/dmp_discrete/TransformSystemDiscrete.h"
#include "amd_clmc_dmp/dmp_discrete/FuncApproximatorDiscrete.h"
#include "amd_clmc_dmp/dmp_discrete/DMPDataIODiscrete.h"

namespace dmp
{

class LearningSystemDiscrete : public LearningSystem
{

protected:

    uint            learning_method;

private:

    VectorN         min_coord;  // minimum coordinate in the trajectory, for each dimension (reserved for future use)
    VectorN         max_coord;  // maximum coordinate in the trajectory, for each dimension (reserved for future use)

public:

    LearningSystemDiscrete();

    /**
     * NEVER RUN IN REAL-TIME, ONLY RUN IN INIT ROUTINE
     *
     * @param dmp_num_dimensions_init Number of trajectory dimensions employed in this DMP
     * @param model_size_init Size of the model (M: number of basis functions or others) used to represent the function
     * @param transformation_system_discrete Discrete transformation system associated with this discrete learning system
     * @param data_logging_system DMPDataIODiscrete associated with this discrete learning system
     * @param selected_learning_method Learning method that will be used to learn the DMP basis functions' weights\n
     *        0=_SCHAAL_LWR_METHOD_: Stefan Schaal's Locally-Weighted Regression (LWR): Ijspeert et al., Neural Comput. 25, 2 (Feb 2013), p328-373\n
     *        1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_: Jan Peters' Global Least Squares method
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     * @param opt_data_directory_path [optional] Full directory path where to save/log the learning data
     */
    LearningSystemDiscrete(uint dmp_num_dimensions_init, uint model_size_init,
                           TransformSystemDiscrete* transformation_system_discrete,
                           DMPDataIODiscrete* data_logging_system,
                           uint selected_learning_method,
                           RealTimeAssertor* real_time_assertor,
                           const char* opt_data_directory_path="");

    /**
     * Checks whether this discrete learning system is valid or not.
     *
     * @return Discrete learning system is valid (true) or discrete learning system is invalid (false)
     */
    bool isValid();

    /**
     * NON-REAL-TIME!!!\n
     * Learns and updates weights of the given discrete function approximator based on the given set of trajectories.
     *
     * @param trajectory_set Set of trajectories to learn from\n
     *        (assuming all trajectories in this set are coming from the same trajectory distribution)
     * @param robot_task_servo_rate Robot's task servo rate
     * @param mean_tau Average of tau over all trajectories in the set (to be returned)
     * @return Success or failure
     */
    bool learnApproximator(const TrajectorySet& trajectory_set,
                           double robot_task_servo_rate, double& mean_tau);

    /**
     * @return The learning method that is used to learn the DMP basis functions' weights\n
     *         0=_SCHAAL_LWR_METHOD_: Stefan Schaal's Locally-Weighted Regression (LWR): Ijspeert et al., Neural Comput. 25, 2 (Feb 2013), p328-373\n
     *         1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_: Jan Peters' Global Least Squares method
     */
    uint getLearningMethod();

    ~LearningSystemDiscrete();

};

}
#endif
