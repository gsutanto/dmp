/*
 * LearningSystem.h
 * (NON-Real-Time!!!)
 *
 *  Abstract class for a system that learns a function approximator.
 *  At the moment all learning parts of DMP is NON-real-time.
 *  Please do NOT call any functions/routines below
 *  (or any of its derivatives) within a real-time operation!!!
 *
 *  Created on: Dec 12, 2014
 *  Author: Yevgen Chebotar
 */

#ifndef LEARNING_SYSTEM_H
#define LEARNING_SYSTEM_H

#include "dmp/dmp_base/DMPDataIO.h"
#include "dmp/dmp_base/TransformationSystem.h"
#include "dmp/utility/DefinitionsDerived.h"
#include "dmp/utility/RealTimeAssertor.h"
#include "dmp/utility/utility.h"

namespace dmp {

class LearningSystem {
 protected:
  // The following field(s) are written as RAW pointers because we just want to
  // point to other data structure(s) that might outlive the instance of this
  // class:
  TransformationSystem* transform_sys;
  DMPDataIO* data_logger;
  RealTimeAssertor* rt_assertor;

  uint dmp_num_dimensions;
  uint model_size;

  // The following field(s) are written as SMART pointers to reduce size
  // overhead in the stack, by allocating them in the heap:
  MatrixNxMPtr
      learned_weights;  // pointer to the matrix of learned weights (matrix of
                        // size dmp_num_dimensionsXmodel_size)

  char data_directory_path[1000];

 public:
  LearningSystem();

  /**
   * NEVER RUN IN REAL-TIME, ONLY RUN IN INIT ROUTINE
   *
   * @param dmp_num_dimensions_init Number of trajectory dimensions employed in
   * this DMP
   * @param model_size_init Size of the model (M: number of basis functions or
   * others) used to represent the function
   * @param transformation_system Transformation system associated with this
   * Learning System
   * @param data_logging_system DMPDataIO associated with this Learning System
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   * @param opt_data_directory_path [optional] Full directory path where to
   * save/log the learning data
   */
  LearningSystem(uint dmp_num_dimensions_init, uint model_size_init,
                 TransformationSystem* transformation_system,
                 DMPDataIO* data_logging_system,
                 RealTimeAssertor* real_time_assertor,
                 const char* opt_data_directory_path = "");

  /**
   * Checks whether this learning system is valid or not.
   *
   * @return Learning system is valid (true) or learning system is invalid
   * (false)
   */
  virtual bool isValid();

  /**
   * NON-REAL-TIME!!!\n
   * Learns the function approximator parameters (weights, etc.) based on the
   * given set of trajectories.
   *
   * @param trajectory_set Set of trajectories to learn from\n
   *        (assuming all trajectories in this set are coming from the same
   * trajectory distribution)
   * @param robot_task_servo_rate Robot's task servo rate
   * @param mean_tau Average of tau over all trajectories in the set (to be
   * returned)
   * @return Success or failure
   */
  virtual bool learnApproximator(const TrajectorySet& trajectory_set,
                                 double robot_task_servo_rate,
                                 double& mean_tau) = 0;

  /**
   * Returns the number of dimensions in the DMP.
   *
   * @return Number of dimensions in the DMP
   */
  virtual uint getDMPNumDimensions();

  /**
   * Returns model size used to represent the function.
   *
   * @return Model size used to represent the function
   */
  virtual uint getModelSize();

  /**
   * Returns the transformation system pointer.
   */
  virtual TransformationSystem* getTransformationSystemPointer();

  ~LearningSystem();
};
}  // namespace dmp
#endif
