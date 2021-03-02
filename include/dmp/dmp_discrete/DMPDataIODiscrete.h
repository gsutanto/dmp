/*
 * DMPDataIODiscrete.h
 *
 *  Data I/O for Discrete DMP.
 *
 *  Created on: Jan 13, 2015
 *  Author: Yevgen Chebotar and Giovanni Sutanto
 */

#ifndef DMP_DATA_IO_DISCRETE_H
#define DMP_DATA_IO_DISCRETE_H

#include <stdio.h>

#include "dmp/dmp_base/DMPDataIO.h"
#include "dmp/dmp_discrete/FuncApproximatorDiscrete.h"
#include "dmp/dmp_discrete/LoggedDMPDiscreteVariables.h"
#include "dmp/dmp_discrete/TransformSystemDiscrete.h"

namespace dmp {

class DMPDataIODiscrete : public DMPDataIO {
 protected:
  // The following field(s) are written as RAW pointers because we just want to
  // point to other data structure(s) that might outlive the instance of this
  // class:
  TransformSystemDiscrete* transform_sys_discrete;
  FuncApproximatorDiscrete* func_approx_discrete;

  // The following field(s) are written as SMART pointers to reduce size
  // overhead in the stack, by allocating them in the heap:
  MatrixTx4Ptr canonical_sys_state_trajectory_buffer;

 public:
  DMPDataIODiscrete();

  /**
   * @param transform_system_discrete Discrete Transformation System, whose data
   * will be recorded
   * @param func_approximator_discrete Discrete Function Approximator, whose
   * data will be recorded
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  DMPDataIODiscrete(TransformSystemDiscrete* transform_system_discrete,
                    FuncApproximatorDiscrete* func_approximator_discrete,
                    RealTimeAssertor* real_time_assertor);

  /**
   * Initialize/reset the data buffers and counters (usually done everytime
   * before start of data saving).
   *
   * @return Success or failure
   */
  bool reset();

  /**
   * Checks whether this discrete-DMP data recorder is valid or not.
   *
   * @return discrete-DMP data recorder is valid (true) or discrete-DMP data
   * recorder is invalid (false)
   */
  bool isValid();

  /**
   * Copy the supplied variables into current column of the buffers.
   *
   * @param logged_dmp_discrete_variables_buffer Discrete DMP variables that
   * will be logged
   * @return Success or failure
   */
  bool collectTrajectoryDataSet(
      const LoggedDMPDiscreteVariables& logged_dmp_discrete_variables_buffer);

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
   * NEVER RUN IN REAL-TIME\n
   * Save weights' matrix into a file.
   *
   * @param dir_path Directory path containing the file
   * @param file_name The file name
   * @return Success or failure
   */
  bool saveWeights(const char* dir_path, const char* file_name);

  /**
   * NEVER RUN IN REAL-TIME\n
   * Load weights' matrix from a file onto the function approximator.
   *
   * @param dir_path Directory path containing the file
   * @param file_name The file name
   * @return Success or failure
   */
  bool loadWeights(const char* dir_path, const char* file_name);

  /**
   * NEVER RUN IN REAL-TIME\n
   * Save transform_sys_discrete's A_learn vector into a file.
   *
   * @param dir_path Directory path containing the file
   * @param file_name The file name
   * @return Success or failure
   */
  bool saveALearn(const char* dir_path, const char* file_name);

  /**
   * NEVER RUN IN REAL-TIME\n
   * Load A_learn vector from a file onto the transform_sys_discrete.
   *
   * @param dir_path Directory path containing the file
   * @param file_name The file name
   * @return Success or failure
   */
  bool loadALearn(const char* dir_path, const char* file_name);

  /**
   * NEVER RUN IN REAL-TIME\n
   * Save basis functions' values over variation of canonical state position
   * into a file.
   *
   * @param file_path
   * @return Success or failure
   */
  bool saveBasisFunctions(const char* file_path);

  /**
   * Returns the discrete transformation system pointer.
   */
  TransformSystemDiscrete* getTransformSysDiscretePointer();

  /**
   * Returns the discrete function approximator pointer.
   */
  FuncApproximatorDiscrete* getFuncApproxDiscretePointer();

  ~DMPDataIODiscrete();
};
}  // namespace dmp

#endif
