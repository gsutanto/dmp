/*
 * DMPDataIO.h
 *
 *  Data loading, data collection, and data saving of DMP parameters and
 * variables from/to file(s).
 *
 *  Created on: January 02, 2016
 *  Author: Giovanni Sutanto
 */

#ifndef DMP_DATA_IO_H
#define DMP_DATA_IO_H

#include <iomanip>

#include "dmp/dmp_base/FunctionApproximator.h"
#include "dmp/dmp_base/LoggedDMPVariables.h"
#include "dmp/dmp_base/TransformationSystem.h"
#include "dmp/utility/DataIO.h"

namespace dmp {

class DMPDataIO : public DataIO {
 protected:
  // The following field(s) are written as RAW pointers because we just want to
  // point to other data structure(s) that might outlive the instance of this
  // class:
  TransformationSystem* transform_sys;
  FunctionApproximator* func_approx;

  uint dmp_num_dimensions;
  uint model_size;

  // The following field(s) are written as SMART pointers to reduce size
  // overhead in the stack, by allocating them in the heap:
  MatrixTx2Ptr tau_trajectory_buffer;
  MatrixTxSPtr transform_sys_state_local_trajectory_buffer;
  MatrixTxSPtr transform_sys_state_global_trajectory_buffer;
  MatrixTxSPtr goal_state_local_trajectory_buffer;
  MatrixTxSPtr goal_state_global_trajectory_buffer;
  MatrixTxNp1Ptr steady_state_goal_position_local_trajectory_buffer;
  MatrixTxNp1Ptr steady_state_goal_position_global_trajectory_buffer;
  MatrixTxMp1Ptr basis_functions_trajectory_buffer;
  MatrixTxNp1Ptr forcing_term_trajectory_buffer;
  MatrixTxNp1Ptr transform_sys_ct_acc_trajectory_buffer;
  MatrixTxSBCPtr save_data_buffer;

  uint trajectory_step_count;

 public:
  DMPDataIO();

  /**
   * @param transformation_system Transformation System, whose data will be
   * recorded
   * @param function_approximator Function Approximator, whose data will be
   * recorded
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  DMPDataIO(TransformationSystem* transformation_system,
            FunctionApproximator* function_approximator,
            RealTimeAssertor* real_time_assertor);

  /**
   * Initialize/reset the data buffers and counters (usually done everytime
   * before start of data saving).
   *
   * @return Success or failure
   */
  virtual bool reset();

  /**
   * Checks whether this data recorder is valid or not.
   *
   * @return Data recorder is valid (true) or data recorder is invalid (false)
   */
  virtual bool isValid();

  /**
   * Copy the supplied variables into current column of the buffers.
   *
   * @param logged_dmp_variables_buffer DMP variables that will be logged
   * @return Success or failure
   */
  virtual bool collectTrajectoryDataSet(
      const LoggedDMPVariables& logged_dmp_variables_buffer);

  /**
   * NON-REAL-TIME!!!\n
   * If you want to maintain real-time performance,
   * do/call this function from a separate (non-real-time) thread!!!
   *
   * @param data_directory Specified directory to save the data files
   * @return Success or failure
   */
  virtual bool saveTrajectoryDataSet(const char* data_directory);

  ~DMPDataIO();
};

}  // namespace dmp
#endif
