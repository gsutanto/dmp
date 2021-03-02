/*
 * DataIO.h
 *
 *  General-purpose data loading and recording/saving from/to file(s).
 *
 *  Created on: Dec 12, 2014
 *  Author: Yevgen Chebotar and Giovanni Sutanto
 */

#ifndef DATA_IO_H
#define DATA_IO_H

#include <iomanip>

#include "dmp/dmp_param/DMPUnrollInitParams.h"
#include "dmp/dmp_state/DMPState.h"
#include "dmp/dmp_state/QuaternionDMPState.h"
#include "dmp/utility/DefinitionsDerived.h"
#include "dmp/utility/RealTimeAssertor.h"
#include "dmp/utility/utility.h"

namespace dmp {

class DataIO {
 protected:
  // The following field(s) are written as RAW pointers because we just want to
  // point to other data structure(s) that might outlive the instance of this
  // class:
  RealTimeAssertor* rt_assertor;

 private:
  /**
   * NON-REAL-TIME!!!\n
   * Extract a Cartesian coordinate trajectory from a text file.
   *
   * @param file_path File path to extract trajectory information from
   * @param cart_coord_trajectory (Blank/empty) Cartesian coordinate trajectory
   * to be filled-in
   * @param is_comma_separated If true, trajectory fields are separated by
   * commas;\n If false, trajectory fields are separated by white-space
   * characters (default is false)
   * @return Success or failure
   */
  bool extractCartCoordTrajectory(const char* file_path,
                                  Trajectory& cart_coord_trajectory,
                                  bool is_comma_separated = false);

 public:
  DataIO();

  /**
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  DataIO(RealTimeAssertor* real_time_assertor);

  /**
   * Checks whether this data recorder is valid or not.
   *
   * @return Data recorder is valid (true) or data recorder is invalid (false)
   */
  bool isValid();

  /**
   * NON-REAL-TIME!!!\n
   * Extract a trajectory from a text file.
   *
   * @param file_path File path to extract trajectory information from
   * @param trajectory (Blank/empty) trajectory to be filled-in
   * @param is_comma_separated If true, trajectory fields are separated by
   * commas;\n If false, trajectory fields are separated by white-space
   * characters (default is false)
   * @return Success or failure
   */
  bool extract1DTrajectory(const char* file_path, Trajectory& trajectory,
                           bool is_comma_separated = false);

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
   * Extract an N-dimensional trajectory from a text file.
   *
   * @param file_path File path to extract trajectory information from
   * @param trajectory (Blank/empty) trajectory to be filled-in
   * @return Success or failure
   */
  bool extractNDTrajectory(const char* file_path, Trajectory& trajectory);

  /**
   * NON-REAL-TIME!!!\n
   * Write a Cartesian coordinate trajectory's information/content to a text
   * file.
   *
   * @param cart_coord_trajectory Cartesian coordinate trajectory to be written
   * to a file
   * @param file_path File path where to write the Cartesian coordinate
   * trajectory's information into
   * @param is_comma_separated If true, Cartesian coordinate trajectory fields
   * will be written with comma-separation;\n If false, Cartesian coordinate
   * trajectory fields will be written with white-space-separation (default is
   * true)
   * @return Success or failure
   */
  bool writeCartCoordTrajectoryToFile(const Trajectory& cart_coord_trajectory,
                                      const char* file_path,
                                      bool is_comma_separated = true);

  /**
   * NON-REAL-TIME!!!\n
   * Extract a Quaternion trajectory from a text file.
   *
   * @param file_path File path to extract trajectory information from
   * @param quat_trajectory (Blank/empty) Quaternion trajectory to be filled-in
   * @param is_comma_separated If true, trajectory fields are separated by
   * commas;\n If false, trajectory fields are separated by white-space
   * characters (default is false)
   * @return Success or failure
   */
  bool extractQuaternionTrajectory(const char* file_path,
                                   QuaternionTrajectory& quat_trajectory,
                                   bool is_comma_separated = false);

  /**
   * NON-REAL-TIME!!!\n
   * Extract a set of N-dimensional trajectory(ies) from text file(s).
   *
   * @param dir_or_file_path Two possibilities:\n
   *        If want to extract a set of N-dimensional trajectories, \n
   *        then specify the directory path containing the text files, \n
   *        each text file represents a single N-dimensional trajectory.
   *        If want to extract just a single N-dimensional trajectory, \n
   *        then specify the file path to the text file representing such
   * trajectory.
   * @param nd_trajectory_set (Blank/empty) set of N-dimensional trajectories to
   * be filled-in
   * @param max_num_trajs [optional] Maximum number of trajectories that could
   * be loaded into a trajectory set (default is 500)
   * @return Success or failure
   */
  bool extractSetNDTrajectories(const char* dir_or_file_path,
                                TrajectorySet& nd_trajectory_set,
                                uint max_num_trajs = 500);

  /**
   * NON-REAL-TIME!!!\n
   * Extract a set of Cartesian coordinate trajectory(ies) from text file(s).
   *
   * @param dir_or_file_path Two possibilities:\n
   *        If want to extract a set of Cartesian coordinate trajectories, \n
   *        then specify the directory path containing the text files, \n
   *        each text file represents a single Cartesian coordinate trajectory.
   *        If want to extract just a single Cartesian coordinate trajectory, \n
   *        then specify the file path to the text file representing such
   * trajectory.
   * @param cart_coord_trajectory_set (Blank/empty) set of Cartesian coordinate
   * trajectories to be filled-in
   * @param is_comma_separated If true, trajectory fields are separated by
   * commas;\n If false, trajectory fields are separated by white-space
   * characters (default is false)
   * @param max_num_trajs [optional] Maximum number of trajectories that could
   * be loaded into a trajectory set (default is 500)
   * @return Success or failure
   */
  bool extractSetCartCoordTrajectories(const char* dir_or_file_path,
                                       TrajectorySet& cart_coord_trajectory_set,
                                       bool is_comma_separated = false,
                                       uint max_num_trajs = 500);

  /**
   * NON-REAL-TIME!!!\n
   * Write a set of Cartesian coordinate trajectories to text files contained in
   * one folder/directory.
   *
   * @param cart_coord_trajectory_set Set of Cartesian coordinate trajectories
   * to be written into the specified directory
   * @param dir_path Directory path where to write the set of Cartesian
   * coordinate trajectories representation into
   * @param is_comma_separated If true, Cartesian coordinate trajectory fields
   * will be written with comma-separation;\n If false, Cartesian coordinate
   * trajectory fields will be written with white-space-separation (default is
   * true)
   * @return Success or failure
   */
  bool writeSetCartCoordTrajectoriesToDir(
      const TrajectorySet& cart_coord_trajectory_set, const char* dir_path,
      bool is_comma_separated = true);

  /**
   * NON-REAL-TIME!!!\n
   * Extract a set of Quaternion trajectory(ies) from text file(s).
   *
   * @param dir_or_file_path Two possibilities:\n
   *        If want to extract a set of Quaternion trajectories, \n
   *        then specify the directory path containing the text files, \n
   *        each text file represents a single Quaternion trajectory.
   *        If want to extract just a single Quaternion trajectory, \n
   *        then specify the file path to the text file representing such
   * trajectory.
   * @param quat_trajectory_set (Blank/empty) set of Quaternion trajectories to
   * be filled-in
   * @param is_comma_separated If true, trajectory fields are separated by
   * commas;\n If false, trajectory fields are separated by white-space
   * characters (default is false)
   * @param max_num_trajs [optional] Maximum number of trajectories that could
   * be loaded into a trajectory set (default is 500)
   * @return Success or failure
   */
  bool extractSetQuaternionTrajectories(
      const char* dir_or_file_path,
      QuaternionTrajectorySet& quat_trajectory_set,
      bool is_comma_separated = false, uint max_num_trajs = 500);

  /**
   * NON-REAL-TIME!!!\n
   * If you want to maintain real-time performance,
   * do/call this function from a separate (non-real-time) thread!!!\n
   * Extract all DMP training demonstrations.
   *
   * @param in_data_dir_path The input data directory
   * @param demo_group_set_global [to-be-filled/returned] Extracted
   * demonstrations \n in the Cartesian global coordinate system
   * @param N_demo_settings [to-be-filled/returned] Total number of different
   * demonstration settings \n (e.g. total number of different obstacle
   * positions, in obstacle avoidance case)
   * @param max_num_trajs_per_setting [optional] Maximum number of trajectories
   * that could be loaded into a trajectory set \n of a demonstration setting \n
   *                                  (e.g. setting = obstacle position in
   * obstacle avoidance context) \n (default is 500)
   * @param selected_obs_avoid_setting_numbers [optional] If we are just
   * interested in learning a subset of the obstacle avoidance demonstration
   * settings,\n then specify the numbers (IDs) of the setting here (as a
   * vector).\n If not specified, then all settings will be considered.
   * @param demo_group_set_dmp_unroll_init_params [optional] DMP unroll
   * initialization parameters of each trajectory in \n the extracted
   * demonstrations (to be returned)
   * @return Success or failure
   */
  bool extractSetCartCoordDemonstrationGroups(
      const char* in_data_dir_path,
      DemonstrationGroupSet& demo_group_set_obs_avoid_global,
      uint& N_demo_settings, uint max_num_trajs_per_setting = 500,
      std::vector<uint>* selected_obs_avoid_setting_numbers = NULL,
      VecVecDMPUnrollInitParams* demo_group_set_dmp_unroll_init_params = NULL);

  /**
   * NON-REAL-TIME!!!\n
   * Extract an arbitrary-length vector of 3D points from a text file.
   *
   * @param file_path File path to extract 3D points information from
   * @param points (Blank/empty) 3D points vector to be filled-in
   * @param is_comma_separated If true, fields are separated by commas;\n
   *        If false, fields are separated by white-space characters (default is
   * false)
   * @return Success or failure
   */
  bool extract3DPoints(const char* file_path, Points& points,
                       bool is_comma_separated = false);

  /**
   * Overloading of templated function readMatrixFromFile() for (T==int).
   */
  bool readMatrixFromFile(const char* file_path, int& matrix_data_structure);

  /**
   * Overloading of templated function readMatrixFromFile() for (T==double).
   */
  bool readMatrixFromFile(const char* file_path, double& matrix_data_structure);

  /**
   * NON-REAL-TIME!!!\n
   * Sorry for the mess here, C++ compiler did NOT allow me to write the
   * following function definition in DataIO.cpp file (if I do, I have to
   * instantiate it, which is annoying).
   *
   * @param dir_path Directory path containing the file onto where the
   * matrix_data_structure will be written
   * @param file_name Name of the file inside dir_path onto where the
   * matrix_data_structure will be written
   * @param matrix_data_structure The matrix data structure that will be written
   * to the file
   * @return Success or failure
   */
  template <class T>
  bool writeMatrixToFile(const char* dir_path, const char* file_name,
                         T const& matrix_data_structure) {
    if (rt_assert(DataIO::isValid()) == false) {
      return false;
    }

    if (rt_assert(createDirIfNotExistYet(dir_path)) == false) {
      return false;
    }

    char file_path[1000];
    sprintf(file_path, "%s/%s", dir_path, file_name);

    return (
        rt_assert(DataIO::writeMatrixToFile(file_path, matrix_data_structure)));
  }

  /**
   * NON-REAL-TIME!!!\n
   * Sorry for the mess here, C++ compiler did NOT allow me to write the
   * following function definition in DataIO.cpp file (if I do, I have to
   * instantiate it, which is annoying).
   *
   * @param dir_path Directory path containing the file from where the content
   * of matrix_data_structure will be read
   * @param file_name Name of the file inside dir_path from where the content of
   * matrix_data_structure will be read
   * @param matrix_data_structure The matrix data structure that will be filled
   * in \n (assuming size is already fixed and known before calling this
   * function)
   * @return Success or failure
   */
  template <class T>
  bool readMatrixFromFile(const char* dir_path, const char* file_name,
                          T& matrix_data_structure) {
    if (rt_assert(DataIO::isValid()) == false) {
      return false;
    }

    char file_path[1000];
    sprintf(file_path, "%s/%s", dir_path, file_name);

    return (rt_assert(
        DataIO::readMatrixFromFile(file_path, matrix_data_structure)));
  }

  /**
   * NON-REAL-TIME!!!\n
   * Sorry for the mess here, C++ compiler did NOT allow me to write the
   * following function definition in DataIO.cpp file (if I do, I have to
   * instantiate it, which is annoying).
   *
   * @param file_path The complete file path onto where the
   * matrix_data_structure will be written
   * @param matrix_data_structure The matrix data structure that will be written
   * to the file
   * @return Success or failure
   */
  template <class T>
  bool writeMatrixToFile(const char* file_path,
                         T const& matrix_data_structure) {
    if (rt_assert(DataIO::isValid()) == false) {
      return false;
    }

    std::ofstream file(file_path);
    if (rt_assert(file.is_open())) {
      file << std::setprecision(15);
      file << matrix_data_structure << '\n';
      file.close();
      return true;
    } else {
      return false;
    }
  }

  /**
   * NON-REAL-TIME!!!\n
   * Sorry for the mess here, C++ compiler did NOT allow me to write the
   * following function definition in DataIO.cpp file (if I do, I have to
   * instantiate it, which is annoying).
   *
   * Read matrix with fixed size from file.
   *
   * @param file_path The complete file path from where the content of
   * matrix_data_structure will be read
   * @param matrix_data_structure The matrix data structure that will be filled
   * in \n (assuming size is already fixed and known before calling this
   * function)
   * @return Success or failure
   */
  template <class T>
  bool readMatrixFromFile(const char* file_path, T& matrix_data_structure) {
    bool is_real_time = false;

    if (rt_assert(DataIO::isValid()) == false) {
      return false;
    }

    int cols = 0;
    int rows = 0;

    DoubleVectorPtr buffer;
    unsigned long alloc_1D_size =
        matrix_data_structure.rows() * matrix_data_structure.cols();
    if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, buffer,
                                              alloc_1D_size)) == false) {
      return false;
    }

    std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    // Read numbers from file into buffer.
    std::ifstream infile;
    infile.open(file_path);
    if (rt_assert(infile.is_open()) == false) {
      return false;
    }
    while (!infile.eof()) {
      std::string line;
      std::getline(infile, line);

      int temp_cols = 0;
      std::stringstream stream(line);
      while (!stream.eof()) {
        stream >> (*buffer)[cols * rows + temp_cols++];
      }

      if (temp_cols == 0) {
        continue;
      }

      if (cols == 0) {
        cols = temp_cols;
      }

      rows++;
    }

    infile.close();

    rows--;

    if (rt_assert((rt_assert(matrix_data_structure.rows() == rows)) &&
                  (rt_assert(matrix_data_structure.cols() == cols))) == false) {
      return false;
    }

    // Populate Eigen data structure (matrix or vector) with numbers.
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        matrix_data_structure(i, j) = (*buffer)[cols * i + j];
      }
    }

    return true;
  }

  /**
   * NON-REAL-TIME!!!\n
   * Sorry for the mess here, C++ compiler did NOT allow me to write the
   * following function definition in DataIO.cpp file (if I do, I have to
   * instantiate it, which is annoying).
   *
   * Read matrix with arbitrary size from file.
   *
   * @param file_path The complete file path from where the content of
   * matrix_data_structure will be read
   * @param matrix_data_structure The smart pointer to the matrix data structure
   * that will be filled in \n (size is unknown and dependent on the file being
   * read, \n after content is read from file, the space is allocated in the
   * heap \n for this matrix)
   * @return Success or failure
   */
  template <class T>
  bool readMatrixFromFile(const char* file_path,
                          std::shared_ptr<T>& matrix_data_structure) {
    bool is_real_time = false;

    if (rt_assert(DataIO::isValid()) == false) {
      return false;
    }

    int cols = 0;
    int rows = 0;

    DoubleVectorPtr buffer;
    unsigned long alloc_1D_size = MAX_TRAJ_SIZE * MAX_DMP_STATE_SIZE;
    if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, buffer,
                                              alloc_1D_size)) == false) {
      return false;
    }

    std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    // Read numbers from file into buffer.
    std::ifstream infile;
    infile.open(file_path);
    if (rt_assert(infile.is_open()) == false) {
      return false;
    }
    while (!infile.eof()) {
      std::string line;
      std::getline(infile, line);

      int temp_cols = 0;
      std::stringstream stream(line);
      while (!stream.eof()) {
        stream >> (*buffer)[cols * rows + temp_cols++];
      }

      if (temp_cols == 0) {
        continue;
      }

      if (cols == 0) {
        cols = temp_cols;
      }

      rows++;
    }

    infile.close();

    rows--;

    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, matrix_data_structure, rows, cols)) == false) {
      return false;
    }

    // Populate Eigen data structure (matrix or vector) with numbers.
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        (*matrix_data_structure)(i, j) = (*buffer)[cols * i + j];
      }
    }

    return true;
  }

  /**
   * NON-REAL-TIME!!!\n
   * Sorry for the mess here, C++ compiler did NOT allow me to write the
   * following function definition in DataIO.cpp file (if I do, I have to
   * instantiate it, which is annoying).
   *
   * @param dir_path Directory path onto where the data will be saved (in the
   * form of a file)
   * @param file_name Name of the file inside dir_path onto where the data will
   * be saved
   * @param row_size The total number of rows of data_collection_buffer that
   * will be saved
   * @param col_size The total number of columns of data_collection_buffer that
   * will be saved
   * @param data_saving_buffer Data buffer whose content will be redirected to
   * the file
   * @param data_collection_buffer Data buffer for collecting data iteratively,
   * \n whose content essence will be copied into the data_saving_buffer
   * @return Success or failure
   */
  template <class T1, class T2>
  bool saveTrajectoryData(const char* dir_path, const char* file_name,
                          const uint& row_size, const uint& col_size,
                          T1& data_saving_buffer, T2& data_collection_buffer) {
    if (rt_assert(DataIO::isValid()) == false) {
      return false;
    }

    data_saving_buffer.resize(row_size, col_size);
    data_saving_buffer = data_collection_buffer.block(0, 0, row_size, col_size);
    if (rt_assert(DataIO::writeMatrixToFile(dir_path, file_name,
                                            data_saving_buffer)) == false) {
      return false;
    }

    return true;
  }

  ~DataIO();
};

}  // namespace dmp
#endif
