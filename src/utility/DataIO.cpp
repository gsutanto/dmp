#include "dmp/utility/DataIO.h"

namespace dmp {

DataIO::DataIO() : rt_assertor(NULL) {}

DataIO::DataIO(RealTimeAssertor* real_time_assertor)
    : rt_assertor(real_time_assertor) {}

bool DataIO::isValid() { return true; }

bool DataIO::extract1DTrajectory(const char* file_path, Trajectory& trajectory,
                                 bool is_comma_separated) {
  bool is_real_time = false;

  if (rt_assert(DataIO::isValid()) == false) {
    return false;
  }

  if (trajectory.empty() == false) {
    trajectory.clear();
  }

  FILE* f = fopen(file_path, "rt");
  if (rt_assert(f != NULL) == false) {
    return false;
  }

  char line[1000];
  VectorNPtr x;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, x, 1, 1)) == false) {
    return false;
  }
  VectorNPtr xd;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, xd, 1, 1)) == false) {
    return false;
  }
  VectorNPtr xdd;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, xdd, 1, 1)) ==
      false) {
    return false;
  }

  while (fgets(line, 1000, f) != NULL) {
    double time;
    if (is_comma_separated) {
      sscanf(line, "%lf,%lf,%lf,%lf", &time, &((*x)[0]), &((*xd)[0]),
             &((*xdd)[0]));
    } else {
      sscanf(line, "%lf %lf %lf %lf", &time, &((*x)[0]), &((*xd)[0]),
             &((*xdd)[0]));
    }
    DMPState state(*x, *xd, *xdd, time, rt_assertor);
    trajectory.push_back(state);
  }

  fclose(f);

  return true;
}

bool DataIO::write1DTrajectoryToFile(const Trajectory& trajectory,
                                     const char* file_path,
                                     bool is_comma_separated) {
  // bool is_real_time = false;

  if (rt_assert(DataIO::isValid()) == false) {
    return false;
  }
  uint traj_size = trajectory.size();
  if (rt_assert(rt_assert(traj_size > 0) &&
                rt_assert(traj_size <= MAX_TRAJ_SIZE)) == false) {
    return false;
  }

  FILE* f = fopen(file_path, "w");
  if (rt_assert(f != NULL) == false) {
    return false;
  }

  for (uint i = 0; i < traj_size; i++) {
    if (rt_assert(trajectory[i].isValid()) == false) {
      fclose(f);
      return false;
    }

    if (is_comma_separated) {
      fprintf(f, "%.05f,%.05f,%.05f,%.05f\n", trajectory[i].getTime(),
              (trajectory[i].getX())[0], (trajectory[i].getXd())[0],
              (trajectory[i].getXdd())[0]);
    } else {
      fprintf(f, "%.05f %.05f %.05f %.05f\n", trajectory[i].getTime(),
              (trajectory[i].getX())[0], (trajectory[i].getXd())[0],
              (trajectory[i].getXdd())[0]);
    }
  }

  fclose(f);

  return true;
}

bool DataIO::extractNDTrajectory(const char* file_path,
                                 Trajectory& trajectory) {
  // bool is_real_time = false;

  if (rt_assert(DataIO::isValid()) == false) {
    return false;
  }

  if (trajectory.empty() == false) {
    trajectory.clear();
  }

  MatrixXxXPtr file_content_as_matrix;
  if (rt_assert(DataIO::readMatrixFromFile(file_path,
                                           file_content_as_matrix)) == false) {
    return false;
  }
  if (rt_assert(((file_content_as_matrix->cols() - 1) % 3) == 0) == false) {
    return false;
  }

  // acquiring dimension size of the DMPState
  uint N = ((file_content_as_matrix->cols() - 1) / 3);
  if (rt_assert(N <= MAX_DMP_NUM_DIMENSIONS) == false) {
    return false;
  }

  for (uint nr = 0; nr < file_content_as_matrix->rows(); nr++) {
    DMPState state(file_content_as_matrix->block(nr, 1, 1, N),
                   file_content_as_matrix->block(nr, (N + 1), 1, N),
                   file_content_as_matrix->block(nr, ((2 * N) + 1), 1, N),
                   (*file_content_as_matrix)(nr, 0), rt_assertor);
    trajectory.push_back(state);
  }

  return true;
}

bool DataIO::extractCartCoordTrajectory(const char* file_path,
                                        Trajectory& cart_coord_trajectory,
                                        bool is_comma_separated) {
  bool is_real_time = false;

  if (rt_assert(DataIO::isValid()) == false) {
    return false;
  }

  if (cart_coord_trajectory.empty() == false) {
    cart_coord_trajectory.clear();
  }

  FILE* f = fopen(file_path, "rt");
  if (rt_assert(f != NULL) == false) {
    return false;
  }

  char line[1000];
  double time;
  VectorNPtr X;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, X, 3, 1)) == false) {
    return false;
  }
  VectorNPtr Xd;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Xd, 3, 1)) == false) {
    return false;
  }
  VectorNPtr Xdd;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Xdd, 3, 1)) ==
      false) {
    return false;
  }

  while (fgets(line, 1000, f) != NULL) {
    if (is_comma_separated) {
      sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &time, &((*X)[0]),
             &((*X)[1]), &((*X)[2]), &((*Xd)[0]), &((*Xd)[1]), &((*Xd)[2]),
             &((*Xdd)[0]), &((*Xdd)[1]), &((*Xdd)[2]));
    } else {
      sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &time, &((*X)[0]),
             &((*X)[1]), &((*X)[2]), &((*Xd)[0]), &((*Xd)[1]), &((*Xd)[2]),
             &((*Xdd)[0]), &((*Xdd)[1]), &((*Xdd)[2]));
    }
    DMPState cart_state(*X, *Xd, *Xdd, time, rt_assertor);
    cart_coord_trajectory.push_back(cart_state);
  }

  fclose(f);

  return true;
}

bool DataIO::writeCartCoordTrajectoryToFile(
    const Trajectory& cart_coord_trajectory, const char* file_path,
    bool is_comma_separated) {
  bool is_real_time = false;

  if (rt_assert(DataIO::isValid()) == false) {
    return false;
  }
  uint traj_size = cart_coord_trajectory.size();
  if (rt_assert(rt_assert(traj_size > 0) &&
                rt_assert(traj_size <= MAX_TRAJ_SIZE)) == false) {
    return false;
  }

  FILE* f = fopen(file_path, "w");
  if (rt_assert(f != NULL) == false) {
    return false;
  }

  double time, x, y, z, xd, yd, zd, xdd, ydd, zdd;
  VectorNPtr X;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, X, 3, 1)) == false) {
    return false;
  }
  VectorNPtr Xd;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Xd, 3, 1)) == false) {
    return false;
  }
  VectorNPtr Xdd;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Xdd, 3, 1)) ==
      false) {
    return false;
  }

  for (uint i = 0; i < traj_size; ++i) {
    if (rt_assert(cart_coord_trajectory[i].isValid()) == false) {
      fclose(f);
      return false;
    }

    time = cart_coord_trajectory[i].getTime();
    *X = cart_coord_trajectory[i].getX();
    *Xd = cart_coord_trajectory[i].getXd();
    *Xdd = cart_coord_trajectory[i].getXdd();
    x = (*X)[0];
    y = (*X)[1];
    z = (*X)[2];
    xd = (*Xd)[0];
    yd = (*Xd)[1];
    zd = (*Xd)[2];
    xdd = (*Xdd)[0];
    ydd = (*Xdd)[1];
    zdd = (*Xdd)[2];

    if (is_comma_separated) {
      fprintf(f,
              "%.05f,%.05f,%.05f,%.05f,%.05f,%.05f,%.05f,%.05f,%.05f,%.05f\n",
              time, x, y, z, xd, yd, zd, xdd, ydd, zdd);
    } else {
      fprintf(f,
              "%.05f %.05f %.05f %.05f %.05f %.05f %.05f %.05f %.05f %.05f\n",
              time, x, y, z, xd, yd, zd, xdd, ydd, zdd);
    }
  }

  fclose(f);

  return true;
}

bool DataIO::extractQuaternionTrajectory(const char* file_path,
                                         QuaternionTrajectory& quat_trajectory,
                                         bool is_comma_separated) {
  bool is_real_time = false;

  if (rt_assert(DataIO::isValid()) == false) {
    return false;
  }

  if (quat_trajectory.empty() == false) {
    quat_trajectory.clear();
  }

  FILE* f = fopen(file_path, "rt");
  if (rt_assert(f != NULL) == false) {
    return false;
  }

  char line[1000];
  double time;
  VectorNPtr Q;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Q, 4, 1)) == false) {
    return false;
  }
  VectorNPtr Qd;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Qd, 4, 1)) == false) {
    return false;
  }
  VectorNPtr Qdd;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Qdd, 4, 1)) ==
      false) {
    return false;
  }
  VectorNPtr omega;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, omega, 3, 1)) ==
      false) {
    return false;
  }
  VectorNPtr omegad;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, omegad, 3, 1)) ==
      false) {
    return false;
  }

  while (fgets(line, 1000, f) != NULL) {
    if (is_comma_separated) {
      sscanf(line,
             "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%"
             "lf,%lf,%lf",
             &time, &((*Q)[0]), &((*Q)[1]), &((*Q)[2]), &((*Q)[3]), &((*Qd)[0]),
             &((*Qd)[1]), &((*Qd)[2]), &((*Qd)[3]), &((*Qdd)[0]), &((*Qdd)[1]),
             &((*Qdd)[2]), &((*Qdd)[3]), &((*omega)[0]), &((*omega)[1]),
             &((*omega)[2]), &((*omegad)[0]), &((*omegad)[1]), &((*omegad)[2]));
    } else {
      sscanf(line,
             "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
             "%lf %lf %lf",
             &time, &((*Q)[0]), &((*Q)[1]), &((*Q)[2]), &((*Q)[3]), &((*Qd)[0]),
             &((*Qd)[1]), &((*Qd)[2]), &((*Qd)[3]), &((*Qdd)[0]), &((*Qdd)[1]),
             &((*Qdd)[2]), &((*Qdd)[3]), &((*omega)[0]), &((*omega)[1]),
             &((*omega)[2]), &((*omegad)[0]), &((*omegad)[1]), &((*omegad)[2]));
    }
    QuaternionDMPState quat_state(*Q, *Qd, *Qdd, *omega, *omegad, time,
                                  rt_assertor);
    quat_trajectory.push_back(quat_state);
  }

  fclose(f);

  return true;
}

bool DataIO::extractSetNDTrajectories(const char* dir_or_file_path,
                                      TrajectorySet& nd_trajectory_set,
                                      uint max_num_trajs) {
  bool is_real_time = false;

  if (rt_assert(DataIO::isValid()) == false) {
    return false;
  }

  if (nd_trajectory_set.empty() == false) {
    nd_trajectory_set.clear();
  }

  char var_file_path[1000];
  TrajectoryPtr nd_trajectory;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, nd_trajectory, 0)) ==
      false) {
    return false;
  }

  if (file_type(dir_or_file_path) == _DIR_)  // if it is a directory
  {
    uint i = 1;
    snprintf(var_file_path, sizeof(var_file_path),
             "%s/%d.txt", dir_or_file_path, i);

    while ((file_type(var_file_path) == _REG_) &&  // while file exists
           (i <= max_num_trajs)) {
      if (rt_assert(extractNDTrajectory(var_file_path, (*nd_trajectory))) ==
          false) {
        return false;
      }
      nd_trajectory_set.push_back(*nd_trajectory);
      i++;
      snprintf(var_file_path, sizeof(var_file_path),
               "%s/%d.txt", dir_or_file_path, i);
    }
  } else if (file_type(dir_or_file_path) == _REG_) {  // if it is a (text) file
    if (rt_assert(extractNDTrajectory(dir_or_file_path, (*nd_trajectory))) ==
        false) {
      return false;
    }
    nd_trajectory_set.push_back(*nd_trajectory);
  } else {
    return false;
  }

  return true;
}

bool DataIO::extractSetCartCoordTrajectories(
    const char* dir_or_file_path, TrajectorySet& cart_coord_trajectory_set,
    bool is_comma_separated, uint max_num_trajs) {
  bool is_real_time = false;

  if (rt_assert(DataIO::isValid()) == false) {
    return false;
  }

  if (cart_coord_trajectory_set.empty() == false) {
    cart_coord_trajectory_set.clear();
  }

  char var_file_path[1000];
  TrajectoryPtr cartesian_trajectory;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, cartesian_trajectory,
                                            0)) == false) {
    return false;
  }

  if (file_type(dir_or_file_path) == _DIR_)  // if it is a directory
  {
    uint i = 1;
    snprintf(var_file_path, sizeof(var_file_path),
             "%s/%d.txt", dir_or_file_path, i);

    while ((file_type(var_file_path) == _REG_) &&  // while file exists
           (i <= max_num_trajs)) {
      if (rt_assert(extractCartCoordTrajectory(
              var_file_path, (*cartesian_trajectory), is_comma_separated)) ==
          false) {
        return false;
      }
      cart_coord_trajectory_set.push_back(*cartesian_trajectory);
      i++;
      snprintf(var_file_path, sizeof(var_file_path),
               "%s/%d.txt", dir_or_file_path, i);
    }
  } else if (file_type(dir_or_file_path) == _REG_) {  // if it is a (text) file
    if (rt_assert(extractCartCoordTrajectory(
            dir_or_file_path, (*cartesian_trajectory), is_comma_separated)) ==
        false) {
      return false;
    }
    cart_coord_trajectory_set.push_back(*cartesian_trajectory);
  } else {
    return false;
  }

  return true;
}

bool DataIO::writeSetCartCoordTrajectoriesToDir(
    const TrajectorySet& cart_coord_trajectory_set, const char* dir_path,
    bool is_comma_separated) {
  // bool is_real_time = false;

  if (rt_assert(DataIO::isValid()) == false) {
    return false;
  }
  uint N_traj = cart_coord_trajectory_set.size();
  if (rt_assert(N_traj > 0) == false) {
    return false;
  }
  if (file_type(dir_path) != _DIR_) {
    mkdir(dir_path, ACCESSPERMS);
  }

  char var_file_path[1000];
  for (uint i = 0; i < N_traj; ++i) {
    snprintf(var_file_path, sizeof(var_file_path),
             "%s/%d.txt", dir_path, (i + 1));

    if (rt_assert(writeCartCoordTrajectoryToFile(
            cart_coord_trajectory_set[i], var_file_path, is_comma_separated)) ==
        false) {
      return false;
    }
  }

  return true;
}

bool DataIO::extractSetQuaternionTrajectories(
    const char* dir_or_file_path, QuaternionTrajectorySet& quat_trajectory_set,
    bool is_comma_separated, uint max_num_trajs) {
  bool is_real_time = false;

  if (rt_assert(DataIO::isValid()) == false) {
    return false;
  }

  if (quat_trajectory_set.empty() == false) {
    quat_trajectory_set.clear();
  }

  char var_file_path[1000];
  QuaternionTrajectoryPtr quat_trajectory;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, quat_trajectory,
                                            0)) == false) {
    return false;
  }

  if (file_type(dir_or_file_path) == _DIR_)  // if it is a directory
  {
    uint i = 1;
    snprintf(var_file_path, sizeof(var_file_path),
             "%s/%d.txt", dir_or_file_path, i);

    while ((file_type(var_file_path) == _REG_) &&  // while file exists
           (i <= max_num_trajs)) {
      if (rt_assert(extractQuaternionTrajectory(
              var_file_path, (*quat_trajectory), is_comma_separated)) ==
          false) {
        return false;
      }
      quat_trajectory_set.push_back(*quat_trajectory);
      i++;
      snprintf(var_file_path, sizeof(var_file_path),
               "%s/%d.txt", dir_or_file_path, i);
    }
  } else if (file_type(dir_or_file_path) == _REG_) {  // if it is a (text) file
    if (rt_assert(extractQuaternionTrajectory(
            dir_or_file_path, (*quat_trajectory), is_comma_separated)) ==
        false) {
      return false;
    }
    quat_trajectory_set.push_back(*quat_trajectory);
  } else {
    return false;
  }

  return true;
}

bool DataIO::extractSetCartCoordDemonstrationGroups(
    const char* in_data_dir_path, DemonstrationGroupSet& demo_group_set_global,
    uint& N_demo_settings, uint max_num_trajs_per_setting,
    std::vector<uint>* selected_obs_avoid_setting_numbers,
    VecVecDMPUnrollInitParams* demo_group_set_dmp_unroll_init_params) {
  bool is_real_time = false;

  // count total number of different obstacle avoidance demonstration settings
  // (or total number of different obstacle positions):
  double N_total_existing_demo_settings =
      countNumberedDirectoriesUnderDirectory(in_data_dir_path);
  if (rt_assert(N_total_existing_demo_settings >= 1) == false) {
    return false;
  }

  if (selected_obs_avoid_setting_numbers == NULL)  // use all settings
  {
    N_demo_settings = N_total_existing_demo_settings;
  } else {
    N_demo_settings = selected_obs_avoid_setting_numbers->size();
    if (rt_assert(N_total_existing_demo_settings >= N_demo_settings) == false) {
      return false;
    }
  }

  TrajectorySetPtr demo_group_global;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, demo_group_global,
                                            0)) == false) {
    return false;
  }

  char in_demo_group_dir_path[1000];
  for (uint n = 1; n <= N_demo_settings; n++) {
    if (selected_obs_avoid_setting_numbers == NULL)  // use all settings
    {
      snprintf(in_demo_group_dir_path, sizeof(in_demo_group_dir_path),
               "%s/%u/endeff_trajs/", in_data_dir_path, n);
    } else {
      if (rt_assert(((*selected_obs_avoid_setting_numbers)[n - 1] > 0) &&
                    ((*selected_obs_avoid_setting_numbers)[n - 1] <=
                     N_total_existing_demo_settings)) == false) {
        return false;
      }
      snprintf(in_demo_group_dir_path, sizeof(in_demo_group_dir_path),
               "%s/%u/endeff_trajs/", in_data_dir_path,
               (*selected_obs_avoid_setting_numbers)[n - 1]);
    }

    // Load the demonstrations for each particular obstacle setting:
    if (rt_assert(DataIO::extractSetCartCoordTrajectories(
            in_demo_group_dir_path, *demo_group_global, false,
            max_num_trajs_per_setting)) == false) {
      return false;
    }

    demo_group_set_global.push_back(*demo_group_global);
  }

  if (rt_assert(demo_group_set_global.size() == N_demo_settings) == false) {
    return false;
  }

  if (demo_group_set_dmp_unroll_init_params != NULL) {
    demo_group_set_dmp_unroll_init_params->clear();

    VecDMPUnrollInitParamsPtr demo_group_dmp_unroll_init_params;
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, demo_group_dmp_unroll_init_params, 0)) == false) {
      return false;
    }

    for (uint i = 0; i < demo_group_set_global.size(); i++) {
      demo_group_dmp_unroll_init_params->clear();

      for (uint j = 0; j < demo_group_set_global[i].size(); j++) {
        demo_group_dmp_unroll_init_params->push_back(
            DMPUnrollInitParams(demo_group_set_global[i][j], rt_assertor));
      }

      if (rt_assert(demo_group_dmp_unroll_init_params->size() ==
                    demo_group_set_global[i].size()) == false) {
        return false;
      }

      demo_group_set_dmp_unroll_init_params->push_back(
          *demo_group_dmp_unroll_init_params);
    }

    if (rt_assert(demo_group_set_dmp_unroll_init_params->size() ==
                  N_demo_settings) == false) {
      return false;
    }
  }

  return true;
}

bool DataIO::extract3DPoints(const char* file_path, Points& points,
                             bool is_comma_separated) {
  // bool is_real_time = false;

  if (rt_assert(DataIO::isValid()) == false) {
    return false;
  }

  if (points.empty() == false) {
    points.clear();
  }

  FILE* f = fopen(file_path, "rt");
  if (rt_assert(f != NULL) == false) {
    return false;
  }

  char line[1000];
  Vector3 point;

  while (fgets(line, 1000, f) != NULL) {
    if (is_comma_separated) {
      sscanf(line, "%lf,%lf,%lf", &(point[0]), &(point[1]), &(point[2]));
    } else {
      sscanf(line, "%lf %lf %lf", &(point[0]), &(point[1]), &(point[2]));
    }
    points.push_back(point);
  }

  fclose(f);

  return true;
}

bool DataIO::readMatrixFromFile(const char* file_path,
                                int& matrix_data_structure) {
  // bool is_real_time = false;

  if (rt_assert(DataIO::isValid()) == false) {
    return false;
  }

  FILE* f = fopen(file_path, "rt");
  if (rt_assert(f != NULL) == false) {
    return false;
  }

  char line[1000];
  if (rt_assert(fgets(line, 1000, f) != NULL)) {
    sscanf(line, "%d", &matrix_data_structure);
  } else {
    fclose(f);
    return false;
  }

  if (rt_assert(fgets(line, 1000, f) == NULL) == false) {
    fclose(f);
    return false;
  }

  fclose(f);

  return true;
}

bool DataIO::readMatrixFromFile(const char* file_path,
                                double& matrix_data_structure) {
  // bool is_real_time = false;

  if (rt_assert(DataIO::isValid()) == false) {
    return false;
  }

  FILE* f = fopen(file_path, "rt");
  if (rt_assert(f != NULL) == false) {
    return false;
  }

  char line[1000];
  if (rt_assert(fgets(line, 1000, f) != NULL)) {
    sscanf(line, "%lf", &matrix_data_structure);
  } else {
    fclose(f);
    return false;
  }

  if (rt_assert(fgets(line, 1000, f) == NULL) == false) {
    fclose(f);
    return false;
  }

  fclose(f);

  return true;
}

DataIO::~DataIO() {}

}  // namespace dmp
