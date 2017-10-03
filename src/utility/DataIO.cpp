#include "amd_clmc_dmp/utility/DataIO.h"

namespace dmp
{

    DataIO::DataIO():
        rt_assertor(NULL)
    {}

    /**
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    DataIO::DataIO(RealTimeAssertor* real_time_assertor):
        rt_assertor(real_time_assertor)
    {}

    /**
     * Checks whether this data recorder is valid or not.
     *
     * @return Data recorder is valid (true) or data recorder is invalid (false)
     */
    bool DataIO::isValid()
    {
        return true;
    }

    /**
     * NON-REAL-TIME!!!\n
     * Extract trajectory from a text file.
     *
     * @param file_path File path to extract trajectory information from
     * @param trajectory (Blank/empty) trajectory to be filled-in
     * @param is_comma_separated If true, trajectory fields are separated by commas;\n
     *        If false, trajectory fields are separated by white-characters (default is false)
     * @return Success or failure
     */
    bool DataIO::extract1DTrajectory(const char* file_path, Trajectory& trajectory,
                                     bool is_comma_separated)
    {
        bool is_real_time   = false;

        if (rt_assert(DataIO::isValid()) == false)
        {
            return false;
        }

        if (trajectory.size() != 0)
        {
            trajectory.clear();
        }

        FILE*   f   = fopen(file_path, "rt");
        if (rt_assert(f != NULL) == false)
        {
            return false;
        }

        char        line[1000];
        VectorNPtr  x;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, x, 1, 1)) == false)
        {
            return false;
        }
        VectorNPtr  xd;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, xd, 1, 1)) == false)
        {
            return false;
        }
        VectorNPtr  xdd;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, xdd, 1, 1)) == false)
        {
            return false;
        }

        while(fgets(line, 1000, f) != NULL)
        {
            double      time;
            if (is_comma_separated)
            {
                sscanf(line, "%lf,%lf,%lf,%lf",
                       &time, &((*x)[0]), &((*xd)[0]), &((*xdd)[0]));
            }
            else
            {
                sscanf(line, "%lf %lf %lf %lf",
                       &time, &((*x)[0]), &((*xd)[0]), &((*xdd)[0]));
            }
            DMPState    state(*x, *xd, *xdd, time, rt_assertor);
            trajectory.push_back(state);
        }

        fclose(f);

        return true;
    }

    /**
     * NON-REAL-TIME!!!\n
     * Write a trajectory's information/content to a text file.
     *
     * @param trajectory Trajectory to be written to a file
     * @param file_path File path where to write the trajectory's information onto
     * @param is_comma_separated If true, trajectory fields will be written with comma-separation;\n
     *        If false, trajectory fields will be written with white-space-separation (default is true)
     * @return Success or failure
     */
    bool DataIO::write1DTrajectoryToFile(const Trajectory& trajectory,
                                         const char* file_path, bool is_comma_separated)
    {
        bool is_real_time   = false;

        if (rt_assert(DataIO::isValid()) == false)
        {
            return false;
        }
        uint    traj_size   = trajectory.size();
        if (rt_assert(rt_assert(traj_size > 0) && rt_assert(traj_size <= MAX_TRAJ_SIZE)) == false)
        {
            return false;
        }

        FILE*   f           = fopen(file_path, "w");
        if (rt_assert(f != NULL) == false)
        {
            return false;
        }

        for (uint i=0; i<traj_size; i++)
        {
            if (rt_assert(trajectory[i].isValid()) == false)
            {
                fclose(f);
                return false;
            }

            if (is_comma_separated)
            {
                fprintf(f, "%.05f,%.05f,%.05f,%.05f\n",
                        trajectory[i].getTime(), (trajectory[i].getX())[0],
                        (trajectory[i].getXd())[0], (trajectory[i].getXdd())[0]);
            }
            else
            {
                fprintf(f, "%.05f %.05f %.05f %.05f\n",
                        trajectory[i].getTime(), (trajectory[i].getX())[0],
                        (trajectory[i].getXd())[0], (trajectory[i].getXdd())[0]);
            }
        }

        fclose(f);

        return true;
    }

    /**
     * NON-REAL-TIME!!!\n
     * Extract an N-dimensional trajectory from a text file.
     *
     * @param file_path File path to extract trajectory information from
     * @param trajectory (Blank/empty) trajectory to be filled-in
     * @return Success or failure
     */
    bool DataIO::extractNDTrajectory(const char* file_path, Trajectory& trajectory)
    {
        bool is_real_time   = false;

        if (rt_assert(DataIO::isValid()) == false)
        {
            return false;
        }

        if (trajectory.size() != 0)
        {
            trajectory.clear();
        }

        MatrixXxXPtr    file_content_as_matrix;
        if (rt_assert(DataIO::readMatrixFromFile(file_path, file_content_as_matrix)) == false)
        {
            return false;
        }
        if (rt_assert(((file_content_as_matrix->cols() - 1) % 3) == 0) == false)
        {
            return false;
        }

        // acquiring dimension size of the DMPState
        uint    N   = ((file_content_as_matrix->cols() - 1) / 3);
        if (rt_assert(N <= MAX_DMP_NUM_DIMENSIONS) == false)
        {
            return false;
        }

        for (uint nr=0; nr<file_content_as_matrix->rows(); nr++)
        {
            DMPState    state(file_content_as_matrix->block(nr,        1 , 1, N),
                              file_content_as_matrix->block(nr, (   N +1), 1, N),
                              file_content_as_matrix->block(nr, ((2*N)+1), 1, N),
                              (*file_content_as_matrix)(nr, 0),
                              rt_assertor);
            trajectory.push_back(state);
        }

        return true;
    }

    /**
     * NON-REAL-TIME!!!\n
     * Extract a Cartesian coordinate trajectory from a text file.
     *
     * @param file_path File path to extract trajectory information from
     * @param cart_coord_trajectory (Blank/empty) Cartesian coordinate trajectory to be filled-in
     * @param is_comma_separated If true, trajectory fields are separated by commas;\n
     *        If false, trajectory fields are separated by white-space characters (default is false)
     * @return Success or failure
     */
    bool DataIO::extractCartCoordTrajectory(const char* file_path,
                                            Trajectory& cart_coord_trajectory,
                                            bool is_comma_separated)
    {
        bool is_real_time   = false;

        if (rt_assert(DataIO::isValid()) == false)
        {
            return false;
        }

        if (cart_coord_trajectory.size() != 0)
        {
            cart_coord_trajectory.clear();
        }

        FILE*       f       = fopen(file_path, "rt");
        if (rt_assert(f != NULL) == false)
        {
            return false;
        }

        char        line[1000];
        double      time;
        VectorNPtr  X;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, X, 3, 1)) == false)
        {
            return false;
        }
        VectorNPtr  Xd;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Xd, 3, 1)) == false)
        {
            return false;
        }
        VectorNPtr  Xdd;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Xdd, 3, 1)) == false)
        {
            return false;
        }

        while(fgets(line, 1000, f) != NULL)
        {
            if (is_comma_separated)
            {
                sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &time,
                       &((*X)[0])  , &((*X)[1])  , &((*X)[2])   ,
                       &((*Xd)[0]) , &((*Xd)[1]) , &((*Xd)[2])  ,
                       &((*Xdd)[0]), &((*Xdd)[1]), &((*Xdd)[2]));
            }
            else
            {
                sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &time,
                       &((*X)[0])  , &((*X)[1])  , &((*X)[2])   ,
                       &((*Xd)[0]) , &((*Xd)[1]) , &((*Xd)[2])  ,
                       &((*Xdd)[0]), &((*Xdd)[1]), &((*Xdd)[2]));
            }
            DMPState    cart_state(*X, *Xd, *Xdd, time, rt_assertor);
            cart_coord_trajectory.push_back(cart_state);
        }

        fclose(f);

        return true;
    }

    /**
     * NON-REAL-TIME!!!\n
     * Write a Cartesian coordinate trajectory's information/content to a text file.
     *
     * @param cart_coord_trajectory Cartesian coordinate trajectory to be written to a file
     * @param file_path File path where to write the Cartesian coordinate trajectory's information onto
     * @param is_comma_separated If true, Cartesian coordinate trajectory fields will be written with comma-separation;\n
     *        If false, Cartesian coordinate trajectory fields will be written with white-space-separation (default is true)
     * @return Success or failure
     */
    bool DataIO::writeCartCoordTrajectoryToFile(const Trajectory& cart_coord_trajectory,
                                                const char* file_path,
                                                bool is_comma_separated)
    {
        bool is_real_time   = false;

        if (rt_assert(DataIO::isValid()) == false)
        {
            return false;
        }
        uint    traj_size   = cart_coord_trajectory.size();
        if (rt_assert(rt_assert(traj_size > 0) && rt_assert(traj_size <= MAX_TRAJ_SIZE)) == false)
        {
            return false;
        }

        FILE*   f           = fopen(file_path, "w");
        if (rt_assert(f != NULL) == false)
        {
            return false;
        }

        double  time, x, y, z, xd, yd, zd, xdd, ydd, zdd;
        VectorNPtr  X;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, X, 3, 1)) == false)
        {
            return false;
        }
        VectorNPtr  Xd;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Xd, 3, 1)) == false)
        {
            return false;
        }
        VectorNPtr  Xdd;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Xdd, 3, 1)) == false)
        {
            return false;
        }

        for (uint i=0; i<traj_size; ++i)
        {
            if (rt_assert(cart_coord_trajectory[i].isValid()) == false)
            {
                fclose(f);
                return false;
            }

            time        = cart_coord_trajectory[i].getTime();
            *X          = cart_coord_trajectory[i].getX();
            *Xd         = cart_coord_trajectory[i].getXd();
            *Xdd        = cart_coord_trajectory[i].getXdd();
            x           = (*X)[0];
            y           = (*X)[1];
            z           = (*X)[2];
            xd          = (*Xd)[0];
            yd          = (*Xd)[1];
            zd          = (*Xd)[2];
            xdd         = (*Xdd)[0];
            ydd         = (*Xdd)[1];
            zdd         = (*Xdd)[2];

            if (is_comma_separated)
            {
                fprintf(f, "%.05f,%.05f,%.05f,%.05f,%.05f,%.05f,%.05f,%.05f,%.05f,%.05f\n", time,
                        x  , y  , z  ,
                        xd , yd , zd ,
                        xdd, ydd, zdd);
            }
            else
            {
                fprintf(f, "%.05f %.05f %.05f %.05f %.05f %.05f %.05f %.05f %.05f %.05f\n", time,
                        x  , y  , z  ,
                        xd , yd , zd ,
                        xdd, ydd, zdd);
            }
        }

        fclose(f);

        return true;
    }

    /**
     * NON-REAL-TIME!!!\n
     * Extract a Quaternion trajectory from a text file.
     *
     * @param file_path File path to extract trajectory information from
     * @param quat_trajectory (Blank/empty) Quaternion trajectory to be filled-in
     * @param is_comma_separated If true, trajectory fields are separated by commas;\n
     *        If false, trajectory fields are separated by white-space characters (default is false)
     * @return Success or failure
     */
    bool DataIO::extractQuaternionTrajectory(const char* file_path,
                                             QuaternionTrajectory& quat_trajectory,
                                             bool is_comma_separated)
    {
        bool is_real_time   = false;

        if (rt_assert(DataIO::isValid()) == false)
        {
            return false;
        }

        if (quat_trajectory.size() != 0)
        {
            quat_trajectory.clear();
        }

        FILE*       f       = fopen(file_path, "rt");
        if (rt_assert(f != NULL) == false)
        {
            return false;
        }

        char        line[1000];
        double      time;
        VectorNPtr  Q;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Q, 4, 1)) == false)
        {
            return false;
        }
        VectorNPtr  Qd;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Qd, 4, 1)) == false)
        {
            return false;
        }
        VectorNPtr  Qdd;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Qdd, 4, 1)) == false)
        {
            return false;
        }
        VectorNPtr  omega;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, omega, 3, 1)) == false)
        {
            return false;
        }
        VectorNPtr  omegad;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, omegad, 3, 1)) == false)
        {
            return false;
        }

        while(fgets(line, 1000, f) != NULL)
        {
            if (is_comma_separated)
            {
                sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                       &time,
                       &((*Q)[0])     , &((*Q)[1])     , &((*Q)[2])    , &((*Q)[3])  ,
                       &((*Qd)[0])    , &((*Qd)[1])    , &((*Qd)[2])   , &((*Qd)[3]) ,
                       &((*Qdd)[0])   , &((*Qdd)[1])   , &((*Qdd)[2])  , &((*Qdd)[3]),
                       &((*omega)[0]) , &((*omega)[1]) , &((*omega)[2]),
                       &((*omegad)[0]), &((*omegad)[1]), &((*omegad)[2]));
            }
            else
            {
                sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                       &time,
                       &((*Q)[0])     , &((*Q)[1])     , &((*Q)[2])    , &((*Q)[3])  ,
                       &((*Qd)[0])    , &((*Qd)[1])    , &((*Qd)[2])   , &((*Qd)[3]) ,
                       &((*Qdd)[0])   , &((*Qdd)[1])   , &((*Qdd)[2])  , &((*Qdd)[3]),
                       &((*omega)[0]) , &((*omega)[1]) , &((*omega)[2]),
                       &((*omegad)[0]), &((*omegad)[1]), &((*omegad)[2]));
            }
            QuaternionDMPState  quat_state(*Q, *Qd, *Qdd, *omega, *omegad, time, rt_assertor);
            quat_trajectory.push_back(quat_state);
        }

        fclose(f);

        return true;
    }

    /**
     * NON-REAL-TIME!!!\n
     * Extract a set of N-dimensional trajectory(ies) from text file(s).
     *
     * @param dir_or_file_path Two possibilities:\n
     *        If want to extract a set of N-dimensional trajectories, \n
     *        then specify the directory path containing the text files, \n
     *        each text file represents a single N-dimensional trajectory.
     *        If want to extract just a single N-dimensional trajectory, \n
     *        then specify the file path to the text file representing such trajectory.
     * @param nd_trajectory_set (Blank/empty) set of N-dimensional trajectories to be filled-in
     * @param max_num_trajs [optional] Maximum number of trajectories that could be loaded into a trajectory set (default is 500)
     * @return Success or failure
     */
    bool DataIO::extractSetNDTrajectories(const char* dir_or_file_path,
                                          TrajectorySet& nd_trajectory_set,
                                          uint max_num_trajs)
    {
        bool is_real_time   = false;

        if (rt_assert(DataIO::isValid()) == false)
        {
            return false;
        }

        if (nd_trajectory_set.size() != 0)
        {
            nd_trajectory_set.clear();
        }

        char            var_file_path[1000];
        TrajectoryPtr   nd_trajectory;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, nd_trajectory,
                                                  0)) == false)
        {
            return false;
        }

        if (file_type(dir_or_file_path) == _DIR_)       // if it is a directory
        {
            uint        i   = 1;
            sprintf(var_file_path, "%s/%d.txt", dir_or_file_path, i);

            while ((file_type(var_file_path) == _REG_) &&   // while file exists
                   (i <= max_num_trajs))
            {
                if (rt_assert(extractNDTrajectory(var_file_path,
                                                  (*nd_trajectory))) == false)
                {
                    return false;
                }
                nd_trajectory_set.push_back(*nd_trajectory);
                i++;
                sprintf(var_file_path, "%s/%d.txt", dir_or_file_path, i);
            }
        }
        else if (file_type(dir_or_file_path) == _REG_)  // if it is a (text) file
        {
            if (rt_assert(extractNDTrajectory(dir_or_file_path,
                                              (*nd_trajectory))) == false)
            {
                return false;
            }
            nd_trajectory_set.push_back(*nd_trajectory);
        }
        else
        {
            return false;
        }

        return true;
    }

    /**
     * NON-REAL-TIME!!!\n
     * Extract a set of Cartesian coordinate trajectory(ies) from text file(s).
     *
     * @param dir_or_file_path Two possibilities:\n
     *        If want to extract a set of Cartesian coordinate trajectories, \n
     *        then specify the directory path containing the text files, \n
     *        each text file represents a single Cartesian coordinate trajectory.
     *        If want to extract just a single Cartesian coordinate trajectory, \n
     *        then specify the file path to the text file representing such trajectory.
     * @param cart_coord_trajectory_set (Blank/empty) set of Cartesian coordinate trajectories to be filled-in
     * @param is_comma_separated If true, trajectory fields are separated by commas;\n
     *        If false, trajectory fields are separated by white-space characters (default is false)
     * @param max_num_trajs [optional] Maximum number of trajectories that could be loaded into a trajectory set (default is 500)
     * @return Success or failure
     */
    bool DataIO::extractSetCartCoordTrajectories(const char* dir_or_file_path,
                                                 TrajectorySet& cart_coord_trajectory_set,
                                                 bool is_comma_separated,
                                                 uint max_num_trajs)
    {
        bool is_real_time   = false;

        if (rt_assert(DataIO::isValid()) == false)
        {
            return false;
        }

        if (cart_coord_trajectory_set.size() != 0)
        {
            cart_coord_trajectory_set.clear();
        }

        char            var_file_path[1000];
        TrajectoryPtr   cartesian_trajectory;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, cartesian_trajectory,
                                                  0)) == false)
        {
            return false;
        }

        if (file_type(dir_or_file_path) == _DIR_)       // if it is a directory
        {
            uint        i   = 1;
            sprintf(var_file_path, "%s/%d.txt", dir_or_file_path, i);

            while ((file_type(var_file_path) == _REG_) &&   // while file exists
                   (i <= max_num_trajs))
            {
                if (rt_assert(extractCartCoordTrajectory(var_file_path,
                                                         (*cartesian_trajectory),
                                                         is_comma_separated)) == false)
                {
                    return false;
                }
                cart_coord_trajectory_set.push_back(*cartesian_trajectory);
                i++;
                sprintf(var_file_path, "%s/%d.txt", dir_or_file_path, i);
            }
        }
        else if (file_type(dir_or_file_path) == _REG_)  // if it is a (text) file
        {
            if (rt_assert(extractCartCoordTrajectory(dir_or_file_path,
                                                     (*cartesian_trajectory),
                                                     is_comma_separated)) == false)
            {
                return false;
            }
            cart_coord_trajectory_set.push_back(*cartesian_trajectory);
        }
        else
        {
            return false;
        }

        return true;
    }

    /**
     * NON-REAL-TIME!!!\n
     * Write a set of Cartesian coordinate trajectories to text files contained in one folder/directory.
     *
     * @param cart_coord_trajectory_set Set of Cartesian coordinate trajectories to be written into the specified directory
     * @param dir_path Directory path where to write the set of Cartesian coordinate trajectories representation into
     * @param is_comma_separated If true, Cartesian coordinate trajectory fields will be written with comma-separation;\n
     *        If false, Cartesian coordinate trajectory fields will be written with white-space-separation (default is true)
     * @return Success or failure
     */
    bool DataIO::writeSetCartCoordTrajectoriesToDir(const TrajectorySet& cart_coord_trajectory_set,
                                                    const char* dir_path,
                                                    bool is_comma_separated)
    {
        bool is_real_time   = false;

        if (rt_assert(DataIO::isValid()) == false)
        {
            return false;
        }
        uint            N_traj      = cart_coord_trajectory_set.size();
        if (rt_assert(N_traj > 0) == false)
        {
            return false;
        }
        if (file_type(dir_path) != _DIR_)
        {
            mkdir(dir_path, ACCESSPERMS);
        }

        char            var_file_path[1000];
        for (uint i=0; i<N_traj; ++i)
        {
            sprintf(var_file_path, "%s/%d.txt", dir_path, (i+1));

            if (rt_assert(writeCartCoordTrajectoryToFile(cart_coord_trajectory_set[i],
                                                         var_file_path,
                                                         is_comma_separated)) == false)
            {
                return false;
            }
        }

        return true;
    }

    /**
     * NON-REAL-TIME!!!\n
     * Extract a set of Quaternion trajectory(ies) from text file(s).
     *
     * @param dir_or_file_path Two possibilities:\n
     *        If want to extract a set of Quaternion trajectories, \n
     *        then specify the directory path containing the text files, \n
     *        each text file represents a single Quaternion trajectory.
     *        If want to extract just a single Quaternion trajectory, \n
     *        then specify the file path to the text file representing such trajectory.
     * @param quat_trajectory_set (Blank/empty) set of Quaternion trajectories to be filled-in
     * @param is_comma_separated If true, trajectory fields are separated by commas;\n
     *        If false, trajectory fields are separated by white-space characters (default is false)
     * @param max_num_trajs [optional] Maximum number of trajectories that could be loaded into a trajectory set (default is 500)
     * @return Success or failure
     */
    bool DataIO::extractSetQuaternionTrajectories(const char* dir_or_file_path,
                                                  QuaternionTrajectorySet& quat_trajectory_set,
                                                  bool is_comma_separated,
                                                  uint max_num_trajs)
    {
        bool is_real_time   = false;

        if (rt_assert(DataIO::isValid()) == false)
        {
            return false;
        }

        if (quat_trajectory_set.size() != 0)
        {
            quat_trajectory_set.clear();
        }

        char                    var_file_path[1000];
        QuaternionTrajectoryPtr quat_trajectory;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, quat_trajectory,
                                                  0)) == false)
        {
            return false;
        }

        if (file_type(dir_or_file_path) == _DIR_)       // if it is a directory
        {
            uint        i   = 1;
            sprintf(var_file_path, "%s/%d.txt", dir_or_file_path, i);

            while ((file_type(var_file_path) == _REG_) &&   // while file exists
                   (i <= max_num_trajs))
            {
                if (rt_assert(extractQuaternionTrajectory(var_file_path,
                                                          (*quat_trajectory),
                                                          is_comma_separated)) == false)
                {
                    return false;
                }
                quat_trajectory_set.push_back(*quat_trajectory);
                i++;
                sprintf(var_file_path, "%s/%d.txt", dir_or_file_path, i);
            }
        }
        else if (file_type(dir_or_file_path) == _REG_)  // if it is a (text) file
        {
            if (rt_assert(extractQuaternionTrajectory(dir_or_file_path,
                                                     (*quat_trajectory),
                                                     is_comma_separated)) == false)
            {
                return false;
            }
            quat_trajectory_set.push_back(*quat_trajectory);
        }
        else
        {
            return false;
        }

        return true;
    }

    /**
     * NON-REAL-TIME!!!\n
     * If you want to maintain real-time performance,
     * do/call this function from a separate (non-real-time) thread!!!\n
     * Extract all DMP training demonstrations.
     *
     * @param in_data_dir_path The input data directory
     * @param demo_group_set_global [to-be-filled/returned] Extracted demonstrations \n
     *                              in the Cartesian global coordinate system
     * @param N_demo_settings [to-be-filled/returned] Total number of different demonstration settings \n
     *                        (e.g. total number of different obstacle positions, in obstacle avoidance case)
     * @param max_num_trajs_per_setting [optional] Maximum number of trajectories that could be loaded into a trajectory set \n
     *                                  of a demonstration setting \n
     *                                  (e.g. setting = obstacle position in obstacle avoidance context) \n
     *                                  (default is 500)
     * @param selected_obs_avoid_setting_numbers [optional] If we are just interested in learning a subset of the obstacle avoidance demonstration settings,\n
     *                                           then specify the numbers (IDs) of the setting here (as a vector).\n
     *                                           If not specified, then all settings will be considered.
     * @param demo_group_set_dmp_unroll_init_params [optional] DMP unroll initialization parameters of each trajectory in \n
     *                                              the extracted demonstrations (to be returned)
     * @return Success or failure
     */
    bool DataIO::extractSetCartCoordDemonstrationGroups(const char* in_data_dir_path,
                                                        DemonstrationGroupSet& demo_group_set_global,
                                                        uint& N_demo_settings,
                                                        uint max_num_trajs_per_setting,
                                                        std::vector<uint>* selected_obs_avoid_setting_numbers,
                                                        VecVecDMPUnrollInitParams* demo_group_set_dmp_unroll_init_params)
    {
        bool is_real_time                       = false;

        // count total number of different obstacle avoidance demonstration settings
        // (or total number of different obstacle positions):
        double  N_total_existing_demo_settings  = countNumberedDirectoriesUnderDirectory(in_data_dir_path);
        if (rt_assert(N_total_existing_demo_settings >= 1) == false)
        {
            return false;
        }

        if (selected_obs_avoid_setting_numbers == NULL) // use all settings
        {
            N_demo_settings                     = N_total_existing_demo_settings;
        }
        else
        {
            N_demo_settings                     = selected_obs_avoid_setting_numbers->size();
            if (rt_assert(N_total_existing_demo_settings >= N_demo_settings) == false)
            {
                return false;
            }
        }

        TrajectorySetPtr                        demo_group_global;
        if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, demo_group_global, 0)) == false)
        {
            return false;
        }

        char    in_demo_group_dir_path[1000];
        for (uint n=1; n<=N_demo_settings; n++)
        {
            if (selected_obs_avoid_setting_numbers == NULL) // use all settings
            {
                sprintf(in_demo_group_dir_path, "%s/%u/endeff_trajs/", in_data_dir_path, n);
            }
            else
            {
                if (rt_assert(((*selected_obs_avoid_setting_numbers)[n-1] >  0) &&
                              ((*selected_obs_avoid_setting_numbers)[n-1] <= N_total_existing_demo_settings)) == false)
                {
                    return false;
                }
                sprintf(in_demo_group_dir_path, "%s/%u/endeff_trajs/", in_data_dir_path, (*selected_obs_avoid_setting_numbers)[n-1]);
            }

            // Load the demonstrations for each particular obstacle setting:
            if (rt_assert(DataIO::extractSetCartCoordTrajectories(in_demo_group_dir_path,
                                                                  *demo_group_global,
                                                                  false,
                                                                  max_num_trajs_per_setting)) == false)
            {
                return false;
            }

            demo_group_set_global.push_back(*demo_group_global);
        }

        if (rt_assert(demo_group_set_global.size() == N_demo_settings) == false)
        {
            return false;
        }

        if (demo_group_set_dmp_unroll_init_params != NULL)
        {
            demo_group_set_dmp_unroll_init_params->clear();

            VecDMPUnrollInitParamsPtr   demo_group_dmp_unroll_init_params;
            if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, demo_group_dmp_unroll_init_params, 0)) == false)
            {
                return false;
            }

            for (uint i=0; i<demo_group_set_global.size(); i++)
            {
                demo_group_dmp_unroll_init_params->clear();

                for (uint j=0; j<demo_group_set_global[i].size(); j++)
                {
                    demo_group_dmp_unroll_init_params->push_back(DMPUnrollInitParams(demo_group_set_global[i][j], rt_assertor));
                }

                if (rt_assert(demo_group_dmp_unroll_init_params->size() == demo_group_set_global[i].size()) == false)
                {
                    return false;
                }

                demo_group_set_dmp_unroll_init_params->push_back(*demo_group_dmp_unroll_init_params);
            }

            if (rt_assert(demo_group_set_dmp_unroll_init_params->size() == N_demo_settings) == false)
            {
                return false;
            }
        }

        return true;
    }

    /**
     * NON-REAL-TIME!!!\n
     * Extract an arbitrary-length vector of 3D points from a text file.
     *
     * @param file_path File path to extract 3D points information from
     * @param points (Blank/empty) 3D points vector to be filled-in
     * @param is_comma_separated If true, fields are separated by commas;\n
     *        If false, fields are separated by white-space characters (default is false)
     * @return Success or failure
     */
    bool DataIO::extract3DPoints(const char* file_path, Points& points, bool is_comma_separated)
    {
        bool is_real_time   = false;

        if (rt_assert(DataIO::isValid()) == false)
        {
            return false;
        }

        if (points.size() != 0)
        {
            points.clear();
        }

        FILE*   f   = fopen(file_path, "rt");
        if (rt_assert(f != NULL) == false)
        {
            return false;
        }

        char        line[1000];
        Vector3     point;

        while(fgets(line, 1000, f) != NULL)
        {
            double      time;
            if (is_comma_separated)
            {
                sscanf(line, "%lf,%lf,%lf",
                       &(point[0]), &(point[1]), &(point[2]));
            }
            else
            {
                sscanf(line, "%lf %lf %lf",
                       &(point[0]), &(point[1]), &(point[2]));
            }
            points.push_back(point);
        }

        fclose(f);

        return true;
    }

    /**
     * Overloading of templated function readMatrixFromFile() for (T==int).
     */
    bool DataIO::readMatrixFromFile(const char* file_path, int& matrix_data_structure)
    {
        bool is_real_time   = false;

        if (rt_assert(DataIO::isValid()) == false)
        {
            return false;
        }

        FILE*   f   = fopen(file_path, "rt");
        if (rt_assert(f != NULL) == false)
        {
            return false;
        }

        char    line[1000];
        if (rt_assert(fgets(line, 1000, f) != NULL))
        {
            sscanf(line, "%d", &matrix_data_structure);
        }
        else
        {
            fclose(f);
            return false;
        }

        if (rt_assert(fgets(line, 1000, f) == NULL) == false)
        {
            fclose(f);
            return false;
        }

        fclose(f);

        return true;
    }

    /**
     * Overloading of templated function readMatrixFromFile() for (T==double).
     */
    bool DataIO::readMatrixFromFile(const char* file_path, double& matrix_data_structure)
    {
        bool is_real_time   = false;

        if (rt_assert(DataIO::isValid()) == false)
        {
            return false;
        }

        FILE*   f   = fopen(file_path, "rt");
        if (rt_assert(f != NULL) == false)
        {
            return false;
        }

        char    line[1000];
        if (rt_assert(fgets(line, 1000, f) != NULL))
        {
            sscanf(line, "%lf", &matrix_data_structure);
        }
        else
        {
            fclose(f);
            return false;
        }

        if (rt_assert(fgets(line, 1000, f) == NULL) == false)
        {
            fclose(f);
            return false;
        }

        fclose(f);

        return true;
    }

    DataIO::~DataIO()
    {}

}
