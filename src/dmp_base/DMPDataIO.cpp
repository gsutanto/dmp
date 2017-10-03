#include "amd_clmc_dmp/dmp_base/DMPDataIO.h"

namespace dmp
{

    DMPDataIO::DMPDataIO():
        DataIO(), transform_sys(NULL), func_approx(NULL),
        tau_trajectory_buffer(new MatrixTx2()),
        transform_sys_state_local_trajectory_buffer(new MatrixTxS()),
        transform_sys_state_global_trajectory_buffer(new MatrixTxS()),
        goal_state_local_trajectory_buffer(new MatrixTxS()),
        goal_state_global_trajectory_buffer(new MatrixTxS()),
        steady_state_goal_position_local_trajectory_buffer(new MatrixTxNp1()),
        steady_state_goal_position_global_trajectory_buffer(new MatrixTxNp1()),
        basis_functions_trajectory_buffer(new MatrixTxMp1()),
        forcing_term_trajectory_buffer(new MatrixTxNp1()),
        transform_sys_ct_acc_trajectory_buffer(new MatrixTxNp1()),
        save_data_buffer(new MatrixTxSBC()),
        dmp_num_dimensions(0), model_size(0)
    {
        DMPDataIO::reset();
    }

    /**
     * @param transformation_system Transformation System, whose data will be recorded
     * @param function_approximator Function Approximator, whose data will be recorded
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    DMPDataIO::DMPDataIO(TransformationSystem* transformation_system,
                         FunctionApproximator* function_approximator,
                         RealTimeAssertor* real_time_assertor):
        DataIO(real_time_assertor),
        transform_sys(transformation_system), func_approx(function_approximator),
        tau_trajectory_buffer(new MatrixTx2()),
        transform_sys_state_local_trajectory_buffer(new MatrixTxS()),
        transform_sys_state_global_trajectory_buffer(new MatrixTxS()),
        goal_state_local_trajectory_buffer(new MatrixTxS()),
        goal_state_global_trajectory_buffer(new MatrixTxS()),
        steady_state_goal_position_global_trajectory_buffer(new MatrixTxNp1()),
        steady_state_goal_position_local_trajectory_buffer(new MatrixTxNp1()),
        basis_functions_trajectory_buffer(new MatrixTxMp1()),
        forcing_term_trajectory_buffer(new MatrixTxNp1()),
        transform_sys_ct_acc_trajectory_buffer(new MatrixTxNp1()),
        save_data_buffer(new MatrixTxSBC()),
        dmp_num_dimensions(func_approx->getDMPNumDimensions()),
        model_size(func_approx->getModelSize())
    {
        DMPDataIO::reset();
    }

    /**
     * Initialize/reset the data buffers and counters (usually done everytime before start of data saving).
     *
     * @return Success or failure
     */
    bool DMPDataIO::reset()
    {
        if (rt_assert((rt_assert(resizeAndReset(tau_trajectory_buffer,
                                                MAX_TRAJ_SIZE, 2)))             &&
                      (rt_assert(resizeAndReset(transform_sys_state_local_trajectory_buffer,
                                                MAX_TRAJ_SIZE,
                                                (1+(3*dmp_num_dimensions)))))   &&
                      (rt_assert(resizeAndReset(transform_sys_state_global_trajectory_buffer,
                                                MAX_TRAJ_SIZE,
                                                (1+(3*dmp_num_dimensions)))))   &&
                      (rt_assert(resizeAndReset(goal_state_local_trajectory_buffer,
                                                MAX_TRAJ_SIZE,
                                                (1+(3*dmp_num_dimensions)))))   &&
                      (rt_assert(resizeAndReset(goal_state_global_trajectory_buffer,
                                                MAX_TRAJ_SIZE,
                                                (1+(3*dmp_num_dimensions)))))   &&
                      (rt_assert(resizeAndReset(steady_state_goal_position_local_trajectory_buffer,
                                                MAX_TRAJ_SIZE,
                                                (1+dmp_num_dimensions))))       &&
                      (rt_assert(resizeAndReset(steady_state_goal_position_global_trajectory_buffer,
                                                MAX_TRAJ_SIZE,
                                                (1+dmp_num_dimensions))))       &&
                      (rt_assert(resizeAndReset(basis_functions_trajectory_buffer,
                                                MAX_TRAJ_SIZE,
                                                (1+model_size))))               &&
                      (rt_assert(resizeAndReset(forcing_term_trajectory_buffer,
                                                MAX_TRAJ_SIZE,
                                                (1+dmp_num_dimensions))))       &&
                      (rt_assert(resizeAndReset(transform_sys_ct_acc_trajectory_buffer,
                                                MAX_TRAJ_SIZE,
                                                (1+dmp_num_dimensions))))       &&
                      (rt_assert(resizeAndReset(save_data_buffer,
                                                MAX_TRAJ_SIZE,
                                                MAX_SAVE_BUFFER_COL_SIZE)))) == false)
        {
            return false;
        }

        trajectory_step_count   = 0;

        return true;
    }

    /**
     * Checks whether this data recorder is valid or not.
     *
     * @return Data recorder is valid (true) or data recorder is invalid (false)
     */
    bool DMPDataIO::isValid()
    {
        if (rt_assert(DataIO::isValid()) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(transform_sys  != NULL)) &&
                      (rt_assert(func_approx    != NULL))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(transform_sys->isValid())) &&
                      (rt_assert(func_approx->isValid()))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(tau_trajectory_buffer                              != NULL)) &&
                      (rt_assert(transform_sys_state_local_trajectory_buffer        != NULL)) &&
                      (rt_assert(transform_sys_state_global_trajectory_buffer       != NULL)) &&
                      (rt_assert(goal_state_local_trajectory_buffer                 != NULL)) &&
                      (rt_assert(goal_state_global_trajectory_buffer                != NULL)) &&
                      (rt_assert(steady_state_goal_position_local_trajectory_buffer != NULL)) &&
                      (rt_assert(steady_state_goal_position_global_trajectory_buffer!= NULL)) &&
                      (rt_assert(basis_functions_trajectory_buffer                  != NULL)) &&
                      (rt_assert(forcing_term_trajectory_buffer                     != NULL)) &&
                      (rt_assert(transform_sys_ct_acc_trajectory_buffer      != NULL)) &&
                      (rt_assert(save_data_buffer                                   != NULL))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(dmp_num_dimensions == transform_sys->getDMPNumDimensions())) &&
                      (rt_assert(dmp_num_dimensions == func_approx->getDMPNumDimensions()))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(dmp_num_dimensions > 0)) &&
                      (rt_assert(dmp_num_dimensions <= MAX_DMP_NUM_DIMENSIONS))) == false)
        {
            return false;
        }
        if (rt_assert(model_size == func_approx->getModelSize()) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(model_size > 0)) && (rt_assert(model_size <= MAX_MODEL_SIZE))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(trajectory_step_count >= 0)) &&
                      (rt_assert(trajectory_step_count <= MAX_TRAJ_SIZE))) == false)
        {
            return false;
        }
        return true;
    }

    /**
     * Copy the supplied variables into current column of the buffers.
     *
     * @param logged_dmp_variables_buffer DMP variables that will be logged
     * @return Success or failure
     */
    bool DMPDataIO::collectTrajectoryDataSet(const LoggedDMPVariables& logged_dmp_variables_buffer)
    {
        // pre-condition checking
        if (rt_assert(DMPDataIO::isValid()) == false)
        {
            return false;
        }
        // trajectory_step_count must be STRICTLY less than MAX_TRAJ_SIZE,
        // otherwise segmentation fault could happen:
        if (rt_assert(trajectory_step_count < MAX_TRAJ_SIZE) == false)
        {
            return false;
        }
        // input checking
        if (rt_assert(logged_dmp_variables_buffer.isValid()) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(dmp_num_dimensions == logged_dmp_variables_buffer.getDMPNumDimensions())) &&
                      (rt_assert(model_size == logged_dmp_variables_buffer.getModelSize()))) == false)
        {
            return false;
        }

        (*tau_trajectory_buffer)(trajectory_step_count,0)   = logged_dmp_variables_buffer.transform_sys_state_local.getTime();
        (*tau_trajectory_buffer)(trajectory_step_count,1)   = logged_dmp_variables_buffer.tau;

        (*transform_sys_state_local_trajectory_buffer)(trajectory_step_count,0) = logged_dmp_variables_buffer.transform_sys_state_local.getTime();
        transform_sys_state_local_trajectory_buffer->block(trajectory_step_count,(1+(0*dmp_num_dimensions)),1,dmp_num_dimensions)   = logged_dmp_variables_buffer.transform_sys_state_local.getX().transpose().block(0,0,1,dmp_num_dimensions);
        transform_sys_state_local_trajectory_buffer->block(trajectory_step_count,(1+(1*dmp_num_dimensions)),1,dmp_num_dimensions)   = logged_dmp_variables_buffer.transform_sys_state_local.getXd().transpose().block(0,0,1,dmp_num_dimensions);
        transform_sys_state_local_trajectory_buffer->block(trajectory_step_count,(1+(2*dmp_num_dimensions)),1,dmp_num_dimensions)   = logged_dmp_variables_buffer.transform_sys_state_local.getXdd().transpose().block(0,0,1,dmp_num_dimensions);

        (*transform_sys_state_global_trajectory_buffer)(trajectory_step_count,0)= logged_dmp_variables_buffer.transform_sys_state_global.getTime();
        transform_sys_state_global_trajectory_buffer->block(trajectory_step_count,(1+(0*dmp_num_dimensions)),1,dmp_num_dimensions)  = logged_dmp_variables_buffer.transform_sys_state_global.getX().transpose().block(0,0,1,dmp_num_dimensions);
        transform_sys_state_global_trajectory_buffer->block(trajectory_step_count,(1+(1*dmp_num_dimensions)),1,dmp_num_dimensions)  = logged_dmp_variables_buffer.transform_sys_state_global.getXd().transpose().block(0,0,1,dmp_num_dimensions);
        transform_sys_state_global_trajectory_buffer->block(trajectory_step_count,(1+(2*dmp_num_dimensions)),1,dmp_num_dimensions)  = logged_dmp_variables_buffer.transform_sys_state_global.getXdd().transpose().block(0,0,1,dmp_num_dimensions);

        (*goal_state_local_trajectory_buffer)(trajectory_step_count,0)  = logged_dmp_variables_buffer.transform_sys_state_local.getTime();
        goal_state_local_trajectory_buffer->block(trajectory_step_count,(1+(0*dmp_num_dimensions)),1,dmp_num_dimensions)    = logged_dmp_variables_buffer.goal_state_local.getX().transpose().block(0,0,1,dmp_num_dimensions);
        goal_state_local_trajectory_buffer->block(trajectory_step_count,(1+(1*dmp_num_dimensions)),1,dmp_num_dimensions)    = logged_dmp_variables_buffer.goal_state_local.getXd().transpose().block(0,0,1,dmp_num_dimensions);
        goal_state_local_trajectory_buffer->block(trajectory_step_count,(1+(2*dmp_num_dimensions)),1,dmp_num_dimensions)    = logged_dmp_variables_buffer.goal_state_local.getXdd().transpose().block(0,0,1,dmp_num_dimensions);

        (*goal_state_global_trajectory_buffer)(trajectory_step_count,0) = logged_dmp_variables_buffer.transform_sys_state_global.getTime();
        goal_state_global_trajectory_buffer->block(trajectory_step_count,(1+(0*dmp_num_dimensions)),1,dmp_num_dimensions)   = logged_dmp_variables_buffer.goal_state_global.getX().transpose().block(0,0,1,dmp_num_dimensions);
        goal_state_global_trajectory_buffer->block(trajectory_step_count,(1+(1*dmp_num_dimensions)),1,dmp_num_dimensions)   = logged_dmp_variables_buffer.goal_state_global.getXd().transpose().block(0,0,1,dmp_num_dimensions);
        goal_state_global_trajectory_buffer->block(trajectory_step_count,(1+(2*dmp_num_dimensions)),1,dmp_num_dimensions)   = logged_dmp_variables_buffer.goal_state_global.getXdd().transpose().block(0,0,1,dmp_num_dimensions);

        (*steady_state_goal_position_local_trajectory_buffer)(trajectory_step_count,0)  = logged_dmp_variables_buffer.transform_sys_state_local.getTime();
        steady_state_goal_position_local_trajectory_buffer->block(trajectory_step_count,1,1,dmp_num_dimensions) = logged_dmp_variables_buffer.steady_state_goal_position_local.transpose().block(0,0,1,dmp_num_dimensions);

        (*steady_state_goal_position_global_trajectory_buffer)(trajectory_step_count,0) = logged_dmp_variables_buffer.transform_sys_state_global.getTime();
        steady_state_goal_position_global_trajectory_buffer->block(trajectory_step_count,1,1,dmp_num_dimensions)   = logged_dmp_variables_buffer.steady_state_goal_position_global.transpose().block(0,0,1,dmp_num_dimensions);

        (*basis_functions_trajectory_buffer)(trajectory_step_count,0)   = logged_dmp_variables_buffer.transform_sys_state_local.getTime();
        basis_functions_trajectory_buffer->block(trajectory_step_count,1,1,model_size)  = logged_dmp_variables_buffer.basis_functions.transpose().block(0,0,1,model_size);

        (*forcing_term_trajectory_buffer)(trajectory_step_count,0)      = logged_dmp_variables_buffer.transform_sys_state_local.getTime();
        forcing_term_trajectory_buffer->block(trajectory_step_count,1,1,dmp_num_dimensions)   = logged_dmp_variables_buffer.forcing_term.transpose().block(0,0,1,dmp_num_dimensions);

        (*transform_sys_ct_acc_trajectory_buffer)(trajectory_step_count,0)   = logged_dmp_variables_buffer.transform_sys_state_local.getTime();
        transform_sys_ct_acc_trajectory_buffer->block(trajectory_step_count,1,1,dmp_num_dimensions)  = logged_dmp_variables_buffer.transform_sys_ct_acc.transpose().block(0,0,1,dmp_num_dimensions);

        trajectory_step_count++;

        // post-conditions checking
        if (rt_assert(trajectory_step_count <= MAX_TRAJ_SIZE) == false)
        {
            return false;
        }

        return true;
    }

    /**
     * NON-REAL-TIME!!!\n
     * If you want to maintain real-time performance,
     * do/call this function from a separate (non-real-time) thread!!!
     *
     * @param data_directory Specified directory to save the data files
     * @return Success or failure
     */
    bool DMPDataIO::saveTrajectoryDataSet(const char* data_directory)
    {
        // pre-conditions checking
        if (rt_assert(DMPDataIO::isValid()) == false)
        {
            return false;
        }

        /*if (rt_assert(DataIO::saveTrajectoryData(data_directory, "tau_trajectory.txt",
                                                 trajectory_step_count, 2, *save_data_buffer,
                                                 *tau_trajectory_buffer)) == false)
        {
            return false;
        }*/
        if (rt_assert(DataIO::saveTrajectoryData(data_directory, "transform_sys_state_local_trajectory.txt",
                                                 trajectory_step_count, (1+(3*dmp_num_dimensions)),
                                                 *save_data_buffer,
                                                 *transform_sys_state_local_trajectory_buffer)) == false)
        {
            return false;
        }
        if (rt_assert(DataIO::saveTrajectoryData(data_directory, "transform_sys_state_global_trajectory.txt",
                                                 trajectory_step_count, (1+(3*dmp_num_dimensions)),
                                                 *save_data_buffer,
                                                 *transform_sys_state_global_trajectory_buffer)) == false)
        {
            return false;
        }
        if (rt_assert(DataIO::saveTrajectoryData(data_directory, "goal_state_local_trajectory.txt",
                                                 trajectory_step_count, (1+(3*dmp_num_dimensions)),
                                                 *save_data_buffer,
                                                 *goal_state_local_trajectory_buffer)) == false)
        {
            return false;
        }
        if (rt_assert(DataIO::saveTrajectoryData(data_directory, "goal_state_global_trajectory.txt",
                                                 trajectory_step_count, (1+(3*dmp_num_dimensions)),
                                                 *save_data_buffer,
                                                 *goal_state_global_trajectory_buffer)) == false)
        {
            return false;
        }
        if (rt_assert(DataIO::saveTrajectoryData(data_directory, "steady_state_goal_position_local_trajectory.txt",
                                                 trajectory_step_count, (1+dmp_num_dimensions),
                                                 *save_data_buffer,
                                                 *steady_state_goal_position_local_trajectory_buffer)) == false)
        {
            return false;
        }
        if (rt_assert(DataIO::saveTrajectoryData(data_directory, "steady_state_goal_position_global_trajectory.txt",
                                                 trajectory_step_count, (1+dmp_num_dimensions),
                                                 *save_data_buffer,
                                                 *steady_state_goal_position_global_trajectory_buffer)) == false)
        {
            return false;
        }
        if (rt_assert(DataIO::saveTrajectoryData(data_directory, "basis_functions_trajectory.txt",
                                                 trajectory_step_count, (1+model_size),
                                                 *save_data_buffer,
                                                 *basis_functions_trajectory_buffer)) == false)
        {
            return false;
        }
        if (rt_assert(DataIO::saveTrajectoryData(data_directory, "forcing_term_trajectory.txt",
                                                 trajectory_step_count, (1+dmp_num_dimensions),
                                                 *save_data_buffer,
                                                 *forcing_term_trajectory_buffer)) == false)
        {
            return false;
        }
        if (rt_assert(DataIO::saveTrajectoryData(data_directory, "transform_sys_ct_acc_trajectory.txt",
                                                 trajectory_step_count, (1+dmp_num_dimensions),
                                                 *save_data_buffer,
                                                 *transform_sys_ct_acc_trajectory_buffer)) == false)
        {
            return false;
        }

        return true;
    }

    DMPDataIO::~DMPDataIO()
    {}

}
