#include "dmp/dmp_discrete/DMPDataIODiscrete.h"

namespace dmp
{

DMPDataIODiscrete::DMPDataIODiscrete():
    DMPDataIO(), transform_sys_discrete(NULL), func_approx_discrete(NULL),
    canonical_sys_state_trajectory_buffer(new MatrixTx4())
{
    DMPDataIODiscrete::reset();
}

/**
 * @param transform_system_discrete Discrete Transformation System, whose data will be recorded
 * @param func_approximator_discrete Discrete Function Approximator, whose data will be recorded
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 */
DMPDataIODiscrete::DMPDataIODiscrete(TransformSystemDiscrete* transform_system_discrete,
                                     FuncApproximatorDiscrete* func_approximator_discrete,
                                     RealTimeAssertor* real_time_assertor):
    DMPDataIO(transform_system_discrete, func_approximator_discrete, real_time_assertor),
    transform_sys_discrete(transform_system_discrete),
    func_approx_discrete(func_approximator_discrete),
    canonical_sys_state_trajectory_buffer(new MatrixTx4())
{
    DMPDataIODiscrete::reset();
}

/**
 * Initialize/reset the data buffers and counters (usually done everytime before start of data saving).
 *
 * @return Success or failure
 */
bool DMPDataIODiscrete::reset()
{
    if (rt_assert(DMPDataIO::reset()) == false)
    {
        return false;
    }

    return (resizeAndReset(canonical_sys_state_trajectory_buffer, MAX_TRAJ_SIZE, 4));
}

/**
 * Checks whether this discrete-DMP data recorder is valid or not.
 *
 * @return discrete-DMP data recorder is valid (true) or discrete-DMP data recorder is invalid (false)
 */
bool DMPDataIODiscrete::isValid()
{
    if (rt_assert(DMPDataIO::isValid()) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(transform_sys_discrete != NULL)) &&
                  (rt_assert(func_approx_discrete != NULL))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(transform_sys_discrete->isValid())) &&
                  (rt_assert(func_approx_discrete->isValid()))) == false)
    {
        return false;
    }
    if (rt_assert(canonical_sys_state_trajectory_buffer != NULL) == false)
    {
        return false;
    }
    return true;
}

/**
 * Copy the supplied variables into current column of the buffers.
 *
 * @param logged_dmp_discrete_variables_buffer Discrete DMP variables that will be logged
 * @return Success or failure
 */
bool DMPDataIODiscrete::collectTrajectoryDataSet(const LoggedDMPDiscreteVariables& logged_dmp_discrete_variables_buffer)
{
    // pre-condition checking
    if (rt_assert(DMPDataIODiscrete::isValid()) == false)
    {
        return false;
    }
    // input checking
    if (rt_assert(logged_dmp_discrete_variables_buffer.isValid()) == false)
    {
        return false;
    }

    (*canonical_sys_state_trajectory_buffer)(trajectory_step_count,0)           = logged_dmp_discrete_variables_buffer.transform_sys_state_local.getTime();
    canonical_sys_state_trajectory_buffer->block(trajectory_step_count,1,1,3)   = logged_dmp_discrete_variables_buffer.canonical_sys_state.transpose();

    return (rt_assert(DMPDataIO::collectTrajectoryDataSet(logged_dmp_discrete_variables_buffer)));
}

/**
 * NON-REAL-TIME!!!\n
 * If you want to maintain real-time performance,
 * do/call this function from a separate (non-real-time) thread!!!
 *
 * @param data_directory Specified directory to save the data files
 * @return Success or failure
 */
bool DMPDataIODiscrete::saveTrajectoryDataSet(const char* data_directory)
{
    // pre-conditions checking
    if (rt_assert(DMPDataIODiscrete::isValid()) == false)
    {
        return false;
    }

    if (rt_assert(DataIO::saveTrajectoryData(data_directory, "canonical_sys_state_trajectory.txt",
                                             trajectory_step_count, 4, *save_data_buffer,
                                             *canonical_sys_state_trajectory_buffer)) == false)
    {
        return false;
    }

    return (rt_assert(DMPDataIO::saveTrajectoryDataSet(data_directory)));
}

/**
 * NEVER RUN IN REAL-TIME\n
 * Save weights' matrix into a file.
 *
 * @param dir_path Directory path containing the file
 * @param file_name The file name
 * @return Success or failure
 */
bool DMPDataIODiscrete::saveWeights(const char* dir_path, const char* file_name)
{
    // pre-conditions checking:
    if (rt_assert(DMPDataIODiscrete::isValid()) == false)
    {
        return false;
    }

    MatrixNxM                   weights(dmp_num_dimensions, model_size);

    if (rt_assert(func_approx_discrete->getWeights(weights)) == false)
    {
        return false;
    }

    return (rt_assert(DMPDataIO::writeMatrixToFile(dir_path, file_name, weights)));
}

/**
 * NEVER RUN IN REAL-TIME\n
 * Load weights' matrix from a file onto the function approximator.
 *
 * @param dir_path Directory path containing the file
 * @param file_name The file name
 * @return Success or failure
 */
bool DMPDataIODiscrete::loadWeights(const char* dir_path, const char* file_name)
{
    // pre-conditions checking:
    if (rt_assert(DMPDataIODiscrete::isValid()) == false)
    {
        return false;
    }

    MatrixNxM                   weights(dmp_num_dimensions, model_size);

    if (rt_assert(DMPDataIO::readMatrixFromFile(dir_path, file_name, weights)) == false)
    {
        return false;
    }

    return (rt_assert(func_approx_discrete->setWeights(weights)));
}

/**
 * NEVER RUN IN REAL-TIME\n
 * Save transform_sys_discrete's A_learn vector into a file.
 *
 * @param dir_path Directory path containing the file
 * @param file_name The file name
 * @return Success or failure
 */
bool DMPDataIODiscrete::saveALearn(const char* dir_path, const char* file_name)
{
    // pre-conditions checking:
    if (rt_assert(DMPDataIODiscrete::isValid()) == false)
    {
        return false;
    }

    VectorN                     A_learn(dmp_num_dimensions);

    A_learn                     = transform_sys_discrete->getLearningAmplitude();

    return (rt_assert(DMPDataIO::writeMatrixToFile(dir_path, file_name, A_learn)));
}

/**
 * NEVER RUN IN REAL-TIME\n
 * Load A_learn vector from a file onto the transform_sys_discrete.
 *
 * @param dir_path Directory path containing the file
 * @param file_name The file name
 * @return Success or failure
 */
bool DMPDataIODiscrete::loadALearn(const char* dir_path, const char* file_name)
{
    // pre-conditions checking:
    if (rt_assert(DMPDataIODiscrete::isValid()) == false)
    {
        return false;
    }

    VectorN                     A_learn(dmp_num_dimensions);

    if (rt_assert(DMPDataIO::readMatrixFromFile(dir_path, file_name, A_learn)) == false)
    {
        return false;
    }

    return (rt_assert(transform_sys_discrete->setLearningAmplitude(A_learn)));
}

/**
 * NEVER RUN IN REAL-TIME\n
 * Save basis functions' values over variation of canonical state position into a file.
 *
 * @param file_path
 * @return Success or failure
 */
bool DMPDataIODiscrete::saveBasisFunctions(const char* file_path)
{
    // pre-conditions checking:
    if (rt_assert(DMPDataIODiscrete::isValid()) == false)
    {
        return false;
    }

    VectorM                     result(model_size);
    double                      x                   = 0.0;
    double                      dx                  = 0.0001;

    FILE*                       f                   = fopen(file_path, "w");
    if (rt_assert(f != NULL) == false)
    {
        return false;
    }

	while (x <= 1.0)
	{
        fprintf(f, "%.05f,", x);
        if (rt_assert(func_approx_discrete->getBasisFunctionVector(x, result)) == false)
        {
            fclose(f);
            return false;
        }

		for (int i=0; i<model_size; ++i)
		{
            fprintf(f, "%.05f", result[i]);
			if (i != (model_size-1))
			{
				fprintf(f, ",");
			}
		}
		x	+= dx;
		fprintf(f, "\n");
	}

	fclose(f);
    return true;
}

/**
 * Returns the discrete transformation system pointer.
 */
TransformSystemDiscrete* DMPDataIODiscrete::getTransformSysDiscretePointer()
{
    return (transform_sys_discrete);
}

/**
 * Returns the discrete function approximator pointer.
 */
FuncApproximatorDiscrete* DMPDataIODiscrete::getFuncApproxDiscretePointer()
{
    return (func_approx_discrete);
}

DMPDataIODiscrete::~DMPDataIODiscrete()
{}

}
