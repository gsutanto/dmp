#include "amd_clmc_dmp/dmp_base/LearningSystem.h"

namespace dmp
{

    LearningSystem::LearningSystem():
        transform_sys(NULL), data_logger(NULL), rt_assertor(NULL),
        dmp_num_dimensions(0), model_size(0),
        learned_weights(new MatrixNxM(dmp_num_dimensions, model_size))
    {
        strcpy(data_directory_path, "");
    }

    /**
     * NEVER RUN IN REAL-TIME, ONLY RUN IN INIT ROUTINE
     *
     * @param dmp_num_dimensions_init Number of trajectory dimensions employed in this DMP
     * @param model_size_init Size of the model (M: number of basis functions or others) used to represent the function
     * @param transformation_system Transformation system associated with this Learning System
     * @param data_logging_system DMPDataIO associated with this Learning System
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     * @param opt_data_directory_path [optional] Full directory path where to save/log the learning data
     */
    LearningSystem::LearningSystem(uint dmp_num_dimensions_init, uint model_size_init,
                                   TransformationSystem* transformation_system,
                                   DMPDataIO* data_logging_system,
                                   RealTimeAssertor* real_time_assertor,
                                   const char* opt_data_directory_path):
        dmp_num_dimensions(dmp_num_dimensions_init), model_size(model_size_init),
        transform_sys(transformation_system), data_logger(data_logging_system),
        rt_assertor(real_time_assertor),
        learned_weights(new MatrixNxM(MAX_DMP_NUM_DIMENSIONS,MAX_MODEL_SIZE))

    {
        learned_weights->resize(dmp_num_dimensions, model_size);
        strcpy(data_directory_path, opt_data_directory_path);
    }

    /**
     * Checks whether this learning system is valid or not.
     *
     * @return Learning system is valid (true) or learning system is invalid (false)
     */
    bool LearningSystem::isValid()
    {
        if (rt_assert(rt_assert(transform_sys   != NULL) &&
                      rt_assert(data_logger     != NULL) &&
                      rt_assert(learned_weights != NULL)) == false)
        {
            return false;
        }
        if (rt_assert(rt_assert(transform_sys->isValid()) &&
                      rt_assert(data_logger->isValid())) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(dmp_num_dimensions >  0)) &&
                      (rt_assert(dmp_num_dimensions <= MAX_DMP_NUM_DIMENSIONS))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(model_size > 0)) &&
                      (rt_assert(model_size <= MAX_MODEL_SIZE))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(learned_weights->rows() == dmp_num_dimensions)) &&
                      (rt_assert(learned_weights->cols() == model_size))) == false)
        {
            return false;
        }
        if (rt_assert(model_size == transform_sys->getFuncApproxPointer()->getModelSize()) == false)
        {
            return false;
        }
        return true;
    }

    /**
     * Returns the number of dimensions in the DMP.
     *
     * @return Number of dimensions in the DMP
     */
    uint LearningSystem::getDMPNumDimensions()
    {
        return dmp_num_dimensions;
    }

    /**
     * Returns model size used to represent the function.
     *
     * @return Model size used to represent the function
     */
    uint LearningSystem::getModelSize()
    {
        return model_size;
    }

    /**
     * Returns the transformation system pointer.
     */
    TransformationSystem* LearningSystem::getTransformationSystemPointer()
    {
        return transform_sys;
    }

    LearningSystem::~LearningSystem()
    {}

}
