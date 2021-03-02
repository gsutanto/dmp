#include "dmp/dmp_base/LearningSystem.h"

namespace dmp {

LearningSystem::LearningSystem()
    : transform_sys(NULL),
      data_logger(NULL),
      rt_assertor(NULL),
      dmp_num_dimensions(0),
      model_size(0),
      learned_weights(new MatrixNxM(dmp_num_dimensions, model_size)) {
  strcpy(data_directory_path, "");
}

LearningSystem::LearningSystem(uint dmp_num_dimensions_init,
                               uint model_size_init,
                               TransformationSystem* transformation_system,
                               DMPDataIO* data_logging_system,
                               RealTimeAssertor* real_time_assertor,
                               const char* opt_data_directory_path)
    : dmp_num_dimensions(dmp_num_dimensions_init),
      model_size(model_size_init),
      transform_sys(transformation_system),
      data_logger(data_logging_system),
      rt_assertor(real_time_assertor),
      learned_weights(new MatrixNxM(MAX_DMP_NUM_DIMENSIONS, MAX_MODEL_SIZE))

{
  learned_weights->resize(dmp_num_dimensions, model_size);
  strcpy(data_directory_path, opt_data_directory_path);
}

bool LearningSystem::isValid() {
  if (rt_assert(rt_assert(transform_sys != NULL) &&
                rt_assert(data_logger != NULL) &&
                rt_assert(learned_weights != NULL)) == false) {
    return false;
  }
  if (rt_assert(rt_assert(transform_sys->isValid()) &&
                rt_assert(data_logger->isValid())) == false) {
    return false;
  }
  if (rt_assert((rt_assert(dmp_num_dimensions > 0)) &&
                (rt_assert(dmp_num_dimensions <= MAX_DMP_NUM_DIMENSIONS))) ==
      false) {
    return false;
  }
  if (rt_assert((rt_assert(model_size > 0)) &&
                (rt_assert(model_size <= MAX_MODEL_SIZE))) == false) {
    return false;
  }
  if (rt_assert((rt_assert(learned_weights->rows() == dmp_num_dimensions)) &&
                (rt_assert(learned_weights->cols() == model_size))) == false) {
    return false;
  }
  if (rt_assert(model_size ==
                transform_sys->getFuncApproxPointer()->getModelSize()) ==
      false) {
    return false;
  }
  return true;
}

uint LearningSystem::getDMPNumDimensions() { return dmp_num_dimensions; }

uint LearningSystem::getModelSize() { return model_size; }

TransformationSystem* LearningSystem::getTransformationSystemPointer() {
  return transform_sys;
}

LearningSystem::~LearningSystem() {}

}  // namespace dmp
