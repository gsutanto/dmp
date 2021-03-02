#include "dmp/dmp_base/FunctionApproximator.h"

namespace dmp {

FunctionApproximator::FunctionApproximator()
    : dmp_num_dimensions(0),
      model_size(0),
      canonical_sys(NULL),
      weights(new MatrixNxM(MAX_DMP_NUM_DIMENSIONS, MAX_MODEL_SIZE)),
      rt_assertor(NULL) {}

FunctionApproximator::FunctionApproximator(uint dmp_num_dimensions_init,
                                           uint model_size_init,
                                           CanonicalSystem* canonical_system,
                                           RealTimeAssertor* real_time_assertor)
    : dmp_num_dimensions(dmp_num_dimensions_init),
      model_size(model_size_init),
      canonical_sys(canonical_system),
      weights(new MatrixNxM(dmp_num_dimensions, model_size)),
      rt_assertor(real_time_assertor) {}

bool FunctionApproximator::isValid() {
  if (rt_assert((rt_assert(dmp_num_dimensions > 0)) &&
                (rt_assert(dmp_num_dimensions <= MAX_DMP_NUM_DIMENSIONS))) ==
      false) {
    return false;
  }
  if (rt_assert((rt_assert(model_size > 0)) &&
                (rt_assert(model_size <= MAX_MODEL_SIZE))) == false) {
    return false;
  }
  if (rt_assert(rt_assert(canonical_sys != NULL) &&
                rt_assert(weights != NULL)) == false) {
    return false;
  }
  if (rt_assert(canonical_sys->isValid()) == false) {
    return false;
  }
  if (rt_assert((rt_assert(weights->rows() == dmp_num_dimensions)) &&
                (rt_assert(weights->cols() == model_size))) == false) {
    return false;
  }
  return true;
}

uint FunctionApproximator::getDMPNumDimensions() { return dmp_num_dimensions; }

uint FunctionApproximator::getModelSize() { return model_size; }

CanonicalSystem* FunctionApproximator::getCanonicalSystemPointer() {
  return canonical_sys;
}

FunctionApproximator::~FunctionApproximator() {}

}  // namespace dmp
