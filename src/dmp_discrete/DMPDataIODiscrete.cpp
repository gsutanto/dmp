#include "dmp/dmp_discrete/DMPDataIODiscrete.h"

namespace dmp {

DMPDataIODiscrete::DMPDataIODiscrete()
    : DMPDataIO(),
      transform_sys_discrete(NULL),
      canonical_sys_state_trajectory_buffer(new MatrixTx4()) {
  DMPDataIODiscrete::reset();
}

DMPDataIODiscrete::DMPDataIODiscrete(
    uint dmp_num_dimensions_init, uint model_size_init,
    TransformSystemDiscrete* transform_system_discrete,
    RealTimeAssertor* real_time_assertor)
    : DMPDataIO(dmp_num_dimensions_init, model_size_init,
                transform_system_discrete, real_time_assertor),
      transform_sys_discrete(transform_system_discrete),
      canonical_sys_state_trajectory_buffer(new MatrixTx4()) {
  DMPDataIODiscrete::reset();
}

bool DMPDataIODiscrete::reset() {
  if (rt_assert(DMPDataIO::reset()) == false) {
    return false;
  }

  return (
      resizeAndReset(canonical_sys_state_trajectory_buffer, MAX_TRAJ_SIZE, 4));
}

bool DMPDataIODiscrete::isValid() {
  if (rt_assert(DMPDataIO::isValid()) == false) {
    return false;
  }
  if (rt_assert(rt_assert(transform_sys_discrete != NULL)) == false) {
    return false;
  }
  if (rt_assert(rt_assert(transform_sys_discrete->isValid())) == false) {
    return false;
  }
  if (rt_assert(canonical_sys_state_trajectory_buffer != NULL) == false) {
    return false;
  }
  return true;
}

bool DMPDataIODiscrete::collectTrajectoryDataSet(
    const LoggedDMPDiscreteVariables& logged_dmp_discrete_variables_buffer) {
  // pre-condition checking
  if (rt_assert(DMPDataIODiscrete::isValid()) == false) {
    return false;
  }
  // input checking
  if (rt_assert(logged_dmp_discrete_variables_buffer.isValid()) == false) {
    return false;
  }

  (*canonical_sys_state_trajectory_buffer)(trajectory_step_count, 0) =
      logged_dmp_discrete_variables_buffer.transform_sys_state_local.getTime();
  canonical_sys_state_trajectory_buffer->block(trajectory_step_count, 1, 1, 3) =
      logged_dmp_discrete_variables_buffer.canonical_sys_state.transpose();

  return (rt_assert(DMPDataIO::collectTrajectoryDataSet(
      logged_dmp_discrete_variables_buffer)));
}

bool DMPDataIODiscrete::saveTrajectoryDataSet(const char* data_directory) {
  // pre-conditions checking
  if (rt_assert(DMPDataIODiscrete::isValid()) == false) {
    return false;
  }

  if (rt_assert(DataIO::saveTrajectoryData(
          data_directory, "canonical_sys_state_trajectory.txt",
          trajectory_step_count, 4, *save_data_buffer,
          *canonical_sys_state_trajectory_buffer)) == false) {
    return false;
  }

  return (rt_assert(DMPDataIO::saveTrajectoryDataSet(data_directory)));
}

bool DMPDataIODiscrete::saveWeights(const char* dir_path,
                                    const char* file_name) {
  // pre-conditions checking:
  if (rt_assert(DMPDataIODiscrete::isValid()) == false) {
    return false;
  }

  MatrixNxM weights(dmp_num_dimensions, model_size);

  if (rt_assert(
          transform_sys_discrete->getFuncApproxDiscretePointer()->getWeights(
              weights)) == false) {
    return false;
  }

  return (
      rt_assert(DMPDataIO::writeMatrixToFile(dir_path, file_name, weights)));
}

bool DMPDataIODiscrete::loadWeights(const char* dir_path,
                                    const char* file_name) {
  // pre-conditions checking:
  if (rt_assert(DMPDataIODiscrete::isValid()) == false) {
    return false;
  }

  MatrixNxM weights(dmp_num_dimensions, model_size);

  if (rt_assert(DMPDataIO::readMatrixFromFile(dir_path, file_name, weights)) ==
      false) {
    return false;
  }

  return (rt_assert(
      transform_sys_discrete->getFuncApproxDiscretePointer()->setWeights(
          weights)));
}

bool DMPDataIODiscrete::saveALearn(const char* dir_path,
                                   const char* file_name) {
  // pre-conditions checking:
  if (rt_assert(DMPDataIODiscrete::isValid()) == false) {
    return false;
  }

  VectorN A_learn(dmp_num_dimensions);

  A_learn = transform_sys_discrete->getLearningAmplitude();

  return (
      rt_assert(DMPDataIO::writeMatrixToFile(dir_path, file_name, A_learn)));
}

bool DMPDataIODiscrete::loadALearn(const char* dir_path,
                                   const char* file_name) {
  // pre-conditions checking:
  if (rt_assert(DMPDataIODiscrete::isValid()) == false) {
    return false;
  }

  VectorN A_learn(dmp_num_dimensions);

  if (rt_assert(DMPDataIO::readMatrixFromFile(dir_path, file_name, A_learn)) ==
      false) {
    return false;
  }

  return (rt_assert(transform_sys_discrete->setLearningAmplitude(A_learn)));
}

bool DMPDataIODiscrete::saveBasisFunctions(const char* file_path) {
  // pre-conditions checking:
  if (rt_assert(DMPDataIODiscrete::isValid()) == false) {
    return false;
  }

  VectorM result(model_size);
  double x = 0.0;
  double dx = 0.0001;

  FILE* f = fopen(file_path, "w");
  if (rt_assert(f != NULL) == false) {
    return false;
  }

  while (x <= 1.0) {
    fprintf(f, "%.05f,", x);
    if (rt_assert(transform_sys_discrete->getFuncApproxDiscretePointer()
                      ->getBasisFunctionVector(x, result)) == false) {
      fclose(f);
      return false;
    }

    for (int i = 0; i < model_size; ++i) {
      fprintf(f, "%.05f", result[i]);
      if (i != (model_size - 1)) {
        fprintf(f, ",");
      }
    }
    x += dx;
    fprintf(f, "\n");
  }

  fclose(f);
  return true;
}

TransformSystemDiscrete* DMPDataIODiscrete::getTransformSysDiscretePointer() {
  return (transform_sys_discrete);
}

std::shared_ptr<FuncApproximatorDiscrete>
DMPDataIODiscrete::getFuncApproxDiscretePointer() {
  return (transform_sys_discrete->getFuncApproxDiscretePointer());
}

DMPDataIODiscrete::~DMPDataIODiscrete() {}

}  // namespace dmp
