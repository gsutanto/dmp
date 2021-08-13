#include "dmp/dmp_discrete/LearningSystemDiscrete.h"

#include <memory>

namespace dmp {

LearningSystemDiscrete::LearningSystemDiscrete()
    : LearningSystem(0, 0, nullptr, nullptr, nullptr, ""),
      learning_method(_SCHAAL_LWR_METHOD_),
      min_coord(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
      max_coord(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)) {
  min_coord.resize(dmp_num_dimensions);
  max_coord.resize(dmp_num_dimensions);
}

LearningSystemDiscrete::LearningSystemDiscrete(
    uint dmp_num_dimensions_init, uint model_size_init,
    TransformSystemDiscrete* transformation_system_discrete,
    DMPDataIODiscrete* data_logging_system, uint selected_learning_method,
    RealTimeAssertor* real_time_assertor, const char* opt_data_directory_path)
    : LearningSystem(dmp_num_dimensions_init, model_size_init,
                     transformation_system_discrete, data_logging_system,
                     real_time_assertor, opt_data_directory_path),
      learning_method(selected_learning_method),
      transform_sys_discrete(transformation_system_discrete),
      min_coord(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)),
      max_coord(ZeroVectorN(MAX_DMP_NUM_DIMENSIONS)) {
  min_coord.resize(dmp_num_dimensions);
  max_coord.resize(dmp_num_dimensions);
}

bool LearningSystemDiscrete::isValid() {
  if (rt_assert(LearningSystem::isValid()) == false) {
    return false;
  }
  if (rt_assert(transform_sys_discrete->isValid()) == false) {
    return false;
  }
  if (rt_assert((rt_assert(learning_method >= _SCHAAL_LWR_METHOD_)) &&
                (rt_assert(learning_method <=
                           _JPETERS_GLOBAL_LEAST_SQUARES_METHOD_))) == false) {
    return false;
  }
  if (rt_assert((rt_assert(min_coord.rows() == dmp_num_dimensions)) &&
                (rt_assert(max_coord.rows() == dmp_num_dimensions))) == false) {
    return false;
  }
  return true;
}

bool LearningSystemDiscrete::learnApproximator(
    const TrajectorySet& trajectory_set, double robot_task_servo_rate,
    double& mean_tau) {
  bool is_real_time = false;

  // pre-conditions checking
  if (rt_assert(LearningSystemDiscrete::isValid()) == false) {
    return false;
  }
  // input checking
  if (rt_assert((robot_task_servo_rate > 0.0) &&
                (robot_task_servo_rate <= MAX_TASK_SERVO_RATE)) == false) {
    return false;
  }

  *learned_weights = ZeroMatrixNxM(dmp_num_dimensions, model_size);

  uint N_traj =
      trajectory_set.size();  // total number of trajectories in the set
  if (rt_assert(N_traj > 0) == false) {
    return false;
  }
  DoubleVectorPtr tau;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, tau, N_traj)) ==
      false) {
    return false;
  }

  mean_tau = 0.0;
  double dt;

  uint N_traj_states = 0;
  for (uint i = 0; i < N_traj; ++i) {
    uint traj_size = trajectory_set[i].size();
    if (rt_assert((rt_assert(traj_size > 0)) &&
                  (rt_assert(traj_size <= MAX_TRAJ_SIZE))) == false) {
      return false;
    }
    N_traj_states += traj_size;

    // if each trajectory is containing the "correct" information of the
    // trajectory timing, then use this timing information to compute tau:
    (*tau)[i] = trajectory_set[i][traj_size - 1].getTime() -
                trajectory_set[i][0].getTime();
    // otherwise use the robot_task_servo_rate to compute tau
    // (assuming that the trajectory is sampled at robot_task_servo_rate):
    if ((*tau)[i] < MIN_TAU) {
      (*tau)[i] = (1.0 * (traj_size - 1)) / robot_task_servo_rate;
    }

    mean_tau += (*tau)[i];
  }
  mean_tau = mean_tau / (N_traj * 1.0);

  /*
   * The following is WHY this function is NON-REAL-TIME: the usage of Eigen's
   * MatrixXd data structures, which are dynamic data structures allocated in
   * the heap by using smart pointers (to preserve stack space), with NO
   * real-time guarantees. May be called by non-real-time threads.
   */
  // Matrix of target/desired forcing function values along/over the
  // demonstrated trajectories (matrix of size
  // dmp_num_dimensionsXN_traj_states):
  MatrixXxXPtr f_target;
  if (rt_assert(allocateMemoryIfNonRealTime(
          is_real_time, f_target, dmp_num_dimensions, N_traj_states)) ==
      false) {
    return false;
  }

  // Matrix of fitted forcing function values along/over the demonstrated
  // trajectories (matrix of size dmp_num_dimensionsXN_traj_states,
  //  used to check regression/learning quality):
  MatrixXxXPtr f_fit;
  if (rt_assert(allocateMemoryIfNonRealTime(
          is_real_time, f_fit, dmp_num_dimensions, N_traj_states)) == false) {
    return false;
  }

  // Matrix of (unnormalized and unmodulated) basis function vector values
  // along/over the demonstrated trajectories (also called the psi vector,
  // each column of this matrix being such vector for a particular state in the
  // trajectory) (matrix of size model_sizeXN_traj_states):
  MatrixXxXPtr Psi;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Psi, model_size,
                                            N_traj_states)) == false) {
    return false;
  }

  // Vector of the phase (canonical system multiplier) variable values
  // along/over the demonstrated trajectories
  // (vector of size 1XN_traj_states (row vector)):
  MatrixXxXPtr Xi;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Xi, 1,
                                            N_traj_states)) == false) {
    return false;
  }

  // Matrix of phase-modulated normalized basis function vector values
  // along/over the demonstrated trajectories
  // (each column of this matrix being such vector for a particular state in the
  // trajectory) (normalized by the sum of all basis function values in the
  // vector
  //  and modulated/multiplied by the phase (canonical system multiplier)
  //  variable value, for each column in this matrix)
  // (matrix of size model_sizeXN_traj_states):
  MatrixXxXPtr Phi;
  if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Phi, model_size,
                                            N_traj_states)) == false) {
    return false;
  }

  // Vector of normalized basis function values, modulated by a canonical
  // variable, on a particular state/point in the trajectory (vector of size
  // model_size):
  VectorN current_f_target(dmp_num_dimensions);
  current_f_target = ZeroVectorN(dmp_num_dimensions);
  VectorM current_psi(model_size);
  current_psi = ZeroVectorM(model_size);
  double current_sum_psi = 0.0;
  double current_xi = 0.0;

  VectorN A_learn_average(dmp_num_dimensions);
  A_learn_average = ZeroVectorN(dmp_num_dimensions);

  FuncApproximatorDiscrete* func_approx_discr =
      transform_sys_discrete->getFuncApproxDiscretePointer();

  uint idx_traj_state = 0;

  for (uint i = 0; i < N_traj; i++) {
    uint traj_size = trajectory_set[i].size();

    // set new value of tau (tau of to-be-learned trajectory):
    if (rt_assert(transform_sys->getTauSystemPointer()->setTauBase(
            (*tau)[i])) == false) {
      return false;
    }

    // (re-)start the canonical system:
    if (rt_assert(transform_sys->getCanonicalSystemPointer()->start()) ==
        false)  // Reset canonical system
    {
      return false;
    }

    DMPState start_state_demo = trajectory_set[i][0];
    DMPState goal_steady_state_demo = trajectory_set[i][traj_size - 1];

    // (re-)start the transformation system:
    if (rt_assert(transform_sys->start(start_state_demo,
                                       goal_steady_state_demo)) == false) {
      return false;
    }

    // For Schaal's DMP Formulation:
    // A_learn        = G_learn - y_0_learn
    A_learn_average +=
        (goal_steady_state_demo.getX() - start_state_demo.getX());

    max_coord = trajectory_set[i][0].getX();
    min_coord = trajectory_set[i][0].getX();

    // Iterate over DMP state sequence
    for (uint j = 0; j < traj_size; j++) {
      DMPState current_state_demo = trajectory_set[i][j];

      VectorN X = current_state_demo.getX();

      for (uint n = 0; n < dmp_num_dimensions; ++n) {
        if (min_coord[n] > X[n]) {
          min_coord[n] = X[n];
        }

        if (max_coord[n] < X[n]) {
          max_coord[n] = X[n];
        }
      }

      if (j == 0) {
        dt = trajectory_set[i][1].getTime() - trajectory_set[i][0].getTime();
      } else {
        // Compute time difference between old state and new state
        dt =
            trajectory_set[i][j].getTime() - trajectory_set[i][j - 1].getTime();
      }
      if (dt == 0.0) {
        dt = 1.0 / robot_task_servo_rate;
      }

      // Compute target force value for this state
      if (rt_assert(transform_sys->getTargetForcingTerm(
              current_state_demo, current_f_target)) == false) {
        return false;
      }
      // f_target->col(idx_traj_state)   = current_f_target;
      //  adapted from Stefan Schaal's MATLAB implementation:
      f_target->col(idx_traj_state) = current_f_target;

      // Compute basis function contributions at the current canonical state
      if (rt_assert(func_approx_discr->getFuncApproxLearningComponents(
              current_psi, current_sum_psi, current_xi)) == false) {
        return false;
      }

      Psi->col(idx_traj_state) = current_psi;
      (*Xi)(0, idx_traj_state) = current_xi;

      // Normalize and multiply with canonical multiplier variable
      Phi->col(idx_traj_state) = current_psi * (current_xi / current_sum_psi);

      if (rt_assert(
              transform_sys->getCanonicalSystemPointer()->updateCanonicalState(
                  dt)) == false) {
        return false;
      }
      if (rt_assert(transform_sys->updateCurrentGoalState(dt)) == false) {
        return false;
      }
      idx_traj_state++;
    }
  }

  // For Schaal's DMP Formulation (average of):
  // A_learn        = G_learn - y_0_learn
  A_learn_average = A_learn_average / N_traj;
  if (rt_assert(((TransformSystemDiscrete*)transform_sys)
                    ->setLearningAmplitude(A_learn_average)) == false) {
    return false;
  }

  if (learning_method == _SCHAAL_LWR_METHOD_) {
    MatrixXxXPtr Xi_squared;
    if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, Xi_squared, 1,
                                              N_traj_states)) == false) {
      return false;
    }

    VectorMPtr sx2;
    if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, sx2, model_size,
                                              1)) == false) {
      return false;
    }

    VectorMPtr sxtd;
    if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, sxtd, model_size,
                                              1)) == false) {
      return false;
    }

    MatrixXxXPtr Xi_times_f_target_1D;
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, Xi_times_f_target_1D, 1, N_traj_states)) == false) {
      return false;
    }

    *Xi_squared = Xi->cwiseProduct(*Xi);
    *sx2 = ((Xi_squared->replicate(model_size, 1)).cwiseProduct(*Psi))
               .rowwise()
               .sum();
    // adapted from Stefan Schaal's MATLAB implementation:
    *sx2 = *sx2 + (1.e-10 * OnesVectorM(model_size));

    for (uint d = 0; d < dmp_num_dimensions; d++) {
      *Xi_times_f_target_1D =
          Xi->cwiseProduct(f_target->block(d, 0, 1, N_traj_states));
      *sxtd =
          ((Xi_times_f_target_1D->replicate(model_size, 1)).cwiseProduct(*Psi))
              .rowwise()
              .sum();

      learned_weights->block(d, 0, 1, model_size) =
          (sxtd->cwiseQuotient(*sx2)).transpose();
    }
  } else if (learning_method == _JPETERS_GLOBAL_LEAST_SQUARES_METHOD_) {
    // R Matrix: the regularization matrix (diagonal matrix of size
    // model_sizeXmodel_size):
    MatrixXxXPtr R;
    if (rt_assert(allocateMemoryIfNonRealTime(is_real_time, R, model_size,
                                              model_size)) == false) {
      return false;
    }

    // Matrix [(Phi * Phi_transpose) + R] (matrix of size
    // model_sizeXmodel_size):
    MatrixXxXPtr Phi_Phi_T_plus_R;
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, Phi_Phi_T_plus_R, model_size, model_size)) == false) {
      return false;
    }

    // Pseudo-Inverse of [(Phi * Phi_transpose) + R] (matrix of size
    // model_sizeXmodel_size):
    MatrixXxXPtr pinv_Phi_Phi_T_plus_R;
    if (rt_assert(allocateMemoryIfNonRealTime(
            is_real_time, pinv_Phi_Phi_T_plus_R, model_size, model_size)) ==
        false) {
      return false;
    }

    double regularization_const = 0.0;
    *R = regularization_const * EyeMatrixXxX(model_size, model_size);

    *Phi_Phi_T_plus_R = ((*Phi) * (Phi->transpose())) + (*R);

    // Compute least-squares solution of the regression with pseudo-inverse
    if (rt_assert(pseudoInverse((*Phi_Phi_T_plus_R),
                                (*pinv_Phi_Phi_T_plus_R))) == false) {
      return false;
    }
    *learned_weights =
        (*f_target) * (Phi->transpose()) * (*pinv_Phi_Phi_T_plus_R);
  }

  if (rt_assert(containsNaN(*learned_weights) == false) == false) {
    return false;
  }

  *f_fit = (*learned_weights) * (*Phi);

  if (strcmp(data_directory_path, "") != 0) {
    if (rt_assert(data_logger->writeMatrixToFile(
            data_directory_path, "f_target.txt", *f_target)) == false) {
      return false;
    }
    if (rt_assert(data_logger->writeMatrixToFile(
            data_directory_path, "f_fit.txt", *f_fit)) == false) {
      return false;
    }
    if (rt_assert(data_logger->writeMatrixToFile(
            data_directory_path, "learned_weights.txt", *learned_weights)) ==
        false) {
      return false;
    }
  }

  // Save weights in the function approximator
  if (rt_assert(func_approx_discr->setWeights(*learned_weights)) == false) {
    return false;
  }

  return true;
}

uint LearningSystemDiscrete::getLearningMethod() { return (learning_method); }

LearningSystemDiscrete::~LearningSystemDiscrete() {}

}  // namespace dmp
