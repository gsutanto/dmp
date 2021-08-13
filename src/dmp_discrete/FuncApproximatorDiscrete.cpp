#include "dmp/dmp_discrete/FuncApproximatorDiscrete.h"

namespace dmp {

FuncApproximatorDiscrete::FuncApproximatorDiscrete()
    : FunctionApproximator(),
      centers(new VectorM(model_size)),
      bandwidths(new VectorM(model_size)),
      psi(new VectorM(model_size)),
      sum_psi(0.0) {}

FuncApproximatorDiscrete::FuncApproximatorDiscrete(
    uint dmp_num_dimensions_init, uint num_basis_functions,
    CanonicalSystemDiscrete* canonical_system_discrete,
    RealTimeAssertor* real_time_assertor)
    : FunctionApproximator(dmp_num_dimensions_init, num_basis_functions,
                           canonical_system_discrete, real_time_assertor),
      centers(new VectorM(model_size)),
      bandwidths(new VectorM(model_size)),
      psi(new VectorM(model_size)),
      sum_psi(0.0),
      canonical_sys_discrete(canonical_system_discrete) {
  initBasisFunctions();
}

void FuncApproximatorDiscrete::initBasisFunctions() {
  uint canonical_order = getCanonicalSystemDiscreteOrder();
  double alpha_canonical = canonical_sys_discrete->getAlpha();
  double tau_reference =
      canonical_sys->getTauSystemPointer()->getTauReference();

  // centers are spanned evenly within <tau_reference> (normally 0.5 seconds)
  // period of the evolution of the canonical state position:
  double dt = ((1.0 - 0.0) / (model_size - 1)) * tau_reference;

  // distribute Gaussian centers within 0.5 seconds period of
  // the decaying-exponential evolution of the canonical state position:
  for (uint m = 0; m < model_size; m++) {
    double t = m * dt;
    if (canonical_order == 2) {
      // analytical solution to differential equation:
      // tau^2 * ydd(t) = alpha * ((beta * (0.0 - y(t))) - (tau * yd(t)))
      // with initial conditions y(0) = 1.0 and yd(0) = 0.0
      // beta = alpha / 4.0 (for critical damping response on the 2nd order
      // system) tau = 1.0 and centers[m] = y(m)
      (*centers)[m] = (1.0 + ((alpha_canonical / 2.0) * t)) *
                      exp(-(alpha_canonical / 2.0) * t);
    } else {  // 1st order canonical system
      // analytical solution to differential equation:
      // tau * yd(t) = -alpha * y(t)
      // with initial conditions y(0) = 1.0
      // tau = 1.0 and centers[m] = y(m)
      (*centers)[m] = exp(-alpha_canonical * t);
    }
  }

  // Define bandwidths around computed centers
  for (uint i = 0; i < (model_size - 1); i++) {
    // (*bandwidths)[i]    = 1.0 / (((*centers)[i+1] - (*centers)[i]) *
    // ((*centers)[i+1] - (*centers)[i]));                // original chebotar's
    //  implementation
    (*bandwidths)[i] =
        1.0 / (0.55 * ((*centers)[i + 1] - (*centers)[i]) * 0.55 *
               ((*centers)[i + 1] -
                (*centers)[i]));  // gsutanto: following Stefan's implementation
                                  // in dcp.m (better overlap between
                                  // neighboring basis functions)
  }
  (*bandwidths)[model_size - 1] = (*bandwidths)[model_size - 2];
}

bool FuncApproximatorDiscrete::isValid() {
  if (rt_assert(FunctionApproximator::isValid()) == false) {
    return false;
  }
  if (rt_assert(canonical_sys_discrete->isValid()) == false) {
    return false;
  }
  if (rt_assert((rt_assert(centers != nullptr)) &&
                (rt_assert(bandwidths != nullptr)) &&
                (rt_assert(psi != nullptr))) == false) {
    return false;
  }
  if (rt_assert((rt_assert(centers->rows() == model_size)) &&
                (rt_assert(bandwidths->rows() == model_size)) &&
                (rt_assert(psi->rows() == model_size))) == false) {
    return false;
  }
  return true;
}

bool FuncApproximatorDiscrete::getForcingTerm(
    VectorN& result_f, VectorM* basis_function_vector,
    VectorM* normalized_basis_func_vector_mult_phase_multiplier) {
  // pre-condition checking
  if (rt_assert(FuncApproximatorDiscrete::isValid()) == false) {
    return false;
  }
  // input checking
  if (rt_assert(result_f.rows() == dmp_num_dimensions) == false) {
    return false;
  }

  VectorM normalized_basis_func_vector_multiplied_phase_multiplier =
      ZeroVectorM(model_size);
  if (rt_assert(getNormalizedBasisFunctionVectorMultipliedPhaseMultiplier(
          normalized_basis_func_vector_multiplied_phase_multiplier)) == false) {
    return false;
  }

  if (basis_function_vector != nullptr) {
    if (rt_assert(basis_function_vector->rows() == model_size) == false) {
      return false;
    }
    *basis_function_vector = *psi;
  }

  if (normalized_basis_func_vector_mult_phase_multiplier != nullptr) {
    (*normalized_basis_func_vector_mult_phase_multiplier) =
        normalized_basis_func_vector_multiplied_phase_multiplier;
  }

  VectorN f = ZeroVectorN(dmp_num_dimensions);
  f = ((*weights) * normalized_basis_func_vector_multiplied_phase_multiplier);

  if (rt_assert(containsNaN(f) == false) == false) {
    return false;
  }

  result_f = f;

  return true;
}

bool FuncApproximatorDiscrete::getFuncApproxLearningComponents(
    VectorM& current_psi, double& current_sum_psi, double& current_xi) {
  // pre-condition checking
  if (rt_assert(FuncApproximatorDiscrete::isValid()) == false) {
    return false;
  }
  // input checking
  if (rt_assert(current_psi.rows() == model_size) == false) {
    return false;
  }

  // Update basis functions vector (psi) and its sum (sum_psi)
  if (rt_assert(getBasisFunctionVector(canonical_sys->getCanonicalPosition(),
                                       (*psi))) == false) {
    return false;
  }
  if (rt_assert(sum_psi != 0.0) == false) {
    return false;
  }

  current_psi = (*psi);
  current_sum_psi = sum_psi;
  current_xi = (canonical_sys->getCanonicalMultiplier());

  return true;
}

bool FuncApproximatorDiscrete::
    getNormalizedBasisFunctionVectorMultipliedPhaseMultiplier(
        VectorM& normalized_basis_func_vector_mult_phase_multiplier) {
  // pre-condition checking
  if (rt_assert(FuncApproximatorDiscrete::isValid()) == false) {
    return false;
  }
  // input checking
  if (rt_assert(normalized_basis_func_vector_mult_phase_multiplier.rows() ==
                model_size) == false) {
    return false;
  }

  // Update basis functions vector (psi) and its sum (sum_psi)
  if (rt_assert(getBasisFunctionVector(canonical_sys->getCanonicalPosition(),
                                       (*psi))) == false) {
    return false;
  }
  if (rt_assert(sum_psi != 0.0) == false) {
    return false;
  }

  normalized_basis_func_vector_mult_phase_multiplier =
      (*psi) * ((canonical_sys->getCanonicalMultiplier()) / sum_psi);

  return true;
}

bool FuncApproximatorDiscrete::getBasisFunctionVector(double x,
                                                      VectorM& result_vector) {
  // pre-condition checking
  if (rt_assert(FuncApproximatorDiscrete::isValid()) == false) {
    return false;
  }
  // input checking
  if (rt_assert(result_vector.rows() == model_size) == false) {
    return false;
  }

  // Compute evaluation of each Gaussian
  for (uint i = 0; i < model_size; ++i) {
    (*psi)[i] = getBasisFunctionValue(x, (*centers)[i], (*bandwidths)[i]);
  }
  //    sum_psi         = (psi->colwise().sum())(0,0);
  sum_psi = (psi->colwise().sum())(0, 0) + (model_size * 1.e-10);

  result_vector = (*psi);

  return true;
}

double FuncApproximatorDiscrete::getBasisFunctionValue(double x, double center,
                                                       double bandwidth) {
  double temp = (x - center);
  return (exp(-0.5 * temp * temp * bandwidth));
}

bool FuncApproximatorDiscrete::getWeights(MatrixNxM& weights_buffer) {
  // pre-condition checking
  if (rt_assert(FuncApproximatorDiscrete::isValid()) == false) {
    return false;
  }

  weights_buffer = *weights;
  return true;
}

bool FuncApproximatorDiscrete::setWeights(const MatrixNxM& new_weights) {
  // pre-condition checking
  if (rt_assert(FuncApproximatorDiscrete::isValid()) == false) {
    return false;
  }
  // input checking
  if (rt_assert((rt_assert(new_weights.rows() == dmp_num_dimensions)) &&
                (rt_assert(new_weights.cols() == model_size))) == false) {
    return false;
  }

  *weights = new_weights;
  return true;
}

uint FuncApproximatorDiscrete::getCanonicalSystemDiscreteOrder() {
  return canonical_sys_discrete->getOrder();
}

double FuncApproximatorDiscrete::getCanonicalSystemDiscretePosition() {
  return canonical_sys_discrete->getX();
}

double FuncApproximatorDiscrete::getCanonicalSystemDiscreteVelocity() {
  return canonical_sys_discrete->getXd();
}

double FuncApproximatorDiscrete::getCanonicalSystemDiscreteAcceleration() {
  return canonical_sys_discrete->getXdd();
}

FuncApproximatorDiscrete::~FuncApproximatorDiscrete() {}

}  // namespace dmp
