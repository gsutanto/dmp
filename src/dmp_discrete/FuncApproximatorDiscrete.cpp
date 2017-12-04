#include "amd_clmc_dmp/dmp_discrete/FuncApproximatorDiscrete.h"

namespace dmp
{

FuncApproximatorDiscrete::FuncApproximatorDiscrete():
    FunctionApproximator(), centers(new VectorM(model_size)),
    bandwidths(new VectorM(model_size)), psi(new VectorM(model_size)),
    sum_psi(0.0)
{}

/**
 * NEVER RUN IN REAL-TIME, ONLY RUN IN INIT ROUTINE
 *
 * @param dmp_num_dimensions_init Number of DMP dimensions (N) to use
 * @param num_basis_functions Number of basis functions (M) to use
 * @param canonical_system_discrete Discrete canonical system that drives this discrete function approximator
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 */
FuncApproximatorDiscrete::FuncApproximatorDiscrete(uint dmp_num_dimensions_init,
                                                   uint num_basis_functions,
                                                   CanonicalSystemDiscrete* canonical_system_discrete,
                                                   RealTimeAssertor* real_time_assertor):
    FunctionApproximator(dmp_num_dimensions_init, num_basis_functions,
                         canonical_system_discrete, real_time_assertor),
    centers(new VectorM(model_size)), bandwidths(new VectorM(model_size)),
    psi(new VectorM(model_size)), sum_psi(0.0)
{
    initBasisFunctions();
}

/**
 * NEVER RUN IN REAL-TIME, ONLY RUN IN INIT ROUTINE
 * Initializes centers and bandwidths of the Gaussian basis functions.
 */
void FuncApproximatorDiscrete::initBasisFunctions()
{
    uint    canonical_order = ((CanonicalSystemDiscrete*) canonical_sys)->getOrder();
    double  alpha_canonical = ((CanonicalSystemDiscrete*) canonical_sys)->getAlpha();
    double  tau_reference   = canonical_sys->getTauSystemPointer()->getTauReference();

    // centers are spanned evenly within <tau_reference> (normally 0.5 seconds) period of
    // the evolution of the canonical state position:
    double  dt              = ((1.0-0.0)/(model_size-1)) * tau_reference;

    // distribute Gaussian centers within 0.5 seconds period of
    // the decaying-exponential evolution of the canonical state position:
    for(uint m = 0; m < model_size; m++)
	{
        double  t           = m * dt;
        if (canonical_order == 2)
		{
            // analytical solution to differential equation:
            // tau^2 * ydd(t) = alpha * ((beta * (0.0 - y(t))) - (tau * yd(t)))
            // with initial conditions y(0) = 1.0 and yd(0) = 0.0
            // beta = alpha / 4.0 (for critical damping response on the 2nd order system)
            // tau = 1.0 and centers[m] = y(m)
            (*centers)[m]   = (1.0+((alpha_canonical/2.0)*t))*exp(-(alpha_canonical/2.0)*t);
		}
		else	// 1st order canonical system
		{
            // analytical solution to differential equation:
            // tau * yd(t) = -alpha * y(t)
            // with initial conditions y(0) = 1.0
            // tau = 1.0 and centers[m] = y(m)
            (*centers)[m]   = exp(-alpha_canonical * t);
		}
	}

	// Define bandwidths around computed centers
    for(uint i = 0; i < (model_size-1); i++)
	{
        //(*bandwidths)[i]    = 1.0 / (((*centers)[i+1] - (*centers)[i]) * ((*centers)[i+1] - (*centers)[i]));                // original chebotar's implementation
        (*bandwidths)[i]    = 1.0 / (0.55 * ((*centers)[i+1] - (*centers)[i]) * 0.55 * ((*centers)[i+1] - (*centers)[i]));  // gsutanto: following Stefan's implementation in dcp.m (better overlap between neighboring basis functions)
	}
    (*bandwidths)[model_size-1] = (*bandwidths)[model_size-2];
}

/**
 * Checks whether this discrete function approximator is valid or not.
 *
 * @return Discrete function approximator is valid (true) or discrete function approximator is invalid (false)
 */
bool FuncApproximatorDiscrete::isValid()
{
    if (rt_assert(FunctionApproximator::isValid()) == false)
    {
        return false;
    }
    if (rt_assert(((CanonicalSystemDiscrete*) canonical_sys)->isValid()) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(centers            != NULL)) &&
                  (rt_assert(bandwidths         != NULL)) &&
                  (rt_assert(psi                != NULL))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(centers->rows()    == model_size)) &&
                  (rt_assert(bandwidths->rows() == model_size)) &&
                  (rt_assert(psi->rows()        == model_size))) == false)
    {
        return false;
    }
    return true;
}

/**
 * Returns the value of the approximated function at current canonical position and \n
 * current canonical multiplier, for each DMP dimensions (as a vector).
 *
 * @param result_f Computed forcing term vector at current canonical position and \n
 *                 current canonical multiplier, each vector component for each DMP dimension (return variable)
 * @param basis_function_vector [optional] If also interested in the basis function vector value, put its pointer here (return variable)
 * @param normalized_basis_func_vector_mult_phase_multiplier [optional] If also interested in \n
 *              the <normalized basis function vector multiplied by the canonical phase multiplier (phase position or phase velocity)> value, \n
 *              put its pointer here (return variable)
 * @return Success or failure
 */
bool FuncApproximatorDiscrete::getForcingTerm(VectorN& result_f, VectorM* basis_function_vector,
                                              VectorM* normalized_basis_func_vector_mult_phase_multiplier)
{
    // pre-condition checking
    if (rt_assert(FuncApproximatorDiscrete::isValid()) == false)
    {
        return false;
    }
    // input checking
    if (rt_assert(result_f.rows() == dmp_num_dimensions) == false)
    {
        return false;
    }

    // Update basis functions vector (psi) and its sum (sum_psi)
    if (rt_assert(getBasisFunctionVector(canonical_sys->getCanonicalPosition(),
                                         (*psi))) == false)
    {
        return false;
    }
    if (rt_assert(sum_psi != 0.0) == false)
    {
        return false;
    }

    if (basis_function_vector != NULL)
    {
        if (rt_assert(basis_function_vector->rows() == model_size) == false)
        {
            return false;
        }
        *basis_function_vector  = *psi;
    }

    if (normalized_basis_func_vector_mult_phase_multiplier != NULL)
    {
        (*normalized_basis_func_vector_mult_phase_multiplier) =
                (*psi) * ((canonical_sys->getCanonicalMultiplier()) / sum_psi);
    }

    VectorN f(dmp_num_dimensions);
    f   = ((*weights) * (*psi) * ((canonical_sys->getCanonicalMultiplier()) / sum_psi));

    if (rt_assert(containsNaN(f) == false) == false)
    {
        return false;
    }

    result_f    = f;

    return true;
}

/**
 * Returns the vector variables required for learning the weights of the basis functions.
 *
 * @param current_psi Current psi vector associated with current phase (canonical system multiplier) variable (return variable)
 * @param current_sum_psi Sum of current_psi vector (return variable)
 * @param current_xi Current phase (canonical system multiplier) variable (return variable)
 * @return Success or failure
 */
bool FuncApproximatorDiscrete::getFuncApproxLearningComponents(VectorM& current_psi,
                                                               double&  current_sum_psi,
                                                               double&  current_xi)
{
    // pre-condition checking
    if (rt_assert(FuncApproximatorDiscrete::isValid()) == false)
    {
        return false;
    }
    // input checking
    if (rt_assert(current_psi.rows() == model_size) == false)
    {
        return false;
    }

    // Update basis functions vector (psi) and its sum (sum_psi)
    if (rt_assert(getBasisFunctionVector(canonical_sys->getCanonicalPosition(),
                                         (*psi))) == false)
    {
        return false;
    }
    if (rt_assert(sum_psi != 0.0) == false)
    {
        return false;
    }

    current_psi     = (*psi);
    current_sum_psi = sum_psi;
    current_xi      = (canonical_sys->getCanonicalMultiplier());

    return true;
}

/**
 * Updates (and returns) the (unnormalized) vector of basis function evaluations at the given position (x).
 *
 * @param x Position to be evaluated at (usually this is the canonical state position)
 * @param result_vector Vector that will be filled with the result (return variable).\n
 *        If NULL then nothing gets returned (only updating basis functions vector (psi) and its sum (sum_psi)).
 * @return Success or failure
 */
bool FuncApproximatorDiscrete::getBasisFunctionVector(double x,
                                                      VectorM& result_vector)
{
    // pre-condition checking
    if (rt_assert(FuncApproximatorDiscrete::isValid()) == false)
    {
        return false;
    }
    // input checking
    if (rt_assert(result_vector.rows() == model_size) == false)
    {
        return false;
    }

	// Compute evaluation of each Gaussian
    for(uint i = 0; i < model_size; ++i)
	{
        (*psi)[i]   = getBasisFunctionValue(x, (*centers)[i], (*bandwidths)[i]);
    }
//    sum_psi         = (psi->colwise().sum())(0,0);
    sum_psi         = (psi->colwise().sum())(0,0) + (model_size * 1.e-10);

    result_vector   = (*psi);

    return true;
}

/**
 * Returns the value of a specified basis function at the given position (x).
 *
 * @param x Position to be evaluated at (usually this is the canonical state position)
 * @param center Center of the basis function curve
 * @param bandwidth Bandwidth of the basis function curve
 * @return The value of a specified basis function at the given position (x)
 */
double FuncApproximatorDiscrete::getBasisFunctionValue(double x, double center, double bandwidth)
{
	double	temp	= (x - center);
    return (exp(-0.5 * temp * temp * bandwidth));
}

/**
 * Get the current weights of the basis functions.
 *
 * @param weights_buffer Weights vector buffer, to which the current weights will be copied onto (to be returned)
 * @return Success or failure
 */
bool FuncApproximatorDiscrete::getWeights(MatrixNxM& weights_buffer)
{
    // pre-condition checking
    if (rt_assert(FuncApproximatorDiscrete::isValid()) == false)
    {
        return false;
    }

    weights_buffer  = *weights;
    return true;
}

/**
 * Set the new weights of the basis functions.
 *
 * @param new_weights New weights to be set
 * @return Success or failure
 */
bool FuncApproximatorDiscrete::setWeights(const MatrixNxM& new_weights)
{
    // pre-condition checking
    if (rt_assert(FuncApproximatorDiscrete::isValid()) == false)
    {
        return false;
    }
    // input checking
    if (rt_assert((rt_assert(new_weights.rows() == dmp_num_dimensions)) &&
                  (rt_assert(new_weights.cols() == model_size))) == false)
    {
        return false;
    }

    *weights        = new_weights;
    return true;
}

FuncApproximatorDiscrete::~FuncApproximatorDiscrete()
{}

}
