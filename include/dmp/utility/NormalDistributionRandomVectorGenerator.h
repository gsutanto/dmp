/*
 * NormalDistributionRandomVectorGenerator.h
 *
 *  Provides a class for generation of random vectors
 *  sampled from Normal (Gaussian) Distribution.
 *
 *  Created on: January 22, 2016
 *  Author: Giovanni Sutanto
 */

#ifndef NORMAL_DISTRIBUTION_RANDOM_VECTOR_GENERATOR_H
#define NORMAL_DISTRIBUTION_RANDOM_VECTOR_GENERATOR_H

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <Eigen/Eigenvalues>
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include "dmp/utility/utility.h"
#include "dmp/utility/RealTimeAssertor.h"


namespace dmp
{

class NormalDistributionRandomVectorGenerator
{

private:

    boost::mt19937                                              rand_num_generator;
    boost::normal_distribution<>                                gaussian_distribution;
    boost::variate_generator< boost::mt19937&,
                              boost::normal_distribution<> >    rand_variate_generator;

    RealTimeAssertor*                                           rt_assertor;

    /**
     * Sample a random number from a standardized normal/Gaussian distribution (mean=0, std=1).
     *
     * @param dummy_number Unused variable/number, will be replaced by the sampled random number
     * @return The sampled random number
     */
    double sampleStandardNormalDistribution(double dummy_number);

public:

    NormalDistributionRandomVectorGenerator();

    /**
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    NormalDistributionRandomVectorGenerator(RealTimeAssertor* real_time_assertor);

    /**
     * REAL-TIME??? (TO BE TESTED)\n
     * Sorry for the mess here, C++ compiler did NOT allow me to write
     * the following function definition in NormalDistributionRandomVectorGenerator.cpp file
     * (if I do, I have to instantiate it, which is annoying).
     * Draw (random_vectors.cols()) random vector samples
     * (each has the same dimension as the mean_vector) from
     * a multivariate normal distribution with a specified mean and covariance.
     *
     * @param mean_vector The mean vector of the multivariate normal distribution
     * @param covariance_matrix The covariance matrix of the multivariate normal distribution
     * @param random_vectors The sampled random vectors (each column being a single random vector) (return variable)
     * @return Success or failure
     */
    template <class T1, class T2, class T3>
    bool drawRandomVectorsFromNormalDistribution(const T1& mean_vector,
                                                 const T2& covariance_matrix,
                                                 T3& random_vectors)
    {
        // input checking:
        if (rt_assert(rt_assert(mean_vector.rows()       == covariance_matrix.cols()) &&
                      rt_assert(mean_vector.cols()       == 1) &&
                      rt_assert(covariance_matrix.rows() == covariance_matrix.cols()) &&
                      rt_assert(random_vectors.rows()    == mean_vector.rows()) &&
                      rt_assert(random_vectors.cols()    >= 1)) == false)
        {
            return false;
        }

        T2  RV_transformer(covariance_matrix.rows(), covariance_matrix.cols());

        // using Eigen's eigen solver to do eigenvalue-eigenvector decomposition of
        // the covariance matrix:
        Eigen::SelfAdjointEigenSolver<T2>   eigen_solver(covariance_matrix);
        RV_transformer  = eigen_solver.eigenvectors() *
                          eigen_solver.eigenvalues().cwiseSqrt().asDiagonal();

        // sample random vectors from standardized normal distribution (mean=0, std=1):
        T3  standard_normal_RV  = T3::Zero(random_vectors.rows(),
                                           random_vectors.cols()).unaryExpr(boost::bind(boost::mem_fn(&NormalDistributionRandomVectorGenerator::sampleStandardNormalDistribution),
                                                                                        this, _1));

        // print out the sampled standardized normal random vectors
        // (if needed for debugging, making sure there are no strange number "repetitions"):
        //std::cout << "standard_normal_RV = \n" << standard_normal_RV << std::endl;

        // perform linear transformation on the sampled standardized normal random vectors:
        random_vectors = mean_vector.replicate(1, random_vectors.cols()) +
                         (RV_transformer * standard_normal_RV);

        return true;
    }

    ~NormalDistributionRandomVectorGenerator();

};

}
#endif
