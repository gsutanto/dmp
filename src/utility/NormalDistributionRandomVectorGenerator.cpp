#include "dmp/utility/NormalDistributionRandomVectorGenerator.h"

namespace dmp
{

NormalDistributionRandomVectorGenerator::NormalDistributionRandomVectorGenerator():
    gaussian_distribution(boost::normal_distribution<>(0.0, 1.0)),
    rand_variate_generator(boost::variate_generator< boost::mt19937&, boost::normal_distribution<> >(rand_num_generator,
                                                                                                     gaussian_distribution)),
    rt_assertor(NULL)
{
    rand_variate_generator.engine().seed(static_cast<unsigned int>(std::time(0)));
}

/**
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 */
NormalDistributionRandomVectorGenerator::NormalDistributionRandomVectorGenerator(RealTimeAssertor* real_time_assertor):
    gaussian_distribution(boost::normal_distribution<>(0.0, 1.0)),
    rand_variate_generator(boost::variate_generator< boost::mt19937&, boost::normal_distribution<> >(rand_num_generator,
                                                                                                     gaussian_distribution)),
    rt_assertor(real_time_assertor)
{
    rand_variate_generator.engine().seed(static_cast<unsigned int>(std::time(0)));
}

/**
 * Sample a random number from a standardized normal/Gaussian distribution (mean=0, std=1).
 *
 * @param dummy_number Unused variable/number, will be replaced by the sampled random number
 * @return The sampled random number
 */
double NormalDistributionRandomVectorGenerator::sampleStandardNormalDistribution(double dummy_number)
{
    return rand_variate_generator();
}

NormalDistributionRandomVectorGenerator::~NormalDistributionRandomVectorGenerator()
{}

}
