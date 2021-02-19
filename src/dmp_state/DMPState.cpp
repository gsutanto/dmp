#include "dmp/dmp_state/DMPState.h"

namespace dmp
{

DMPState::DMPState()
{
    x                   = ZeroVectorN(MAX_DMP_NUM_DIMENSIONS);
    xd                  = ZeroVectorN(MAX_DMP_NUM_DIMENSIONS);
    xdd                 = ZeroVectorN(MAX_DMP_NUM_DIMENSIONS);
    time                = 0.0;
    dmp_num_dimensions  = 0;
    x.resize(dmp_num_dimensions);
    xd.resize(dmp_num_dimensions);
    xdd.resize(dmp_num_dimensions);
    rt_assertor         = NULL;
}

/**
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 */
DMPState::DMPState(RealTimeAssertor* real_time_assertor)
{
    x                   = ZeroVectorN(MAX_DMP_NUM_DIMENSIONS);
    xd                  = ZeroVectorN(MAX_DMP_NUM_DIMENSIONS);
    xdd                 = ZeroVectorN(MAX_DMP_NUM_DIMENSIONS);
    time                = 0.0;
    dmp_num_dimensions  = 0;
    x.resize(dmp_num_dimensions);
    xd.resize(dmp_num_dimensions);
    xdd.resize(dmp_num_dimensions);
    rt_assertor         = real_time_assertor;
}

/**
 * @param dmp_num_dimensions_init Initialization of the number of dimensions that will be used in the DMP
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 */
DMPState::DMPState(uint dmp_num_dimensions_init, RealTimeAssertor* real_time_assertor)
{
    dmp_num_dimensions  = dmp_num_dimensions_init;
    x.resize(dmp_num_dimensions);
    xd.resize(dmp_num_dimensions);
    xdd.resize(dmp_num_dimensions);

    x                   = ZeroVectorN(dmp_num_dimensions);
    xd                  = ZeroVectorN(dmp_num_dimensions);
    xdd                 = ZeroVectorN(dmp_num_dimensions);
    time                = 0.0;
    rt_assertor         = real_time_assertor;
}

/**
 * @param x_init Initialization value of N-dimensional position vector
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 */
DMPState::DMPState(const VectorN& x_init, RealTimeAssertor* real_time_assertor)
{
    dmp_num_dimensions  = x_init.rows();
    x.resize(dmp_num_dimensions);
    xd.resize(dmp_num_dimensions);
    xdd.resize(dmp_num_dimensions);

    x                   = x_init;
    xd                  = ZeroVectorN(dmp_num_dimensions);
    xdd                 = ZeroVectorN(dmp_num_dimensions);
    time                = 0.0;
    rt_assertor         = real_time_assertor;
}

/**
 * @param x_init Initialization value of N-dimensional position vector
 * @param xd_init Initialization value of N-dimensional velocity vector
 * @param xdd_init Initialization value of N-dimensional acceleration vector
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 */
DMPState::DMPState(const VectorN& x_init,
                   const VectorN& xd_init,
                   const VectorN& xdd_init,
                   RealTimeAssertor* real_time_assertor)
{
    dmp_num_dimensions  = x_init.rows();
    x.resize(dmp_num_dimensions);
    xd.resize(dmp_num_dimensions);
    xdd.resize(dmp_num_dimensions);

    x                   = x_init;
    xd                  = xd_init;
    xdd                 = xdd_init;
    time                = 0.0;
    rt_assertor         = real_time_assertor;
}

/**
 * @param x_init Initialization value of N-dimensional position vector
 * @param xd_init Initialization value of N-dimensional velocity vector
 * @param xdd_init Initialization value of N-dimensional acceleration vector
 * @param time_init Time instant where this state exists within the course of the trajectory
 * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
 */
DMPState::DMPState(const VectorN& x_init,
                   const VectorN& xd_init,
                   const VectorN& xdd_init,
                   double time_init,
                   RealTimeAssertor* real_time_assertor)
{
    dmp_num_dimensions  = x_init.rows();
    x.resize(dmp_num_dimensions);
    xd.resize(dmp_num_dimensions);
    xdd.resize(dmp_num_dimensions);

    x                   = x_init;
    xd                  = xd_init;
    xdd                 = xdd_init;
    time                = time_init;
    rt_assertor         = real_time_assertor;
}

/**
 * @param other_dmpstate Another DMPState whose value will be copied into this DMPState
 */
DMPState::DMPState(const DMPState& other_dmpstate)
{
    dmp_num_dimensions  = other_dmpstate.dmp_num_dimensions;
    x.resize(dmp_num_dimensions);
    xd.resize(dmp_num_dimensions);
    xdd.resize(dmp_num_dimensions);

    x                   = other_dmpstate.x;
    xd                  = other_dmpstate.xd;
    xdd                 = other_dmpstate.xdd;
    time                = other_dmpstate.time;
    rt_assertor         = other_dmpstate.rt_assertor;
}

/**
 * Checks whether the dimensions of x, xd, and xdd are all equivalent to dmp_num_dimensions
 * and time is greater than or equal to 0.0.
 *
 * @return DMPState is valid (true) or DMPState is invalid (false)
 */
bool DMPState::isValid() const
{
    if (rt_assert(x.rows() == dmp_num_dimensions) == false)
    {
        return false;
    }
    if (rt_assert(xd.rows() == dmp_num_dimensions) == false)
    {
        return false;
    }
    if (rt_assert(xdd.rows() == dmp_num_dimensions) == false)
    {
        return false;
    }
    if (rt_assert(time >= 0.0) == false)
    {
        return false;
    }
    return true;
}

/**
 * Assignment operator ( operator= ) overloading.
 *
 * @param other_dmpstate Another DMPState whose value will be copied into this DMPState
 */
DMPState& DMPState::operator=(const DMPState& other_dmpstate)
{
    dmp_num_dimensions  = other_dmpstate.dmp_num_dimensions;
    x.resize(dmp_num_dimensions);
    xd.resize(dmp_num_dimensions);
    xdd.resize(dmp_num_dimensions);

    x                   = other_dmpstate.x;
    xd                  = other_dmpstate.xd;
    xdd                 = other_dmpstate.xdd;
    time                = other_dmpstate.time;
    rt_assertor         = other_dmpstate.rt_assertor;
    return (*this);
}

uint DMPState::getDMPNumDimensions() const
{
    return dmp_num_dimensions;
}

VectorN DMPState::getX() const
{
    return x;
}

VectorN DMPState::getXd() const
{
    return xd;
}

VectorN DMPState::getXdd() const
{
    return xdd;
}

double DMPState::getTime() const
{
    return time;
}

bool DMPState::setX(const VectorN& new_x)
{
    // pre-condition(s) checking
    if (rt_assert(DMPState::isValid()) == false)
    {
        return false;
    }

    if (rt_assert(dmp_num_dimensions == new_x.rows()))
    {
        x   = new_x;
        return true;
    }
    else
    {
        return false;
    }
}

bool DMPState::setXd(const VectorN& new_xd)
{
    // pre-condition(s) checking
    if (rt_assert(DMPState::isValid()) == false)
    {
        return false;
    }

    if (rt_assert(dmp_num_dimensions == new_xd.rows()))
    {
        xd  = new_xd;
        return true;
    }
    else
    {
        return false;
    }
}

bool DMPState::setXdd(const VectorN& new_xdd)
{
    // pre-condition(s) checking
    if (rt_assert(DMPState::isValid()) == false)
    {
        return false;
    }

    if (rt_assert(dmp_num_dimensions == new_xdd.rows()))
    {
        xdd = new_xdd;
        return true;
    }
    else
    {
        return false;
    }
}

bool DMPState::setTime(double new_time)
{
    // pre-condition(s) checking
    if (rt_assert(DMPState::isValid()) == false)
    {
        return false;
    }

    if (rt_assert(new_time >= 0.0))
    {
        time    = new_time;
        return true;
    }
    else
    {
        return false;
    }
}

DMPState::~DMPState()
{}

}
