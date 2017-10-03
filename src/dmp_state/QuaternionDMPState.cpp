#include "amd_clmc_dmp/dmp_state/QuaternionDMPState.h"

namespace dmp
{
    QuaternionDMPState::QuaternionDMPState()
    {
        dmp_num_dimensions  = 3;
        x.resize(4);
        xd.resize(4);
        xdd.resize(4);

        x                   = ZeroVector4;
        x[0]                = 1.0;
        xd                  = ZeroVector4;
        xdd                 = ZeroVector4;
        omega               = ZeroVector3;
        omegad              = ZeroVector3;
        time                = 0.0;
        rt_assertor         = NULL;
    }

    /**
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState::QuaternionDMPState(RealTimeAssertor* real_time_assertor)
    {
        dmp_num_dimensions  = 3;
        x.resize(4);
        xd.resize(4);
        xdd.resize(4);

        x                   = ZeroVector4;
        x[0]                = 1.0;
        xd                  = ZeroVector4;
        xdd                 = ZeroVector4;
        omega               = ZeroVector3;
        omegad              = ZeroVector3;
        time                = 0.0;
        rt_assertor         = real_time_assertor;
    }

    /**
     * @param Q_init Initialization value of 4-dimensional Quaternion "position" vector
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState::QuaternionDMPState(const Vector4& Q_init, RealTimeAssertor* real_time_assertor)
    {
        dmp_num_dimensions  = 3;
        x.resize(4);
        xd.resize(4);
        xdd.resize(4);

        x                   = Q_init;
        xd                  = ZeroVector4;
        xdd                 = ZeroVector4;
        omega               = ZeroVector3;
        omegad              = ZeroVector3;
        time                = 0.0;
        rt_assertor         = real_time_assertor;
    }

    /**
     * @param Q_init Initialization value of 4-dimensional Quaternion "position" vector
     * @param omega_init Initialization value of 3-dimensional angular velocity vector
     * @param omegad_init Initialization value of 3-dimensional angular acceleration vector
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState::QuaternionDMPState(const Vector4& Q_init,
                                           const Vector3& omega_init,
                                           const Vector3& omegad_init,
                                           RealTimeAssertor* real_time_assertor)
    {
        dmp_num_dimensions  = 3;
        x.resize(4);
        xd.resize(4);
        xdd.resize(4);

        x                   = Q_init;
        xd                  = ZeroVector4;
        xdd                 = ZeroVector4;
        omega               = omega_init;
        omegad              = omegad_init;
        time                = 0.0;
        rt_assertor         = real_time_assertor;
    }

    /**
     * @param Q_init Initialization value of 4-dimensional Quaternion "position" vector
     * @param Qd_init Initialization value of 4-dimensional Quaternion "velocity" vector
     * @param Qdd_init Initialization value of 4-dimensional Quaternion "acceleration" vector
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState::QuaternionDMPState(const Vector4& Q_init,
                                           const Vector4& Qd_init,
                                           const Vector4& Qdd_init,
                                           RealTimeAssertor* real_time_assertor)
    {
        dmp_num_dimensions  = 3;
        x.resize(4);
        xd.resize(4);
        xdd.resize(4);

        x                   = Q_init;
        xd                  = Qd_init;
        xdd                 = Qdd_init;
        omega               = ZeroVector3;
        omegad              = ZeroVector3;
        time                = 0.0;
        rt_assertor         = real_time_assertor;
    }

    /**
     * @param Q_init Initialization value of 4-dimensional Quaternion "position" vector
     * @param Qd_init Initialization value of 4-dimensional Quaternion "velocity" vector
     * @param Qdd_init Initialization value of 4-dimensional Quaternion "acceleration" vector
     * @param omega_init Initialization value of 3-dimensional angular velocity vector
     * @param omegad_init Initialization value of 3-dimensional angular acceleration vector
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState::QuaternionDMPState(const Vector4& Q_init,
                                           const Vector4& Qd_init,
                                           const Vector4& Qdd_init,
                                           const Vector3& omega_init,
                                           const Vector3& omegad_init,
                                           RealTimeAssertor* real_time_assertor)
    {
        dmp_num_dimensions  = 3;
        x.resize(4);
        xd.resize(4);
        xdd.resize(4);

        x                   = Q_init;
        xd                  = Qd_init;
        xdd                 = Qdd_init;
        omega               = omega_init;
        omegad              = omegad_init;
        time                = 0.0;
        rt_assertor         = real_time_assertor;
    }

    /**
     * @param Q_init Initialization value of 4-dimensional Quaternion "position" vector
     * @param omega_init Initialization value of 3-dimensional angular velocity vector
     * @param omegad_init Initialization value of 3-dimensional angular acceleration vector
     * @param time_init Time instant where this state exists within the course of the trajectory
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState::QuaternionDMPState(const Vector4& Q_init,
                                           const Vector3& omega_init,
                                           const Vector3& omegad_init,
                                           double time_init,
                                           RealTimeAssertor* real_time_assertor)
    {
        dmp_num_dimensions  = 3;
        x.resize(4);
        xd.resize(4);
        xdd.resize(4);

        x                   = Q_init;
        xd                  = ZeroVector4;
        xdd                 = ZeroVector4;
        omega               = omega_init;
        omegad              = omegad_init;
        time                = time_init;
        rt_assertor         = real_time_assertor;
    }

    /**
     * @param Q_init Initialization value of 4-dimensional Quaternion "position" vector
     * @param Qd_init Initialization value of 4-dimensional Quaternion "velocity" vector
     * @param Qdd_init Initialization value of 4-dimensional Quaternion "acceleration" vector
     * @param time_init Time instant where this state exists within the course of the trajectory
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState::QuaternionDMPState(const Vector4& Q_init,
                                           const Vector4& Qd_init,
                                           const Vector4& Qdd_init,
                                           double time_init,
                                           RealTimeAssertor* real_time_assertor)
    {
        dmp_num_dimensions  = 3;
        x.resize(4);
        xd.resize(4);
        xdd.resize(4);

        x                   = Q_init;
        xd                  = Qd_init;
        xdd                 = Qdd_init;
        omega               = ZeroVector3;
        omegad              = ZeroVector3;
        time                = time_init;
        rt_assertor         = real_time_assertor;
    }

    /**
     * @param Q_init Initialization value of 4-dimensional Quaternion "position" vector
     * @param Qd_init Initialization value of 4-dimensional Quaternion "velocity" vector
     * @param Qdd_init Initialization value of 4-dimensional Quaternion "acceleration" vector
     * @param omega_init Initialization value of 3-dimensional angular velocity vector
     * @param omegad_init Initialization value of 3-dimensional angular acceleration vector
     * @param time_init Time instant where this state exists within the course of the trajectory
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState::QuaternionDMPState(const Vector4& Q_init,
                                           const Vector4& Qd_init,
                                           const Vector4& Qdd_init,
                                           const Vector3& omega_init,
                                           const Vector3& omegad_init,
                                           double time_init,
                                           RealTimeAssertor* real_time_assertor)
    {
        dmp_num_dimensions  = 3;
        x.resize(4);
        xd.resize(4);
        xdd.resize(4);

        x                   = Q_init;
        xd                  = Qd_init;
        xdd                 = Qdd_init;
        omega               = omega_init;
        omegad              = omegad_init;
        time                = time_init;
        rt_assertor         = real_time_assertor;
    }

    /**
     * @param other_quaterniondmpstate Another QuaternionDMPState whose value will be copied into this QuaternionDMPState
     */
    QuaternionDMPState::QuaternionDMPState(const QuaternionDMPState& other_quaterniondmpstate)
    {
        dmp_num_dimensions  = 3;
        x.resize(4);
        xd.resize(4);
        xdd.resize(4);

        x                   = other_quaterniondmpstate.x;
        xd                  = other_quaterniondmpstate.xd;
        xdd                 = other_quaterniondmpstate.xdd;
        omega               = other_quaterniondmpstate.omega;
        omegad              = other_quaterniondmpstate.omegad;
        time                = other_quaterniondmpstate.time;
        rt_assertor         = other_quaterniondmpstate.rt_assertor;
    }

    /**
     * Checks whether the dimensions of Q, Qd, and Qdd are all equivalent to dmp_num_dimensions (4)
     * and time is greater than or equal to 0.0.
     *
     * @return QuaternionDMPState is valid (true) or QuaternionDMPState is invalid (false)
     */
    bool QuaternionDMPState::isValid() const
    {
        if (rt_assert(dmp_num_dimensions == 3) == false)
        {
            return false;
        }
        if (rt_assert(x.rows() == 4) == false)
        {
            return false;
        }
        if (rt_assert(x.norm() > 0) == false)
        {
            return false;
        }
        if (rt_assert(xd.rows() == 4) == false)
        {
            return false;
        }
        if (rt_assert(xdd.rows() == 4) == false)
        {
            return false;
        }
        if (rt_assert(omega.rows() == dmp_num_dimensions) == false)
        {
            return false;
        }
        if (rt_assert(omegad.rows() == dmp_num_dimensions) == false)
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
     * @param other_quaterniondmpstate Another QuaternionDMPState whose value will be copied into this QuaternionDMPState
     */
    QuaternionDMPState& QuaternionDMPState::operator=(const QuaternionDMPState& other_quaterniondmpstate)
    {
        dmp_num_dimensions  = 3;
        x.resize(4);
        xd.resize(4);
        xdd.resize(4);

        x                   = other_quaterniondmpstate.x;
        xd                  = other_quaterniondmpstate.xd;
        xdd                 = other_quaterniondmpstate.xdd;
        omega               = other_quaterniondmpstate.omega;
        omegad              = other_quaterniondmpstate.omegad;
        time                = other_quaterniondmpstate.time;
        rt_assertor         = other_quaterniondmpstate.rt_assertor;
        return (*this);
    }

    Vector4 QuaternionDMPState::getQ() const
    {
        return x;
    }

    Vector4 QuaternionDMPState::getQd() const
    {
        return xd;
    }

    Vector4 QuaternionDMPState::getQdd() const
    {
        return xdd;
    }

    Vector3 QuaternionDMPState::getOmega() const
    {
        return omega;
    }

    Vector3 QuaternionDMPState::getOmegad() const
    {
        return omegad;
    }

    bool QuaternionDMPState::setQ(const Vector4& new_Q)
    {
        // pre-condition(s) checking
        if (rt_assert(QuaternionDMPState::isValid()) == false)
        {
            return false;
        }

        Vector4 Q   = new_Q;
        if (rt_assert(normalizeQuaternion(Q)) == false)
        {
            return false;
        }
        x           = Q;
        return true;
    }

    bool QuaternionDMPState::setQd(const Vector4& new_Qd)
    {
        // pre-condition(s) checking
        if (rt_assert(QuaternionDMPState::isValid()) == false)
        {
            return false;
        }

        xd          = new_Qd;
        return true;
    }

    bool QuaternionDMPState::setQdd(const Vector4& new_Qdd)
    {
        // pre-condition(s) checking
        if (rt_assert(QuaternionDMPState::isValid()) == false)
        {
            return false;
        }

        xdd         = new_Qdd;
        return true;
    }

    bool QuaternionDMPState::setOmega(const Vector3& new_omega)
    {
        // pre-condition(s) checking
        if (rt_assert(QuaternionDMPState::isValid()) == false)
        {
            return false;
        }

        if (rt_assert(new_omega.rows() == 3))
        {
            omega   = new_omega;
            return true;
        }
        else
        {
            return false;
        }
    }

    bool QuaternionDMPState::setOmegad(const Vector3& new_omegad)
    {
        // pre-condition(s) checking
        if (rt_assert(QuaternionDMPState::isValid()) == false)
        {
            return false;
        }

        if (rt_assert(new_omegad.rows() == 3))
        {
            omegad  = new_omegad;
            return true;
        }
        else
        {
            return false;
        }
    }

    bool QuaternionDMPState::computeQdAndQdd()
    {
        // pre-condition(s) checking
        if (rt_assert(QuaternionDMPState::isValid()) == false)
        {
            return false;
        }

        Vector4 Q   = x.block(0,0,4,1);
        if (rt_assert(normalizeQuaternion(Q)) == false)
        {
            return false;
        }
        x           = Q;

        Vector4 Qd  = ZeroVector4;
        if (rt_assert(computeQd(x, omega, Qd)) == false)
        {
            return false;
        }
        xd          = Qd;
        Vector4 Qdd = ZeroVector4;
        if (rt_assert(computeQdd(x, xd, omega, omegad, Qdd)) == false)
        {
            return false;
        }
        xdd         = Qdd;

        return true;
    }

    QuaternionDMPState::~QuaternionDMPState()
    {}

}