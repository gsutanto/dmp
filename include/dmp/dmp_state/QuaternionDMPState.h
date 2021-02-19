/*
 * QuaternionDMPState.h
 *
 *  Data object that holds variables that can be needed for estimating
 *  the current Quaternion DMP state including Q (4D Quaternion "position"),
 *  Qd (4D Quaternion "velocity"), Qdd (4D Quaternion "acceleration"),
 *  omega (3D angular velocity), omegad (3D angular acceleration), and time.
 *  The convention for the Quaternion Q is as follows:
 *  >> first component Q[0] is the real component of the Quaternion,
 *  >> the last three (Q[1], Q[2], Q[3]) are the complex components of the Quaternion.
 *  For the sake of uniqueness, since Quaternion Q and Quaternion -Q represent
 *  exactly the same pose, we always use the "standardized" version of Quaternion Q, which has
 *  positive real component (Q[0] >= 0.0). The term "standardization" process for Quaternion Q here
 *  refers to the process ensuring Q has positive real component (Q[0] >= 0.0), i.e.
 *  by multiplying Q with -1 if it has negative real component (Q[0] < 0.0).
 *
 *  See references:
 *  [1] A. Ude, B. Nemec, T. Petric, and J. Morimoto, “Orientation in cartesian space
 *      dynamic movement primitives,” in IEEE International Conference on Robotics and Automation (ICRA),
 *      2014, pp. 2997–3004.
 *  [2] P. Pastor, L. Righetti, M. Kalakrishnan, and S. Schaal, “Online movement adaptation based
 *      on previous sensor experiences,” in IEEE International Conference on Intelligent Robots and
 *      Systems (IROS), 2011, pp. 365–371.
 *  [3] A.J. Ijspeert, J. Nakanishi, H. Hoffmann, P. Pastor, and S. Schaal;
 *      Dynamical movement primitives: Learning attractor models for motor behaviors;
 *      Neural Comput. 25, 2 (February 2013), 328-373
 *
 *  Created on: Jan 6, 2017
 *  Author: Giovanni Sutanto
 */

#ifndef QUATERNION_DMP_STATE_H
#define QUATERNION_DMP_STATE_H

#include "dmp/dmp_state/DMPState.h"
#include "dmp/utility/utility_quaternion.h"

namespace dmp
{

class QuaternionDMPState : public DMPState
{

protected:

    Vector3     omega;  // vector of length 3 (3D), representing angular velocity
    Vector3     omegad; // vector of length 3 (3D), representing angular acceleration

public:

    QuaternionDMPState();

    /**
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState(RealTimeAssertor* real_time_assertor);

    /**
     * @param Q_init Initialization value of 4-dimensional Quaternion "position" vector
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState(const Vector4& Q_init, RealTimeAssertor* real_time_assertor);

    /**
     * @param Q_init Initialization value of 4-dimensional Quaternion "position" vector
     * @param omega_init Initialization value of 3-dimensional angular velocity vector
     * @param omegad_init Initialization value of 3-dimensional angular acceleration vector
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState(const Vector4& Q_init,
                       const Vector3& omega_init,
                       const Vector3& omegad_init,
                       RealTimeAssertor* real_time_assertor);

    /**
     * @param Q_init Initialization value of 4-dimensional Quaternion "position" vector
     * @param Qd_init Initialization value of 4-dimensional Quaternion "velocity" vector
     * @param Qdd_init Initialization value of 4-dimensional Quaternion "acceleration" vector
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState(const Vector4& Q_init,
                       const Vector4& Qd_init,
                       const Vector4& Qdd_init,
                       RealTimeAssertor* real_time_assertor);

    /**
     * @param Q_init Initialization value of 4-dimensional Quaternion "position" vector
     * @param Qd_init Initialization value of 4-dimensional Quaternion "velocity" vector
     * @param Qdd_init Initialization value of 4-dimensional Quaternion "acceleration" vector
     * @param omega_init Initialization value of 3-dimensional angular velocity vector
     * @param omegad_init Initialization value of 3-dimensional angular acceleration vector
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState(const Vector4& Q_init,
                       const Vector4& Qd_init,
                       const Vector4& Qdd_init,
                       const Vector3& omega_init,
                       const Vector3& omegad_init,
                       RealTimeAssertor* real_time_assertor);

    /**
     * @param Q_init Initialization value of 4-dimensional Quaternion "position" vector
     * @param omega_init Initialization value of 3-dimensional angular velocity vector
     * @param omegad_init Initialization value of 3-dimensional angular acceleration vector
     * @param time_init Time instant where this state exists within the course of the trajectory
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState(const Vector4& Q_init,
                       const Vector3& omega_init,
                       const Vector3& omegad_init,
                       double time_init,
                       RealTimeAssertor* real_time_assertor);

    /**
     * @param Q_init Initialization value of 4-dimensional Quaternion "position" vector
     * @param Qd_init Initialization value of 4-dimensional Quaternion "velocity" vector
     * @param Qdd_init Initialization value of 4-dimensional Quaternion "acceleration" vector
     * @param time_init Time instant where this state exists within the course of the trajectory
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState(const Vector4& Q_init,
                       const Vector4& Qd_init,
                       const Vector4& Qdd_init,
                       double time_init,
                       RealTimeAssertor* real_time_assertor);

    /**
     * @param Q_init Initialization value of 4-dimensional Quaternion "position" vector
     * @param Qd_init Initialization value of 4-dimensional Quaternion "velocity" vector
     * @param Qdd_init Initialization value of 4-dimensional Quaternion "acceleration" vector
     * @param omega_init Initialization value of 3-dimensional angular velocity vector
     * @param omegad_init Initialization value of 3-dimensional angular acceleration vector
     * @param time_init Time instant where this state exists within the course of the trajectory
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    QuaternionDMPState(const Vector4& Q_init,
                       const Vector4& Qd_init,
                       const Vector4& Qdd_init,
                       const Vector3& omega_init,
                       const Vector3& omegad_init,
                       double time_init,
                       RealTimeAssertor* real_time_assertor);

    /**
     * @param other_quaterniondmpstate Another QuaternionDMPState whose value will be copied into this QuaternionDMPState
     */
    QuaternionDMPState(const QuaternionDMPState& other_quaterniondmpstate);

    /**
     * Checks whether the dimensions of Q, Qd, and Qdd are all equivalent to dmp_num_dimensions (4)
     * and time is greater than or equal to 0.0.
     *
     * @return QuaternionDMPState is valid (true) or QuaternionDMPState is invalid (false)
     */
    bool isValid() const;

    /**
     * Assignment operator ( operator= ) overloading.
     *
     * @param other_quaterniondmpstate Another QuaternionDMPState whose value will be copied into this QuaternionDMPState
     */
    QuaternionDMPState& operator=(const QuaternionDMPState& other_quaterniondmpstate);

    Vector4 getQ() const;
    Vector4 getQd() const;
    Vector4 getQdd() const;
    Vector3 getOmega() const;
    Vector3 getOmegad() const;

    bool setQ(const Vector4& new_Q);
    bool setQd(const Vector4& new_Qd);
    bool setQdd(const Vector4& new_Qdd);
    bool setOmega(const Vector3& new_omega);
    bool setOmegad(const Vector3& new_omegad);

    bool computeQdAndQdd();

    ~QuaternionDMPState();

};

}
#endif
