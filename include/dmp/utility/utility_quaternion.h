/*
 * utility_quaternion.h
 * (Real-Time or not: please check the description of each individual function!)
 *
 *  Compilation of utility functions for Quaternions.
 *  The convention for the Quaternion Q is as follows:
 *  >> first component Q[0] is the real component of the Quaternion,
 *  >> the last three (Q[1], Q[2], Q[3]) are the complex components of the
 * Quaternion. For the sake of uniqueness, since Quaternion Q and Quaternion -Q
 * represent exactly the same pose, we always use the "standardized" version of
 * Quaternion Q, which has positive real component (Q[0] >= 0.0). The term
 * "standardization" process for Quaternion Q here refers to the process
 * ensuring Q has positive real component (Q[0] >= 0.0), i.e. by multiplying Q
 * with -1 if it has negative real component (Q[0] < 0.0).
 *
 *  Created on: Jan 2017
 *  Author: Giovanni Sutanto
 */

#ifndef UTILITY_QUATERNION_H
#define UTILITY_QUATERNION_H

#include "dmp/utility/DefinitionsBase.h"

namespace dmp {

/**
 * Normalize a quaternion Q, such that it has norm(Q)=1.0 (becomes a unit
 * quaternion).
 * @param Q Input and output quaternion Q
 * @return Success (true) or failure (false)
 */
bool normalizeQuaternion(Vector4& Q);

/**
 * The convention for the quaternion Q is as follows:
 *  >> first component Q[0] is the real component of the quaternion,
 *  >> the last three (Q[1], Q[2], Q[3]) are the complex components of the
 * quaternion. For the sake of uniqueness, since quaternion Q and quaternion -Q
 * represent exactly the same pose, we always use the "standardized" version of
 * quaternion Q, which has positive real component (Q[0] >= 0.0). The term
 * "standardization" process for quaternion Q here refers to the process of
 * ensuring Q has positive real component (Q[0] >= 0.0), i.e. by multiplying Q
 * with -1 if it has negative real component (Q[0] < 0.0). On the other hand,
 * "normalization" process for quaternion Q here refers to the process of
 *  ensuring norm(Q) = 1.0.
 *
 * @param Q Input and output quaternion Q
 * @return Success (true) or failure (false)
 */
bool standardizeNormalizeQuaternion(Vector4& Q);

/**
 * Written as computeQuatProduct() in the MATLAB version.
 * Essentially performs quaternion (complex number) multiplication, notationally
 * written as: Q_a o Q_b ( o is the quaternion composition operator). Please
 * note that this operation is NOT commutative; order of input quaternions
 * matters!
 *
 * @param Q_A Input quaternion 1
 * @param Q_B Input quaternion 2
 * @return Output quaternion
 */
Vector4 computeQuaternionComposition(Vector4 Q_a, Vector4 Q_b);

/**
 * Compute the conjugate of input quaternion Q (considering quaternion as a
 * hyper-complex number). The output is denoted as Q* .
 * @param Q Input quaternion
 * @return Output quaternion (conjugated input quaternion)
 */
Vector4 computeQuaternionConjugate(const Vector4& Q);

/**
 * Given a quaternion Q and omega (angular velocity) as inputs, compute Qd.
 * @param input_Q Input quaternion Q
 * @param input_omega Input angular velocity
 * @param output_Qd Output Qd
 * @return Success (true) or failure (false)
 */
bool computeQd(Vector4 input_Q, const Vector3& input_omega, Vector4& output_Qd);

/**
 * Given a quaternion Q, Qd, omega (angular velocity), and omegad (angular
 * acceleration) as inputs, compute Qdd.
 * @param input_Q Input quaternion Q
 * @param input_Qd Input Qd
 * @param input_omega Input angular velocity
 * @param input_omegad Input angular acceleration
 * @param output_Qdd Output Qdd
 * @return Success (true) or failure (false)
 */
bool computeQdd(Vector4 input_Q, const Vector4& input_Qd,
                const Vector3& input_omega, const Vector3& input_omegad,
                Vector4& output_Qdd);

/**
 * Perform exponential mapping so(3) => SO(3).
 * @param input_r Input so(3)
 * @param output_exp_r Output SO(3)
 * @return Success (true) or failure (false)
 */
bool computeExpMap_so3_to_SO3(const Vector3& input_r, Vector4& output_exp_r);

/**
 * Perform logarithm mapping SO(3) => so(3)
 * @param input_Q Input SO(3)
 * @param output_log_Q Output so(3)
 * @return Success (true) or failure (false)
 */
bool computeLogMap_SO3_to_so3(const Vector4& input_Q, Vector3& output_log_Q);

/**
 * Compute the orientation control error signal: 2 * log(in_Q1 o in_Q2*)
 * (please see the notation definitions in functions
 * computeQuaternionConjugate(), computeQuaternionComposition(), and
 * computeLogMap_SO3_to_so3()).
 * @param in_Q1 Input quaternion 1
 * @param in_Q2 Input quaternion 2
 * @param out_result Output orientation control error signal 2 * log(in_Q1 o
 * in_Q2*)
 * @return Success (true) or failure (false)
 */
bool computeLogQuaternionDifference(Vector4 in_Q1, Vector4 in_Q2,
                                    Vector3& out_result);

/**
 * Perform 1 time-step forward quaternion-preserving numerical integration.
 * @param in_Q_t Input quaternion at time t
 * @param in_omega_t Input angular velocity (omega) at time t
 * @param in_dt Input time step
 * @param out_Q_t_plus_1 Output quaternion at time (t+1), the result of the
 * integration
 * @return Success (true) or failure (false)
 */
bool integrateQuaternion(Vector4 in_Q_t, Vector3 in_omega_t, double in_dt,
                         Vector4& out_Q_t_plus_1);

}  // namespace dmp
#endif
