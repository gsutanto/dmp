/*
 * utility_cartesian.h
 * (Real-Time or not: please check the description of each individual function!)
 *
 *  Compilation of utility functions for Cartesians (consists of Cartesian Coordinates and Quaternions).
 *
 *  Created on: Feb 2017
 *  Author: Giovanni Sutanto
 */

#ifndef UTILITY_CARTESIAN_H
#define UTILITY_CARTESIAN_H

#include <Eigen/Geometry>

#include "dmp/utility/utility_quaternion.h"
#include "dmp/dmp_state/DMPState.h"
#include "dmp/dmp_state/QuaternionDMPState.h"

namespace dmp
{

    // REAL-TIME REQUIREMENTS
    /**
     * Convert a 3D vector into its representation as a skew-symmetric matrix,
     * for example for performing a cross-product.
     * @param V Input 3D vector
     * @param V_hat Output 3X3 skew-symmetric matrix
     * @return Success (true) or failure (false)
     */
    bool convertVector3ToSkewSymmetricMatrix(const Vector3& V,
                                             Matrix3x3& V_hat);

    // REAL-TIME REQUIREMENTS
    /**
     * Compute the homogeneous transformation matrix,
     * given translation vector t and rotation represented as a unit quaternion.
     * @param t Input translation vector
     * @param Q Input rotation, represented as a unit quaternion
     * @param T Output homogenenous transformation matrix
     * @return Success (true) or failure (false)
     */
    bool computeHomogeneousTransformMatrix(Vector3 t,
                                           Vector4 Q,
                                           Matrix4x4& T);

    // REAL-TIME REQUIREMENTS
    /**
     * Compute the inverse of a homogeneous transformation matrix T = [R, t; 0, 1],
     * i.e. T_inv = [R^T, -R^T * t; 0, 1].
     * @param T Input homogeneous transformation matrix
     * @param T_inv Output inverse homogeneous transformation matrix
     * @return Success (true) or failure (false)
     */
    bool computeInverseHomogeneousTransformMatrix(Matrix4x4 T,
                                                  Matrix4x4& T_inv);

    // REAL-TIME REQUIREMENTS
    /**
     * Convert the representation of a pose state (position and orientation) coordsys3 (e.g. robot end-effector)
     * -- currently represented w.r.t. coordsys2 (e.g. robot base) --,
     * into its representation w.r.t. coordsys1 (e.g. a Vicon-tracked object, fixed w.r.t. robot base).
     * Assumption: T_coordsys1_to_coordsys2 is constant/fixed in time.
     * @param T_coordsys1_to_coordsys2 Input relative pose of coordsys2 seen from/w.r.t. coordsys1 (assumed constant/fixed in time)
     * @param cart_coord_state_coordsys2_to_coordsys3 Input position state of coordsys3 w.r.t. coordsys2
     * @param quat_state_coordsys2_to_coordsys3 Input orientation state of coordsys3 w.r.t. coordsys2 (as quaternion)
     * @param cart_coord_state_coordsys1_to_coordsys3 Output position state of coordsys3 w.r.t. coordsys1
     * @param quat_state_coordsys1_to_coordsys3 Output orientation state of coordsys3 w.r.t. coordsys1 (as quaternion)
     * @return Success (true) or failure (false)
     */
    bool performStateCoordTransformation(Matrix4x4 T_coordsys1_to_coordsys2,
                                         DMPState cart_coord_state_coordsys2_to_coordsys3,
                                         QuaternionDMPState quat_state_coordsys2_to_coordsys3,
                                         DMPState& cart_coord_state_coordsys1_to_coordsys3,
                                         QuaternionDMPState& quat_state_coordsys1_to_coordsys3);

    // REAL-TIME REQUIREMENTS
    /**
     * Given:
     * >> The representation of a pose state (position and orientation) coordsys3
     * (e.g. robot end-effector) relatively represented w.r.t. coordsys2 (e.g. robot base);
     * >> T_coordsys1_to_coordsys2: Relative pose of coordsys2 (e.g. robot base)
     * w.r.t. coordsys1 (e.g. a Vicon-tracked object, constant/fixed in time w.r.t. robot base);
     * >> T_coordsys3_to_coordsys4: Relative pose of coordsys4 (e.g. tool)
     * w.r.t. coordsys3 (e.g. robot end-effector; tool is constant/fixed in time w.r.t. robot end-effector);
     * Compute the relative pose of coordsys4 (e.g. tool) w.r.t. coordsys1 (e.g. a Vicon-tracked object).
     * Assumption: T_coordsys1_to_coordsys2 and T_coordsys3_to_coordsys4 are constant/fixed in time.
     * @param T_coordsys1_to_coordsys2 Input relative pose of coordsys2 (e.g. robot base) \n
     * w.r.t. coordsys1 (e.g. a Vicon-tracked object, constant/fixed in time w.r.t. robot base)
     * @param cart_coord_state_coordsys2_to_coordsys3 Input position state of coordsys3 w.r.t. coordsys2
     * @param quat_state_coordsys2_to_coordsys3 Input orientation state of coordsys3 w.r.t. coordsys2 (as quaternion)
     * @param T_coordsys3_to_coordsys4 Input relative pose of coordsys4 (e.g. tool) \n
     * w.r.t. coordsys3 (e.g. robot end-effector; tool is constant/fixed in time w.r.t. robot end-effector)
     * @param cart_coord_state_coordsys1_to_coordsys4 Output position state of coordsys4 w.r.t. coordsys1
     * @param quat_state_coordsys1_to_coordsys4 Output orientation state of coordsys4 w.r.t. coordsys1 (as quaternion)
     * @return Success (true) or failure (false)
     */
    bool performStateCoordTransformation(Matrix4x4 T_coordsys1_to_coordsys2,
                                         DMPState cart_coord_state_coordsys2_to_coordsys3,
                                         QuaternionDMPState quat_state_coordsys2_to_coordsys3,
                                         Matrix4x4 T_coordsys3_to_coordsys4,
                                         DMPState& cart_coord_state_coordsys1_to_coordsys4,
                                         QuaternionDMPState& quat_state_coordsys1_to_coordsys4);

}
#endif