/*
 * CartesianCoordTransformer.h
 *
 *  Class definition to deal with Cartesian (3D) coordinate transformations and representations.
 *
 *  Created on: July 3, 2015
 *  Author: Giovanni Sutanto
 */

#ifndef CARTESIAN_COORD_TRANSFORMER_H
#define CARTESIAN_COORD_TRANSFORMER_H

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <vector>
#include <Eigen/Dense>
#include "amd_clmc_dmp/utility/RealTimeAssertor.h"
#include "amd_clmc_dmp/utility/DefinitionsDerived.h"
#include "amd_clmc_dmp/dmp_state/DMPState.h"

#define PARALLEL_VECTOR_PROJECTION_THRESHOLD        0.9 // two vectors will be considered parallel, if the projection of one vector on the other is greater than this value
#define MIN_CTRAJ_LOCAL_COORD_OPTION_NO             0
#define MAX_CTRAJ_LOCAL_COORD_OPTION_NO             3
#define N_GSUTANTO_LOCAL_COORD_FRAME_BASIS_VECTORS  5

enum CartCoordDMPLocalCoordinateFrameMethod // definitions are in CartesianCoordDMP.h: CartesianCoordDMP(...) (constructor)
{
    _NO_LOCAL_COORD_FRAME_          = 0,
    _GSUTANTO_LOCAL_COORD_FRAME_    = 1,
    _SCHAAL_LOCAL_COORD_FRAME_      = 2,
    _KROEMER_LOCAL_COORD_FRAME_     = 3
};

namespace dmp
{

class CartesianCoordTransformer
{

private:

    uint                cart_trajectory_size;       // keep track of the Cartesian trajectory size

    // The following field(s) are written as SMART pointers to reduce size overhead
    // in the stack, by allocating them in the heap:
    // (must be pre-allocated before real-time operations, at CartesianCoordTransformer's
    // initialization)
    VectorTPtr          cctraj_time_vector;
    Matrix4xTPtr        cctraj_old_coord_H_matrix;  // Homogeneous coordinates (on the global coordinate system) of every points on the trajectory, stacked as a matrix of 4-rows
    Matrix3xTPtr        cctrajd_old_coord_matrix;
    Matrix3xTPtr        cctrajdd_old_coord_matrix;
    Matrix4xTPtr        cctraj_new_coord_H_matrix;  // Homogeneous coordinates (on the local coordinate system) of every points on the trajectory, stacked as a matrix of 4-rows
    Matrix3xTPtr        cctrajd_new_coord_matrix;
    Matrix3xTPtr        cctrajdd_new_coord_matrix;

    // Special fields for G. Sutanto's trajectory local coordinate system formulation (_GSUTANTO_LOCAL_COORD_FRAME_):
    VectorMPtr          theta_kernel_centers;
    VectorMPtr          theta_kernel_Ds;
    VectorMPtr          basis_vector_weights_unnormalized;
    Matrix3xMPtr        basis_matrix;

protected:

    // The following field(s) are written as RAW pointers because we just want to point
    // to other data structure(s) that might outlive the instance of this class:
    RealTimeAssertor*   rt_assertor;

public:

    CartesianCoordTransformer();

    /**
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    CartesianCoordTransformer(RealTimeAssertor* real_time_assertor);

    /**
     * Initialize/reset the data buffers.
     *
     * @return Success or failure
     */
    bool reset();

    /**
     * Checks whether this Cartesian Coordinate Transformer is valid or not.
     *
     * @return Cartesian Coordinate Transformer is valid (true) or Cartesian Coordinate Transformer is invalid (false)
     */
    bool isValid();

    /**
     * Computes homogeneous transformation matrices between the trajectory's local and global coordinate systems.
     *
     * @param cart_trajectory Cartesian trajectory representation in the global coordinate system
     * @param rel_homogen_transform_matrix_local_to_global Relative homogeneous transformation matrix from the trajectory's local to global coordinate system (or local as seen from global coordinate system) (to be computed)
     * @param rel_homogen_transform_matrix_global_to_local Relative homogeneous transformation matrix from the trajectory's global to local coordinate system (or global as seen from local coordinate system) (to be computed)
     * @param ctraj_local_coordinate_frame_selection Trajectory local coordinate frame that will be used as \n
     *                                               the reference frame. Possible values can be seen in \n
     *                                               the definition of CartesianCoordDMP in CartesianCoordDMP.h
     * @return Success or failure
     */
    bool computeCTrajCoordTransforms(const Trajectory& cart_trajectory,
                                     Matrix4x4& rel_homogen_transform_matrix_local_to_global,
                                     Matrix4x4& rel_homogen_transform_matrix_global_to_local,
                                     uint ctraj_local_coordinate_frame_selection);

    /**
     * Convert the Cartesian trajectory from its vectors of DMPState representation into its matrices representation.
     *
     * @param cart_trajectory Vector of DMPState of trajectory (position, velocity, and acceleration)
     * @param ctraj_H_matrix Matrix of homogeneous coordinates of the Cartesian position
     * @param ctrajd_matrix Matrix of the Cartesian velocity
     * @param ctrajdd_matrix Matrix of the Cartesian acceleration
     * @param ctraj_time_vector Vector of time line of the Cartesian trajectory
     * @return Success or failure
     */
    bool convertCTrajDMPStatesToMatrices(const Trajectory& cart_trajectory,
                                         Matrix4xT& ctraj_H_matrix,
                                         Matrix3xT& ctrajd_matrix,
                                         Matrix3xT& ctrajdd_matrix,
                                         VectorT& ctraj_time_vector);

    /**
     * Computes the trajectory's representation on the new coordinate system.
     *
     * @param ctraj_H_matrix_old Matrix of homogeneous coordinates of the Cartesian position on the old coordinate system
     * @param ctrajd_matrix_old Matrix of the Cartesian velocity on the old coordinate system
     * @param ctrajdd_matrix_old Matrix of the Cartesian acceleration on the old coordinate system
     * @param rel_homogen_transform_matrix_old_to_new Relative homogeneous transformation matrix from the old to the new coordinate system
     * @param ctraj_H_matrix_new Matrix of homogeneous coordinates of the Cartesian position on the new coordinate system
     * @param ctrajd_matrix_new Matrix of the Cartesian velocity on the new coordinate system
     * @param ctrajdd_matrix_new Matrix of the Cartesian acceleration on the new coordinate system
     * @return Success or failure
     */
    bool computeCTrajAtNewCoordSys(const Matrix4xT& ctraj_H_matrix_old,
                                   const Matrix3xT& ctrajd_matrix_old,
                                   const Matrix3xT& ctrajdd_matrix_old,
                                   const Matrix4x4& rel_homogen_transform_matrix_old_to_new,
                                   Matrix4xT& ctraj_H_matrix_new,
                                   Matrix3xT& ctrajd_matrix_new,
                                   Matrix3xT& ctrajdd_matrix_new);

    /**
     * Computes a Cartesian state and/or vector representation on the new coordinate system.
     *
     * @param cart_H_old Homogeneous coordinates of the Cartesian position on the old coordinate system
     * @param cartd_old Cartesian velocity on the old coordinate system
     * @param cartdd_old Cartesian acceleration on the old coordinate system
     * @param cart_vector_old Any arbitrary (3D) Cartesian vector on the old coordinate system
     * @param rel_homogen_transform_matrix_old_to_new Relative homogeneous transformation matrix from the old to the new coordinate system
     * @param cart_H_new Homogeneous coordinates of the Cartesian position on the new coordinate system
     * @param cartd_new Cartesian velocity on the new coordinate system
     * @param cartdd_new Cartesian acceleration on the new coordinate system
     * @param cart_vector_new Any arbitrary (3D) Cartesian vector on the new coordinate system
     * @return Success or failure
     */
    bool computeCTrajAtNewCoordSys(Vector4* cart_H_old,
                                   Vector3* cartd_old,
                                   Vector3* cartdd_old,
                                   Vector3* cart_vector_old,
                                   const Matrix4x4& rel_homogen_transform_matrix_old_to_new,
                                   Vector4* cart_H_new,
                                   Vector3* cartd_new,
                                   Vector3* cartdd_new,
                                   Vector3* cart_vector_new);

    /**
     * Computes a Cartesian position (x-y-z) representation on the new coordinate system.
     */
    bool computeCPosAtNewCoordSys(const Vector3& pos_3D_old,
                                  const Matrix4x4& rel_homogen_transform_matrix_old_to_new,
                                  Vector3& pos_3D_new);

    /**
     * Convert the Cartesian trajectory from its matrices representation into its vectors of DMPState representation.
     *
     * @param ctraj_H_matrix Matrix of homogeneous coordinates of the Cartesian position
     * @param ctrajd_matrix Matrix of the Cartesian velocity
     * @param ctrajdd_matrix Matrix of the Cartesian acceleration
     * @param ctraj_time_vector Vector of time line of the Cartesian trajectory
     * @param cart_trajectory Vector of DMPState of trajectory (position, velocity, and acceleration)
     * @return Success or failure
     */
    bool convertCTrajMatricesToDMPStates(const Matrix4xT& ctraj_H_matrix,
                                         const Matrix3xT& ctrajd_matrix,
                                         const Matrix3xT& ctrajdd_matrix,
                                         const VectorT& ctraj_time_vector,
                                         Trajectory& cart_trajectory);

    /**
     * Convert the Cartesian trajectory from old coordinate system to new coordinate system (both are in vectors of DMPState representation).
     *
     * @param cart_trajectory_old_vector Vector of DMPState of trajectory (position, velocity, and acceleration) of the old coordinate system
     * @param rel_homogen_transform_matrix_old_to_new Relative homogeneous transformation matrix from the old to the new coordinate system
     * @param cart_trajectory_new_vector Vector of DMPState of trajectory (position, velocity, and acceleration) of the new coordinate system
     * @return
     */
    bool convertCTrajAtOldToNewCoordSys(const Trajectory& cart_trajectory_old_vector,
                                        const Matrix4x4& rel_homogen_transform_matrix_old_to_new,
                                        Trajectory& cart_trajectory_new_vector);

    /**
     * Convert a DMPState from old coordinate system to new coordinate system.
     *
     * @param dmpstate_old DMPState (position, velocity, and acceleration) of the old coordinate system
     * @param rel_homogen_transform_matrix_old_to_new Relative homogeneous transformation matrix from the old to the new coordinate system
     * @param dmpstate_new DMPState (position, velocity, and acceleration) of the new coordinate system
     * @return
     */
    bool convertCTrajAtOldToNewCoordSys(const DMPState& dmpstate_old,
                                        const Matrix4x4& rel_homogen_transform_matrix_old_to_new,
                                        DMPState& dmpstate_new);

    ~CartesianCoordTransformer();

};

}
#endif
