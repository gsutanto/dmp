#include "dmp/cart_dmp/cart_coord_dmp/CartesianCoordTransformer.h"

namespace dmp {

CartesianCoordTransformer::CartesianCoordTransformer()
    : rt_assertor(NULL),
      theta_kernel_centers(new VectorM(MAX_MODEL_SIZE)),
      theta_kernel_Ds(new VectorM(MAX_MODEL_SIZE)),
      basis_vector_weights_unnormalized(new VectorM(MAX_MODEL_SIZE)),
      basis_matrix(new Matrix3xM(3, MAX_MODEL_SIZE)) {
  bool is_real_time = false;

  cart_trajectory_size = MAX_TRAJ_SIZE;

  allocateMemoryIfNonRealTime(is_real_time, cctraj_time_vector,
                              cart_trajectory_size, 1);
  allocateMemoryIfNonRealTime(is_real_time, cctraj_old_coord_H_matrix, 4,
                              cart_trajectory_size);
  allocateMemoryIfNonRealTime(is_real_time, cctrajd_old_coord_matrix, 3,
                              cart_trajectory_size);
  allocateMemoryIfNonRealTime(is_real_time, cctrajdd_old_coord_matrix, 3,
                              cart_trajectory_size);
  allocateMemoryIfNonRealTime(is_real_time, cctraj_new_coord_H_matrix, 4,
                              cart_trajectory_size);
  allocateMemoryIfNonRealTime(is_real_time, cctrajd_new_coord_matrix, 3,
                              cart_trajectory_size);
  allocateMemoryIfNonRealTime(is_real_time, cctrajdd_new_coord_matrix, 3,
                              cart_trajectory_size);
}

CartesianCoordTransformer::CartesianCoordTransformer(
    RealTimeAssertor* real_time_assertor)
    : rt_assertor(real_time_assertor),
      theta_kernel_centers(
          new VectorM(N_GSUTANTO_LOCAL_COORD_FRAME_BASIS_VECTORS)),
      theta_kernel_Ds(new VectorM(N_GSUTANTO_LOCAL_COORD_FRAME_BASIS_VECTORS)),
      basis_vector_weights_unnormalized(
          new VectorM(N_GSUTANTO_LOCAL_COORD_FRAME_BASIS_VECTORS)),
      basis_matrix(
          new Matrix3xM(3, N_GSUTANTO_LOCAL_COORD_FRAME_BASIS_VECTORS)) {
  bool is_real_time = false;

  cart_trajectory_size = MAX_TRAJ_SIZE;

  allocateMemoryIfNonRealTime(is_real_time, cctraj_time_vector,
                              cart_trajectory_size, 1);
  allocateMemoryIfNonRealTime(is_real_time, cctraj_old_coord_H_matrix, 4,
                              cart_trajectory_size);
  allocateMemoryIfNonRealTime(is_real_time, cctrajd_old_coord_matrix, 3,
                              cart_trajectory_size);
  allocateMemoryIfNonRealTime(is_real_time, cctrajdd_old_coord_matrix, 3,
                              cart_trajectory_size);
  allocateMemoryIfNonRealTime(is_real_time, cctraj_new_coord_H_matrix, 4,
                              cart_trajectory_size);
  allocateMemoryIfNonRealTime(is_real_time, cctrajd_new_coord_matrix, 3,
                              cart_trajectory_size);
  allocateMemoryIfNonRealTime(is_real_time, cctrajdd_new_coord_matrix, 3,
                              cart_trajectory_size);
}

bool CartesianCoordTransformer::reset() {
  cart_trajectory_size = MAX_TRAJ_SIZE;
  if (rt_assert((rt_assert(resizeAndReset(cctraj_time_vector,
                                          cart_trajectory_size, 1))) &&
                (rt_assert(resizeAndReset(cctraj_old_coord_H_matrix, 4,
                                          cart_trajectory_size))) &&
                (rt_assert(resizeAndReset(cctrajd_old_coord_matrix, 3,
                                          cart_trajectory_size))) &&
                (rt_assert(resizeAndReset(cctrajdd_old_coord_matrix, 3,
                                          cart_trajectory_size))) &&
                (rt_assert(resizeAndReset(cctraj_new_coord_H_matrix, 4,
                                          cart_trajectory_size))) &&
                (rt_assert(resizeAndReset(cctrajd_new_coord_matrix, 3,
                                          cart_trajectory_size))) &&
                (rt_assert(resizeAndReset(cctrajdd_new_coord_matrix, 3,
                                          cart_trajectory_size)))) == false) {
    return false;
  }

  return true;
}

bool CartesianCoordTransformer::isValid() {
  if (rt_assert((rt_assert(cart_trajectory_size > 0)) &&
                (rt_assert(cart_trajectory_size <= MAX_TRAJ_SIZE))) == false) {
    return false;
  }
  if (rt_assert(isMemoryAllocated(cctraj_time_vector, cart_trajectory_size,
                                  1)) == false) {
    return false;
  }
  if (rt_assert(isMemoryAllocated(cctraj_old_coord_H_matrix, 4,
                                  cart_trajectory_size)) == false) {
    return false;
  }
  if (rt_assert(isMemoryAllocated(cctrajd_old_coord_matrix, 3,
                                  cart_trajectory_size)) == false) {
    return false;
  }
  if (rt_assert(isMemoryAllocated(cctrajdd_old_coord_matrix, 3,
                                  cart_trajectory_size)) == false) {
    return false;
  }
  if (rt_assert(isMemoryAllocated(cctraj_new_coord_H_matrix, 4,
                                  cart_trajectory_size)) == false) {
    return false;
  }
  if (rt_assert(isMemoryAllocated(cctrajd_new_coord_matrix, 3,
                                  cart_trajectory_size)) == false) {
    return false;
  }
  if (rt_assert(isMemoryAllocated(cctrajdd_new_coord_matrix, 3,
                                  cart_trajectory_size)) == false) {
    return false;
  }
  if (rt_assert((rt_assert(theta_kernel_centers != NULL)) &&
                (rt_assert(theta_kernel_Ds != NULL)) &&
                (rt_assert(basis_vector_weights_unnormalized != NULL)) &&
                (rt_assert(basis_matrix != NULL))) == false) {
    return false;
  }
  if (rt_assert((rt_assert(theta_kernel_centers->rows() ==
                           N_GSUTANTO_LOCAL_COORD_FRAME_BASIS_VECTORS)) &&
                (rt_assert(theta_kernel_Ds->rows() ==
                           N_GSUTANTO_LOCAL_COORD_FRAME_BASIS_VECTORS)) &&
                (rt_assert(basis_vector_weights_unnormalized->rows() ==
                           N_GSUTANTO_LOCAL_COORD_FRAME_BASIS_VECTORS)) &&
                (rt_assert(basis_matrix->rows() == 3)) &&
                (rt_assert(basis_matrix->cols() ==
                           N_GSUTANTO_LOCAL_COORD_FRAME_BASIS_VECTORS))) ==
      false) {
    return false;
  }
  return true;
}

bool CartesianCoordTransformer::computeCTrajCoordTransforms(
    const Trajectory& cart_trajectory,
    Matrix4x4& rel_homogen_transform_matrix_local_to_global,
    Matrix4x4& rel_homogen_transform_matrix_global_to_local,
    uint ctraj_local_coordinate_frame_selection) {
  // pre-conditions checking
  if (rt_assert(CartesianCoordTransformer::isValid()) == false) {
    return false;
  }
  // input checking
  if (rt_assert((rt_assert(cart_trajectory.size() >= 2)) &&
                (rt_assert(cart_trajectory.size() <= MAX_TRAJ_SIZE))) ==
      false) {
    return false;
  }
  if (rt_assert(((ctraj_local_coordinate_frame_selection == 3) &&
                 (cart_trajectory.size() < 3)) == false) == false) {
    return false;
  }
  if (rt_assert(
          (rt_assert(cart_trajectory[0].isValid())) &&
          (rt_assert(cart_trajectory[cart_trajectory.size() - 1].isValid()))) ==
      false) {
    return false;
  }
  if (rt_assert((rt_assert(cart_trajectory[0].getDMPNumDimensions() == 3)) &&
                (rt_assert(cart_trajectory[cart_trajectory.size() - 1]
                               .getDMPNumDimensions() == 3))) == false) {
    return false;
  }
  if (rt_assert((rt_assert(ctraj_local_coordinate_frame_selection >=
                           MIN_CTRAJ_LOCAL_COORD_OPTION_NO)) &&
                (rt_assert(ctraj_local_coordinate_frame_selection <=
                           MAX_CTRAJ_LOCAL_COORD_OPTION_NO))) == false) {
    return false;
  }

  Vector3 start_cart_position = cart_trajectory[0].getX().block(0, 0, 3, 1);
  Vector3 goal_cart_position =
      cart_trajectory[cart_trajectory.size() - 1].getX().block(0, 0, 3, 1);
  if (rt_assert((rt_assert(start_cart_position.rows() == 3)) &&
                (rt_assert(goal_cart_position.rows() == 3))) == false) {
    return false;
  }

  Matrix3x3 rotation_matrix;
  Vector3 translation_vector = start_cart_position;
  Vector3 global_x_axis;
  Vector3 global_y_axis;
  Vector3 global_z_axis;
  Vector3 local_x_axis;
  Vector3 local_y_axis;
  Vector3 local_z_axis;
  Vector3 anchor_local_z_axis;

  global_x_axis << 1.0, 0.0, 0.0;
  global_y_axis << 0.0, 1.0, 0.0;
  global_z_axis << 0.0, 0.0, 1.0;

  if (ctraj_local_coordinate_frame_selection ==
      _NO_LOCAL_COORD_FRAME_) {  // no local coordinate transformation (or local
                                 // coord sys = global coord sys)
    local_x_axis = global_x_axis;
    local_y_axis = global_y_axis;
    local_z_axis = global_z_axis;
    translation_vector << 0.0, 0.0, 0.0;
  } else if (ctraj_local_coordinate_frame_selection ==
             _GSUTANTO_LOCAL_COORD_FRAME_) {  // G. Sutanto's trajectory local
                                              // coordinate system formulation
    local_x_axis = goal_cart_position - start_cart_position;
    local_x_axis =
        local_x_axis / local_x_axis.norm();  // normalize local x-axis

    // theta is the angle formed between global +z axis and local +x axis:
    double cos_theta = local_x_axis.dot(global_z_axis);
    double theta = fabs(acos(cos_theta));

    // compute the kernel centers and spread/standard deviation (D):
    *theta_kernel_centers = Eigen::VectorXd::LinSpaced(
        N_GSUTANTO_LOCAL_COORD_FRAME_BASIS_VECTORS, 0, M_PI);
    for (uint i = 0; i < N_GSUTANTO_LOCAL_COORD_FRAME_BASIS_VECTORS; i++) {
      if (i < (N_GSUTANTO_LOCAL_COORD_FRAME_BASIS_VECTORS - 1)) {
        (*theta_kernel_Ds)(i, 0) =
            (*theta_kernel_centers)(i + 1, 0) - (*theta_kernel_centers)(i, 0);
      } else if (i == (N_GSUTANTO_LOCAL_COORD_FRAME_BASIS_VECTORS - 1)) {
        (*theta_kernel_Ds)(i, 0) =
            (*theta_kernel_centers)(i, 0) - (*theta_kernel_centers)(i - 1, 0);
      }
    }
    *theta_kernel_Ds =
        (0.55 * theta_kernel_Ds->array()).array().square().cwiseInverse();

    // compute the kernel/basis vector weights:
    for (uint i = 0; i < N_GSUTANTO_LOCAL_COORD_FRAME_BASIS_VECTORS; i++) {
      (*basis_vector_weights_unnormalized)(i, 0) =
          exp(-0.5 * pow((theta - (*theta_kernel_centers)(i, 0)), 2) *
              (*theta_kernel_Ds)(i, 0));
    }

    // columns of basis_matrix below are the basis vectors:
    (*basis_matrix) = ZeroMatrix3xM(N_GSUTANTO_LOCAL_COORD_FRAME_BASIS_VECTORS);
    // theta == 0 (local_x_axis is perfectly aligned with global_z_axis)
    // corresponds to anchor_local_z_axis = global_y_axis:
    basis_matrix->block(0, 0, 3, 1) = global_y_axis;
    // theta == pi/4 corresponds to anchor_local_z_axis = global_z_axis X
    // local_x_axis:
    basis_matrix->block(0, 1, 3, 1) = global_z_axis.cross(local_x_axis);
    // theta == pi/2 (local_x_axis is perpendicular from global_z_axis)
    // corresponds to anchor_local_z_axis = global_z_axis:
    basis_matrix->block(0, 2, 3, 1) = global_z_axis;
    // theta == 3*pi/4 corresponds to anchor_local_z_axis = -global_z_axis X
    // local_x_axis:
    basis_matrix->block(0, 3, 3, 1) = -global_z_axis.cross(local_x_axis);
    // theta == pi (local_x_axis is perfectly aligned with -global_z_axis)
    // corresponds to anchor_local_z_axis = -global_y_axis:
    basis_matrix->block(0, 4, 3, 1) = -global_y_axis;

    // anchor_local_z_axis are the normalized weighted combination of the basis
    // vectors:
    anchor_local_z_axis =
        (*basis_matrix) * (*basis_vector_weights_unnormalized) /
        (basis_vector_weights_unnormalized->colwise().sum())(0, 0);
    anchor_local_z_axis =
        anchor_local_z_axis /
        (anchor_local_z_axis.norm());  // normalize the anchor_local_z_axis

    local_y_axis = anchor_local_z_axis.cross(local_x_axis);
    local_y_axis =
        local_y_axis / (local_y_axis.norm());  // normalize local y-axis
    local_z_axis = local_x_axis.cross(local_y_axis);
    local_z_axis = local_z_axis /
                   (local_z_axis.norm());  // normalize the real local z-axis
  } else if (ctraj_local_coordinate_frame_selection ==
             _SCHAAL_LOCAL_COORD_FRAME_) {  // Schaal's trajectory local
                                            // coordinate system formulation
    /*
     * [IMPORTANT] Notice the IF-ELSE block below:
     * >> IF   block: condition handled: local x-axis is "NOT TOO parallel" to
     * global z-axis (checked by: if dot-product or projection length is below
     * the threshold) handling:          local x-axis and global z-axis (~=
     * local z-axis) become the anchor axes, local y-axis is the cross-product
     * between them (z X x)
     * >> ELSE block: condition handled: local x-axis is "TOO parallel" to
     * global z-axis (checked by: if dot-product or projection length is above
     * the threshold) handling:          local x-axis and global y-axis (~=
     * local y-axis) become the anchor axes, local z-axis is the cross-product
     * between them (x X y) REMEMBER THESE ASSUMPTIONS!!! as this may introduce
     * a potential problems at some conditions in the future... To avoid
     * problems, during trajectory demonstrations, please AVOID demonstrating
     * trajectories whose start and end points are having very close global
     * z-coordinates (i.e. "vertical" trajectories)!!!
     */
    local_x_axis = goal_cart_position - start_cart_position;
    local_x_axis =
        local_x_axis / local_x_axis.norm();  // normalize local x-axis
    // check if local x-axis is "too parallel" with the global z-axis; if not:
    if (fabs(local_x_axis.dot(global_z_axis)) <=
        PARALLEL_VECTOR_PROJECTION_THRESHOLD) {
      local_z_axis = global_z_axis;
      local_y_axis = local_z_axis.cross(local_x_axis);
      local_y_axis =
          local_y_axis / local_y_axis.norm();  // normalize local y-axis
      local_z_axis = local_x_axis.cross(local_y_axis);
      local_z_axis =
          local_z_axis / local_z_axis.norm();  // normalize local z-axis
    } else {                                   // if it is "parallel enough":
      local_y_axis = global_y_axis;
      local_z_axis = local_x_axis.cross(local_y_axis);
      local_z_axis =
          local_z_axis / local_z_axis.norm();  // normalize local z-axis
      local_y_axis = local_z_axis.cross(local_x_axis);
      local_y_axis =
          local_y_axis / local_y_axis.norm();  // normalize local y-axis
    }
  } else if (ctraj_local_coordinate_frame_selection ==
             _KROEMER_LOCAL_COORD_FRAME_) {  // Kroemer's trajectory local
                                             // coordinate system formulation
    /*
     * See O.B. Kroemer, R. Detry, J. Piater, and J. Peters;
     *     Combining active learning and reactive control for robot grasping;
     *     Robotics and Autonomous Systems 58 (2010), page 1111 (Figure 5)
     * See the definition of CartesianCoordDMP in CartesianCoordDMP.h.
     */
    Vector3 approaching_goal_cart_position =
        cart_trajectory[cart_trajectory.size() - 2].getX().block(0, 0, 3, 1);
    local_x_axis = goal_cart_position - approaching_goal_cart_position;
    local_x_axis =
        local_x_axis / local_x_axis.norm();  // normalize local x-axis
    local_y_axis = goal_cart_position - start_cart_position;
    local_y_axis = local_y_axis /
                   local_y_axis.norm();  // normalize "temporary" local y-axis
    // check if local x-axis is "too parallel" with the "temporary" local
    // y-axis; if not:
    if (fabs(local_x_axis.dot(local_y_axis)) <=
        PARALLEL_VECTOR_PROJECTION_THRESHOLD) {
      local_z_axis = local_x_axis.cross(local_y_axis);
      local_z_axis =
          local_z_axis / local_z_axis.norm();  // normalize local z-axis
      local_y_axis = local_z_axis.cross(local_x_axis);
      local_y_axis =
          local_y_axis / local_y_axis.norm();  // normalize local y-axis
    } else {  // if it is "parallel enough" (following the same method as
              // Schaal's trajectory local coordinate system formulation):
        // check if local x-axis is "too parallel" with the global z-axis; if
        // not:
        if (fabs(local_x_axis.dot(global_z_axis)) <=
            PARALLEL_VECTOR_PROJECTION_THRESHOLD) {
          local_z_axis = global_z_axis;
          local_y_axis = local_z_axis.cross(local_x_axis);
          local_y_axis =
              local_y_axis / local_y_axis.norm();  // normalize local y-axis
          local_z_axis = local_x_axis.cross(local_y_axis);
          local_z_axis =
              local_z_axis / local_z_axis.norm();  // normalize local z-axis
        } else {  // if it is "parallel enough":
          local_y_axis = global_y_axis;
          local_z_axis = local_x_axis.cross(local_y_axis);
          local_z_axis =
              local_z_axis / local_z_axis.norm();  // normalize local z-axis
          local_y_axis = local_z_axis.cross(local_x_axis);
          local_y_axis =
              local_y_axis / local_y_axis.norm();  // normalize local y-axis
        }
      }
    }

    rotation_matrix << local_x_axis, local_y_axis, local_z_axis;

    rel_homogen_transform_matrix_local_to_global.block(0, 0, 3, 3)
        << rotation_matrix;
    rel_homogen_transform_matrix_local_to_global.block(0, 3, 3, 1)
        << translation_vector;
    rel_homogen_transform_matrix_local_to_global.block(3, 0, 1, 4) << 0.0, 0.0,
        0.0, 1.0;

    rel_homogen_transform_matrix_global_to_local.block(0, 0, 3, 3)
        << rotation_matrix.transpose();
    rel_homogen_transform_matrix_global_to_local.block(0, 3, 3, 1)
        << -(rotation_matrix.transpose()) * translation_vector;
    rel_homogen_transform_matrix_global_to_local.block(3, 0, 1, 4) << 0.0, 0.0,
        0.0, 1.0;

    return true;
  }

  bool CartesianCoordTransformer::convertCTrajDMPStatesToMatrices(
      const Trajectory& cart_trajectory, Matrix4xT& ctraj_H_matrix,
      Matrix3xT& ctrajd_matrix, Matrix3xT& ctrajdd_matrix,
      VectorT& ctraj_time_vector) {
    // pre-conditions checking
    if (rt_assert(CartesianCoordTransformer::isValid()) == false) {
      return false;
    }
    // input checking:
    cart_trajectory_size = cart_trajectory.size();
    if (rt_assert((rt_assert(cart_trajectory_size > 0)) &&
                  (rt_assert(cart_trajectory_size <= MAX_TRAJ_SIZE))) ==
        false) {
      return false;
    }
    if (rt_assert((rt_assert(ctraj_H_matrix.cols() == cart_trajectory_size)) &&
                  (rt_assert(ctrajd_matrix.cols() == cart_trajectory_size)) &&
                  (rt_assert(ctrajdd_matrix.cols() == cart_trajectory_size)) &&
                  (rt_assert(ctraj_time_vector.rows() ==
                             cart_trajectory_size))) == false) {
      return false;
    }

    for (int i = 0; i < cart_trajectory_size; i++) {
      if (rt_assert(cart_trajectory[i].getDMPNumDimensions() == 3) == false) {
        return false;
      }

      ctraj_H_matrix.block(0, i, 3, 1) =
          cart_trajectory[i].getX().block(0, 0, 3, 1);
      ctrajd_matrix.block(0, i, 3, 1) =
          cart_trajectory[i].getXd().block(0, 0, 3, 1);
      ctrajdd_matrix.block(0, i, 3, 1) =
          cart_trajectory[i].getXdd().block(0, 0, 3, 1);
      ctraj_time_vector(i, 0) = cart_trajectory[i].getTime();
    }
    ctraj_H_matrix.row(3).setOnes();

    return true;
  }

  bool CartesianCoordTransformer::computeCTrajAtNewCoordSys(
      const Matrix4xT& ctraj_H_matrix_old, const Matrix3xT& ctrajd_matrix_old,
      const Matrix3xT& ctrajdd_matrix_old,
      const Matrix4x4& rel_homogen_transform_matrix_old_to_new,
      Matrix4xT& ctraj_H_matrix_new, Matrix3xT& ctrajd_matrix_new,
      Matrix3xT& ctrajdd_matrix_new) {
    // pre-conditions checking
    if (rt_assert(CartesianCoordTransformer::isValid()) == false) {
      return false;
    }
    // input checking:
    cart_trajectory_size = ctrajdd_matrix_new.cols();
    if (rt_assert((rt_assert(cart_trajectory_size > 0)) &&
                  (rt_assert(cart_trajectory_size <= MAX_TRAJ_SIZE))) ==
        false) {
      return false;
    }
    if (rt_assert(
            (rt_assert(ctraj_H_matrix_old.cols() == cart_trajectory_size)) &&
            (rt_assert(ctrajd_matrix_old.cols() == cart_trajectory_size)) &&
            (rt_assert(ctrajdd_matrix_old.cols() == cart_trajectory_size)) &&
            (rt_assert(ctraj_H_matrix_new.cols() == cart_trajectory_size)) &&
            (rt_assert(ctrajd_matrix_new.cols() == cart_trajectory_size))) ==
        false) {
      return false;
    }
    if (rt_assert((rt_assert(ctraj_H_matrix_old.isZero() == false)) &&
                  (rt_assert(rel_homogen_transform_matrix_old_to_new.isZero() ==
                             false))) == false) {
      return false;
    }

    ctraj_H_matrix_new =
        rel_homogen_transform_matrix_old_to_new * ctraj_H_matrix_old;
    ctrajd_matrix_new =
        rel_homogen_transform_matrix_old_to_new.block(0, 0, 3, 3) *
        ctrajd_matrix_old;
    ctrajdd_matrix_new =
        rel_homogen_transform_matrix_old_to_new.block(0, 0, 3, 3) *
        ctrajdd_matrix_old;

    return true;
  }

  bool CartesianCoordTransformer::computeCTrajAtNewCoordSys(
      Vector4 * cart_H_old, Vector3 * cartd_old, Vector3 * cartdd_old,
      Vector3 * cart_vector_old,
      const Matrix4x4& rel_homogen_transform_matrix_old_to_new,
      Vector4* cart_H_new, Vector3* cartd_new, Vector3* cartdd_new,
      Vector3* cart_vector_new) {
    // pre-conditions checking
    if (rt_assert(CartesianCoordTransformer::isValid()) == false) {
      return false;
    }
    // input checking:
    if (rt_assert(rel_homogen_transform_matrix_old_to_new.isZero() == false) ==
        false) {
      return false;
    }

    if (cart_H_old) {
      if (rt_assert((*cart_H_old)(3) == 1.0) == false) {
        return false;
      }
      if (rt_assert(cart_H_new != NULL) == false) {
        return false;
      }

      (*cart_H_new) = rel_homogen_transform_matrix_old_to_new * (*cart_H_old);
    }

    if (cartd_old) {
      if (rt_assert(cartd_new != NULL) == false) {
        return false;
      }

      (*cartd_new) = rel_homogen_transform_matrix_old_to_new.block<3, 3>(0, 0) *
                     (*cartd_old);
    }

    if (cartdd_old) {
      if (rt_assert(cartdd_new != NULL) == false) {
        return false;
      }

      (*cartdd_new) =
          rel_homogen_transform_matrix_old_to_new.block<3, 3>(0, 0) *
          (*cartdd_old);
    }

    if (cart_vector_old) {
      if (rt_assert(cart_vector_new != NULL) == false) {
        return false;
      }

      (*cart_vector_new) =
          rel_homogen_transform_matrix_old_to_new.block<3, 3>(0, 0) *
          (*cart_vector_old);
    }

    return true;
  }

  bool CartesianCoordTransformer::computeCPosAtNewCoordSys(
      const Vector3& pos_3D_old,
      const Matrix4x4& rel_homogen_transform_matrix_old_to_new,
      Vector3& pos_3D_new) {
    // pre-conditions checking
    if (rt_assert(CartesianCoordTransformer::isValid()) == false) {
      return false;
    }
    // input checking:
    if (rt_assert(rel_homogen_transform_matrix_old_to_new.isZero() == false) ==
        false) {
      return false;
    }

    Vector4 pos_3D_H_old = ZeroVector4;
    Vector4 pos_3D_H_new = ZeroVector4;

    pos_3D_H_old.block(0, 0, 3, 1) = pos_3D_old;
    pos_3D_H_old(3) = 1.0;
    pos_3D_H_new = rel_homogen_transform_matrix_old_to_new * pos_3D_H_old;
    if (rt_assert(pos_3D_H_new(3) == 1.0) == false) {
      return false;
    }
    pos_3D_new = pos_3D_H_new.block(0, 0, 3, 1);

    return true;
  }

  bool CartesianCoordTransformer::convertCTrajMatricesToDMPStates(
      const Matrix4xT& ctraj_H_matrix, const Matrix3xT& ctrajd_matrix,
      const Matrix3xT& ctrajdd_matrix, const VectorT& ctraj_time_vector,
      Trajectory& cart_trajectory) {
    // pre-conditions checking
    if (rt_assert(CartesianCoordTransformer::isValid()) == false) {
      return false;
    }
    // input checking:
    cart_trajectory_size = ctrajdd_matrix.cols();
    if (rt_assert((rt_assert(cart_trajectory_size > 0)) &&
                  (rt_assert(cart_trajectory_size <= MAX_TRAJ_SIZE))) ==
        false) {
      return false;
    }
    if (rt_assert((rt_assert(ctraj_H_matrix.cols() == cart_trajectory_size)) &&
                  (rt_assert(ctrajd_matrix.cols() == cart_trajectory_size)) &&
                  (rt_assert(ctraj_time_vector.rows() ==
                             cart_trajectory_size))) == false) {
      return false;
    }
    if (rt_assert(cart_trajectory.size() == cart_trajectory_size) == false) {
      return false;
    }
    if (rt_assert((rt_assert(ctraj_H_matrix.isZero() == false)) &&
                  (rt_assert(ctraj_H_matrix.row(3).isOnes()))) == false) {
      return false;
    }

    for (int i = 0; i < cart_trajectory_size; i++) {
      cart_trajectory[i] = DMPState(ctraj_H_matrix.block(0, i, 3, 1),
                                    ctrajd_matrix.block(0, i, 3, 1),
                                    ctrajdd_matrix.block(0, i, 3, 1),
                                    ctraj_time_vector(i, 0), rt_assertor);
    }

    return true;
  }

  bool CartesianCoordTransformer::convertCTrajAtOldToNewCoordSys(
      const Trajectory& cart_trajectory_old_vector,
      const Matrix4x4& rel_homogen_transform_matrix_old_to_new,
      Trajectory& cart_trajectory_new_vector) {
    // pre-conditions checking
    if (rt_assert(CartesianCoordTransformer::isValid()) == false) {
      return false;
    }
    // input checking:
    cart_trajectory_size = cart_trajectory_new_vector.size();
    if (rt_assert((rt_assert(cart_trajectory_size > 0)) &&
                  (rt_assert(cart_trajectory_size <= MAX_TRAJ_SIZE))) ==
        false) {
      return false;
    }
    if (rt_assert(cart_trajectory_old_vector.size() == cart_trajectory_size) ==
        false) {
      return false;
    }
    if (rt_assert(rel_homogen_transform_matrix_old_to_new.isZero() == false) ==
        false) {
      return false;
    }

    cctraj_time_vector->resize(cart_trajectory_size);

    cctraj_old_coord_H_matrix->resize(4, cart_trajectory_size);
    cctrajd_old_coord_matrix->resize(3, cart_trajectory_size);
    cctrajdd_old_coord_matrix->resize(3, cart_trajectory_size);

    cctraj_new_coord_H_matrix->resize(4, cart_trajectory_size);
    cctrajd_new_coord_matrix->resize(3, cart_trajectory_size);
    cctrajdd_new_coord_matrix->resize(3, cart_trajectory_size);

    if (rt_assert(convertCTrajDMPStatesToMatrices(
            cart_trajectory_old_vector, *cctraj_old_coord_H_matrix,
            *cctrajd_old_coord_matrix, *cctrajdd_old_coord_matrix,
            *cctraj_time_vector)) == false) {
      return false;
    }
    if (rt_assert(computeCTrajAtNewCoordSys(
            *cctraj_old_coord_H_matrix, *cctrajd_old_coord_matrix,
            *cctrajdd_old_coord_matrix, rel_homogen_transform_matrix_old_to_new,
            *cctraj_new_coord_H_matrix, *cctrajd_new_coord_matrix,
            *cctrajdd_new_coord_matrix)) == false) {
      return false;
    }
    if (rt_assert(convertCTrajMatricesToDMPStates(
            *cctraj_new_coord_H_matrix, *cctrajd_new_coord_matrix,
            *cctrajdd_new_coord_matrix, *cctraj_time_vector,
            cart_trajectory_new_vector)) == false) {
      return false;
    }

    return true;
  }

  bool CartesianCoordTransformer::convertCTrajAtOldToNewCoordSys(
      const DMPState& dmpstate_old,
      const Matrix4x4& rel_homogen_transform_matrix_old_to_new,
      DMPState& dmpstate_new) {
    // pre-conditions checking
    if (rt_assert(CartesianCoordTransformer::isValid()) == false) {
      return false;
    }
    // input checking:
    if (rt_assert(rel_homogen_transform_matrix_old_to_new.isZero() == false) ==
        false) {
      return false;
    }
    if (rt_assert((rt_assert(dmpstate_old.getDMPNumDimensions() == 3)) &&
                  (rt_assert(dmpstate_new.getDMPNumDimensions() == 3))) ==
        false) {
      return false;
    }

    Vector4 cart_H_old;
    Vector3 cartd_old;
    Vector3 cartdd_old;

    Vector4 cart_H_new;
    Vector3 cartd_new;
    Vector3 cartdd_new;

    cart_H_old.block(0, 0, 3, 1) = dmpstate_old.getX();
    cart_H_old(3) = 1.0;
    cartd_old.block(0, 0, 3, 1) = dmpstate_old.getXd();
    cartdd_old.block(0, 0, 3, 1) = dmpstate_old.getXdd();

    if (rt_assert(computeCTrajAtNewCoordSys(
            &cart_H_old, &cartd_old, &cartdd_old, NULL,
            rel_homogen_transform_matrix_old_to_new, &cart_H_new, &cartd_new,
            &cartdd_new, NULL)) == false) {
      return false;
    }

    if (rt_assert(dmpstate_new.setX(cart_H_new.block(0, 0, 3, 1))) == false) {
      return false;
    }
    if (rt_assert(dmpstate_new.setXd(cartd_new.block(0, 0, 3, 1))) == false) {
      return false;
    }
    if (rt_assert(dmpstate_new.setXdd(cartdd_new.block(0, 0, 3, 1))) == false) {
      return false;
    }
    if (rt_assert(dmpstate_new.setTime(dmpstate_old.getTime())) == false) {
      return false;
    }

    return true;
  }

  CartesianCoordTransformer::~CartesianCoordTransformer() {}

}  // namespace dmp
