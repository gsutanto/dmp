#include "dmp/utility/utility_cartesian.h"

namespace dmp {

bool convertVector3ToSkewSymmetricMatrix(const Vector3& V, Matrix3x3& V_hat) {
  V_hat = ZeroMatrix3x3;
  V_hat(0, 1) = -V(2, 0);
  V_hat(1, 0) = V(2, 0);
  V_hat(0, 2) = V(1, 0);
  V_hat(2, 0) = -V(1, 0);
  V_hat(1, 2) = -V(0, 0);
  V_hat(2, 1) = V(0, 0);

  return true;
}

bool computeHomogeneousTransformMatrix(Vector3 t, Vector4 Q, Matrix4x4& T) {
  if (normalizeQuaternion(Q) == false) {
    return false;
  }
  Eigen::Quaternion<double> q(Q(0, 0), Q(1, 0), Q(2, 0), Q(3, 0));

  T = EyeMatrix4x4;
  T.block(0, 0, 3, 3) = q.matrix();
  T.block(0, 3, 3, 1) = t;

  return true;
}

bool computeInverseHomogeneousTransformMatrix(Matrix4x4 T, Matrix4x4& T_inv) {
  dmp::Vector3 t = T.block(0, 3, 3, 1);

  dmp::Matrix3x3 R = T.block(0, 0, 3, 3);
  Eigen::Quaternion<double> q(R);
  q.normalize();
  R = q.matrix();
  dmp::Matrix3x3 R_transpose = R.transpose();

  T_inv = EyeMatrix4x4;
  T_inv.block(0, 0, 3, 3) = R_transpose;
  T_inv.block(0, 3, 3, 1) = -R_transpose * t;

  return true;
}

bool performStateCoordTransformation(
    Matrix4x4 T_coordsys1_to_coordsys2,
    DMPState cart_coord_state_coordsys2_to_coordsys3,
    QuaternionDMPState quat_state_coordsys2_to_coordsys3,
    DMPState& cart_coord_state_coordsys1_to_coordsys3,
    QuaternionDMPState& quat_state_coordsys1_to_coordsys3) {
  Matrix3x3 R_coordsys1_to_coordsys2 =
      T_coordsys1_to_coordsys2.block(0, 0, 3, 3);
  Vector3 X_coordsys2_to_coordsys3 =
      cart_coord_state_coordsys2_to_coordsys3.getX();
  Vector4 Q_coordsys2_to_coordsys3 = quat_state_coordsys2_to_coordsys3.getQ();

  Matrix4x4 T_coordsys2_to_coordsys3;
  if (computeHomogeneousTransformMatrix(X_coordsys2_to_coordsys3,
                                        Q_coordsys2_to_coordsys3,
                                        T_coordsys2_to_coordsys3) == false) {
    return false;
  }

  Matrix4x4 T_coordsys1_to_coordsys3 =
      T_coordsys1_to_coordsys2 * T_coordsys2_to_coordsys3;

  Vector3 X_coordsys1_to_coordsys3 = T_coordsys1_to_coordsys3.block(0, 3, 3, 1);
  Vector3 Xd_coordsys1_to_coordsys3 =
      R_coordsys1_to_coordsys2 *
      cart_coord_state_coordsys2_to_coordsys3.getXd();
  Vector3 Xdd_coordsys1_to_coordsys3 =
      R_coordsys1_to_coordsys2 *
      cart_coord_state_coordsys2_to_coordsys3.getXdd();

  Matrix3x3 R_coordsys1_to_coordsys3 =
      T_coordsys1_to_coordsys3.block(0, 0, 3, 3);
  Eigen::Quaternion<double> q_coordsys1_to_coordsys3(R_coordsys1_to_coordsys3);
  Vector4 Q_coordsys1_to_coordsys3;
  Q_coordsys1_to_coordsys3(0, 0) = q_coordsys1_to_coordsys3.w();
  Q_coordsys1_to_coordsys3(1, 0) = q_coordsys1_to_coordsys3.x();
  Q_coordsys1_to_coordsys3(2, 0) = q_coordsys1_to_coordsys3.y();
  Q_coordsys1_to_coordsys3(3, 0) = q_coordsys1_to_coordsys3.z();
  Vector3 omega_coordsys1_to_coordsys3 =
      R_coordsys1_to_coordsys2 * quat_state_coordsys2_to_coordsys3.getOmega();
  Vector3 omegad_coordsys1_to_coordsys3 =
      R_coordsys1_to_coordsys2 * quat_state_coordsys2_to_coordsys3.getOmegad();

  if (cart_coord_state_coordsys1_to_coordsys3.setX(X_coordsys1_to_coordsys3) ==
      false) {
    return false;
  }
  if (cart_coord_state_coordsys1_to_coordsys3.setXd(
          Xd_coordsys1_to_coordsys3) == false) {
    return false;
  }
  if (cart_coord_state_coordsys1_to_coordsys3.setXdd(
          Xdd_coordsys1_to_coordsys3) == false) {
    return false;
  }

  if (quat_state_coordsys1_to_coordsys3.setQ(Q_coordsys1_to_coordsys3) ==
      false) {
    return false;
  }
  if (quat_state_coordsys1_to_coordsys3.setOmega(
          omega_coordsys1_to_coordsys3) == false) {
    return false;
  }
  if (quat_state_coordsys1_to_coordsys3.setOmegad(
          omegad_coordsys1_to_coordsys3) == false) {
    return false;
  }
  if (quat_state_coordsys1_to_coordsys3.computeQdAndQdd() == false) {
    return false;
  }

  return true;
}

bool performStateCoordTransformation(
    Matrix4x4 T_coordsys1_to_coordsys2,
    DMPState cart_coord_state_coordsys2_to_coordsys3,
    QuaternionDMPState quat_state_coordsys2_to_coordsys3,
    Matrix4x4 T_coordsys3_to_coordsys4,
    DMPState& cart_coord_state_coordsys1_to_coordsys4,
    QuaternionDMPState& quat_state_coordsys1_to_coordsys4) {
  Matrix3x3 R_coordsys1_to_coordsys2 =
      T_coordsys1_to_coordsys2.block(0, 0, 3, 3);
  Vector3 X_coordsys2_to_coordsys3 =
      cart_coord_state_coordsys2_to_coordsys3.getX();
  Vector4 Q_coordsys2_to_coordsys3 = quat_state_coordsys2_to_coordsys3.getQ();

  Matrix4x4 T_coordsys2_to_coordsys3;
  if (computeHomogeneousTransformMatrix(X_coordsys2_to_coordsys3,
                                        Q_coordsys2_to_coordsys3,
                                        T_coordsys2_to_coordsys3) == false) {
    return false;
  }

  Matrix4x4 T_coordsys1_to_coordsys4 = T_coordsys1_to_coordsys2 *
                                       T_coordsys2_to_coordsys3 *
                                       T_coordsys3_to_coordsys4;

  // compute pose (position and orientation):
  Vector3 X_coordsys1_to_coordsys4 = T_coordsys1_to_coordsys4.block(0, 3, 3, 1);
  Matrix3x3 R_coordsys1_to_coordsys4 =
      T_coordsys1_to_coordsys4.block(0, 0, 3, 3);
  Eigen::Quaternion<double> q_coordsys1_to_coordsys4(R_coordsys1_to_coordsys4);
  Vector4 Q_coordsys1_to_coordsys4;
  Q_coordsys1_to_coordsys4(0, 0) = q_coordsys1_to_coordsys4.w();
  Q_coordsys1_to_coordsys4(1, 0) = q_coordsys1_to_coordsys4.x();
  Q_coordsys1_to_coordsys4(2, 0) = q_coordsys1_to_coordsys4.y();
  Q_coordsys1_to_coordsys4(3, 0) = q_coordsys1_to_coordsys4.z();

  // compute velocity, acceleration, angular velocity, and angular acceleration:
  Vector3 Xd_coordsys2_to_coordsys3 =
      cart_coord_state_coordsys2_to_coordsys3.getXd();
  Vector3 Xdd_coordsys2_to_coordsys3 =
      cart_coord_state_coordsys2_to_coordsys3.getXdd();

  Vector3 omega_coordsys2_to_coordsys3 =
      quat_state_coordsys2_to_coordsys3.getOmega();
  Vector3 omegad_coordsys2_to_coordsys3 =
      quat_state_coordsys2_to_coordsys3.getOmegad();

  Matrix3x3 omega_cross_product_matrix_coordsys2_to_coordsys3;
  Matrix3x3 omegad_cross_product_matrix_coordsys2_to_coordsys3;

  if (convertVector3ToSkewSymmetricMatrix(
          omega_coordsys2_to_coordsys3,
          omega_cross_product_matrix_coordsys2_to_coordsys3) == false) {
    return false;
  }

  if (convertVector3ToSkewSymmetricMatrix(
          omegad_coordsys2_to_coordsys3,
          omegad_cross_product_matrix_coordsys2_to_coordsys3) == false) {
    return false;
  }

  Eigen::Quaternion<double> q_coordsys2_to_coordsys3(
      Q_coordsys2_to_coordsys3(0, 0), Q_coordsys2_to_coordsys3(1, 0),
      Q_coordsys2_to_coordsys3(2, 0), Q_coordsys2_to_coordsys3(3, 0));
  Matrix3x3 R_coordsys2_to_coordsys3 = q_coordsys2_to_coordsys3.matrix();

  Vector3 t_coordsys3_to_coordsys4 = T_coordsys3_to_coordsys4.block(0, 3, 3, 1);

  Vector3 Xd_coordsys2_to_coordsys4 =
      Xd_coordsys2_to_coordsys3 +
      (omega_cross_product_matrix_coordsys2_to_coordsys3 *
       R_coordsys2_to_coordsys3 * t_coordsys3_to_coordsys4);
  Vector3 omega_coordsys2_to_coordsys4 = omega_coordsys2_to_coordsys3;

  Vector3 Xdd_coordsys2_to_coordsys4 =
      Xdd_coordsys2_to_coordsys3 +
      (omegad_cross_product_matrix_coordsys2_to_coordsys3 *
       R_coordsys2_to_coordsys3 * t_coordsys3_to_coordsys4);
  Vector3 omegad_coordsys2_to_coordsys4 = omegad_coordsys2_to_coordsys3;

  Vector3 Xd_coordsys1_to_coordsys4 =
      R_coordsys1_to_coordsys2 * Xd_coordsys2_to_coordsys4;
  Vector3 Xdd_coordsys1_to_coordsys4 =
      R_coordsys1_to_coordsys2 * Xdd_coordsys2_to_coordsys4;
  Vector3 omega_coordsys1_to_coordsys4 =
      R_coordsys1_to_coordsys2 * omega_coordsys2_to_coordsys4;
  Vector3 omegad_coordsys1_to_coordsys4 =
      R_coordsys1_to_coordsys2 * omegad_coordsys2_to_coordsys4;

  if (cart_coord_state_coordsys1_to_coordsys4.setX(X_coordsys1_to_coordsys4) ==
      false) {
    return false;
  }
  if (cart_coord_state_coordsys1_to_coordsys4.setXd(
          Xd_coordsys1_to_coordsys4) == false) {
    return false;
  }
  if (cart_coord_state_coordsys1_to_coordsys4.setXdd(
          Xdd_coordsys1_to_coordsys4) == false) {
    return false;
  }

  if (quat_state_coordsys1_to_coordsys4.setQ(Q_coordsys1_to_coordsys4) ==
      false) {
    return false;
  }
  if (quat_state_coordsys1_to_coordsys4.setOmega(
          omega_coordsys1_to_coordsys4) == false) {
    return false;
  }
  if (quat_state_coordsys1_to_coordsys4.setOmegad(
          omegad_coordsys1_to_coordsys4) == false) {
    return false;
  }
  if (quat_state_coordsys1_to_coordsys4.computeQdAndQdd() == false) {
    return false;
  }

  return true;
}

}  // namespace dmp
