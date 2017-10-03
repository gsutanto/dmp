#include "amd_clmc_dmp/utility/utility_cartesian.h"

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
                                             Matrix3x3& V_hat)
    {
        V_hat       = ZeroMatrix3x3;
        V_hat(0,1)  = -V(2,0);
        V_hat(1,0)  = V(2,0);
        V_hat(0,2)  = V(1,0);
        V_hat(2,0)  = -V(1,0);
        V_hat(1,2)  = -V(0,0);
        V_hat(2,1)  = V(0,0);

        return true;
    }

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
                                           Matrix4x4& T)
    {
        if (normalizeQuaternion(Q) == false)
        {
            return false;
        }
        Eigen::Quaternion<double>   q(Q(0,0), Q(1,0), Q(2,0), Q(3,0));

        T                           = EyeMatrix4x4;
        T.block(0,0,3,3)            = q.matrix();
        T.block(0,3,3,1)            = t;

        return true;
    }

    // REAL-TIME REQUIREMENTS
    /**
     * Compute the inverse of a homogeneous transformation matrix T = [R, t; 0, 1],
     * i.e. T_inv = [R^T, -R^T * t; 0, 1].
     * @param T Input homogeneous transformation matrix
     * @param T_inv Output inverse homogeneous transformation matrix
     * @return Success (true) or failure (false)
     */
    bool computeInverseHomogeneousTransformMatrix(Matrix4x4 T,
                                                  Matrix4x4& T_inv)
    {
        dmp::Vector3    t           = T.block(0,3,3,1);

        dmp::Matrix3x3  R           = T.block(0,0,3,3);
        Eigen::Quaternion<double>   q(R);
        q.normalize();
        R                           = q.matrix();
        dmp::Matrix3x3  R_transpose = R.transpose();

        T_inv                       = EyeMatrix4x4;
        T_inv.block(0,0,3,3)        = R_transpose;
        T_inv.block(0,3,3,1)        = -R_transpose * t;

        return true;
    }

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
                                         QuaternionDMPState& quat_state_coordsys1_to_coordsys3)
    {
        Matrix3x3   R_coordsys1_to_coordsys2    = T_coordsys1_to_coordsys2.block(0,0,3,3);
        Vector3     X_coordsys2_to_coordsys3    = cart_coord_state_coordsys2_to_coordsys3.getX();
        Vector4     Q_coordsys2_to_coordsys3    = quat_state_coordsys2_to_coordsys3.getQ();

        Matrix4x4   T_coordsys2_to_coordsys3;
        if (computeHomogeneousTransformMatrix(X_coordsys2_to_coordsys3,
                                              Q_coordsys2_to_coordsys3,
                                              T_coordsys2_to_coordsys3) == false)
        {
            return false;
        }

        Matrix4x4   T_coordsys1_to_coordsys3    = T_coordsys1_to_coordsys2 * T_coordsys2_to_coordsys3;

        Vector3     X_coordsys1_to_coordsys3    = T_coordsys1_to_coordsys3.block(0,3,3,1);
        Vector3     Xd_coordsys1_to_coordsys3   = R_coordsys1_to_coordsys2 * cart_coord_state_coordsys2_to_coordsys3.getXd();
        Vector3     Xdd_coordsys1_to_coordsys3  = R_coordsys1_to_coordsys2 * cart_coord_state_coordsys2_to_coordsys3.getXdd();

        Matrix3x3   R_coordsys1_to_coordsys3    = T_coordsys1_to_coordsys3.block(0,0,3,3);
        Eigen::Quaternion<double>               q_coordsys1_to_coordsys3(R_coordsys1_to_coordsys3);
        Vector4                                 Q_coordsys1_to_coordsys3;
        Q_coordsys1_to_coordsys3(0,0)               = q_coordsys1_to_coordsys3.w();
        Q_coordsys1_to_coordsys3(1,0)               = q_coordsys1_to_coordsys3.x();
        Q_coordsys1_to_coordsys3(2,0)               = q_coordsys1_to_coordsys3.y();
        Q_coordsys1_to_coordsys3(3,0)               = q_coordsys1_to_coordsys3.z();
        Vector3     omega_coordsys1_to_coordsys3    = R_coordsys1_to_coordsys2 * quat_state_coordsys2_to_coordsys3.getOmega();
        Vector3     omegad_coordsys1_to_coordsys3   = R_coordsys1_to_coordsys2 * quat_state_coordsys2_to_coordsys3.getOmegad();

        if (cart_coord_state_coordsys1_to_coordsys3.setX(X_coordsys1_to_coordsys3) == false)
        {
            return false;
        }
        if (cart_coord_state_coordsys1_to_coordsys3.setXd(Xd_coordsys1_to_coordsys3) == false)
        {
            return false;
        }
        if (cart_coord_state_coordsys1_to_coordsys3.setXdd(Xdd_coordsys1_to_coordsys3) == false)
        {
            return false;
        }

        if (quat_state_coordsys1_to_coordsys3.setQ(Q_coordsys1_to_coordsys3) == false)
        {
            return false;
        }
        if (quat_state_coordsys1_to_coordsys3.setOmega(omega_coordsys1_to_coordsys3) == false)
        {
            return false;
        }
        if (quat_state_coordsys1_to_coordsys3.setOmegad(omegad_coordsys1_to_coordsys3) == false)
        {
            return false;
        }
        if (quat_state_coordsys1_to_coordsys3.computeQdAndQdd() == false)
        {
            return false;
        }

        return true;
    }

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
                                         QuaternionDMPState& quat_state_coordsys1_to_coordsys4)
    {
        Matrix3x3   R_coordsys1_to_coordsys2    = T_coordsys1_to_coordsys2.block(0,0,3,3);
        Vector3     X_coordsys2_to_coordsys3    = cart_coord_state_coordsys2_to_coordsys3.getX();
        Vector4     Q_coordsys2_to_coordsys3    = quat_state_coordsys2_to_coordsys3.getQ();

        Matrix4x4   T_coordsys2_to_coordsys3;
        if (computeHomogeneousTransformMatrix(X_coordsys2_to_coordsys3,
                                              Q_coordsys2_to_coordsys3,
                                              T_coordsys2_to_coordsys3) == false)
        {
            return false;
        }

        Matrix4x4   T_coordsys1_to_coordsys4    = T_coordsys1_to_coordsys2 * T_coordsys2_to_coordsys3 * T_coordsys3_to_coordsys4;

        // compute pose (position and orientation):
        Vector3     X_coordsys1_to_coordsys4    = T_coordsys1_to_coordsys4.block(0,3,3,1);
        Matrix3x3   R_coordsys1_to_coordsys4    = T_coordsys1_to_coordsys4.block(0,0,3,3);
        Eigen::Quaternion<double>               q_coordsys1_to_coordsys4(R_coordsys1_to_coordsys4);
        Vector4                                 Q_coordsys1_to_coordsys4;
        Q_coordsys1_to_coordsys4(0,0)           = q_coordsys1_to_coordsys4.w();
        Q_coordsys1_to_coordsys4(1,0)           = q_coordsys1_to_coordsys4.x();
        Q_coordsys1_to_coordsys4(2,0)           = q_coordsys1_to_coordsys4.y();
        Q_coordsys1_to_coordsys4(3,0)           = q_coordsys1_to_coordsys4.z();

        // compute velocity, acceleration, angular velocity, and angular acceleration:
        Vector3     Xd_coordsys2_to_coordsys3   = cart_coord_state_coordsys2_to_coordsys3.getXd();
        Vector3     Xdd_coordsys2_to_coordsys3  = cart_coord_state_coordsys2_to_coordsys3.getXdd();

        Vector3     omega_coordsys2_to_coordsys3    = quat_state_coordsys2_to_coordsys3.getOmega();
        Vector3     omegad_coordsys2_to_coordsys3   = quat_state_coordsys2_to_coordsys3.getOmegad();

        Matrix3x3   omega_cross_product_matrix_coordsys2_to_coordsys3;
        Matrix3x3   omegad_cross_product_matrix_coordsys2_to_coordsys3;

        if (convertVector3ToSkewSymmetricMatrix(omega_coordsys2_to_coordsys3,
                                                omega_cross_product_matrix_coordsys2_to_coordsys3) == false)
        {
            return false;
        }

        if (convertVector3ToSkewSymmetricMatrix(omegad_coordsys2_to_coordsys3,
                                                omegad_cross_product_matrix_coordsys2_to_coordsys3) == false)
        {
            return false;
        }

        Eigen::Quaternion<double>   q_coordsys2_to_coordsys3(Q_coordsys2_to_coordsys3(0,0),
                                                             Q_coordsys2_to_coordsys3(1,0),
                                                             Q_coordsys2_to_coordsys3(2,0),
                                                             Q_coordsys2_to_coordsys3(3,0));
        Matrix3x3   R_coordsys2_to_coordsys3    = q_coordsys2_to_coordsys3.matrix();

        Vector3     t_coordsys3_to_coordsys4    = T_coordsys3_to_coordsys4.block(0,3,3,1);

        Vector3     Xd_coordsys2_to_coordsys4   = Xd_coordsys2_to_coordsys3 
                                                  + (omega_cross_product_matrix_coordsys2_to_coordsys3 
                                                     * R_coordsys2_to_coordsys3 * t_coordsys3_to_coordsys4);
        Vector3     omega_coordsys2_to_coordsys4= omega_coordsys2_to_coordsys3;

        Vector3     Xdd_coordsys2_to_coordsys4      = Xdd_coordsys2_to_coordsys3
                                                      + (omegad_cross_product_matrix_coordsys2_to_coordsys3
                                                         * R_coordsys2_to_coordsys3 * t_coordsys3_to_coordsys4);
        Vector3     omegad_coordsys2_to_coordsys4   = omegad_coordsys2_to_coordsys3;

        Vector3     Xd_coordsys1_to_coordsys4       = R_coordsys1_to_coordsys2 * Xd_coordsys2_to_coordsys4;
        Vector3     Xdd_coordsys1_to_coordsys4      = R_coordsys1_to_coordsys2 * Xdd_coordsys2_to_coordsys4;
        Vector3     omega_coordsys1_to_coordsys4    = R_coordsys1_to_coordsys2 * omega_coordsys2_to_coordsys4;
        Vector3     omegad_coordsys1_to_coordsys4   = R_coordsys1_to_coordsys2 * omegad_coordsys2_to_coordsys4;

        if (cart_coord_state_coordsys1_to_coordsys4.setX(X_coordsys1_to_coordsys4) == false)
        {
            return false;
        }
        if (cart_coord_state_coordsys1_to_coordsys4.setXd(Xd_coordsys1_to_coordsys4) == false)
        {
            return false;
        }
        if (cart_coord_state_coordsys1_to_coordsys4.setXdd(Xdd_coordsys1_to_coordsys4) == false)
        {
            return false;
        }

        if (quat_state_coordsys1_to_coordsys4.setQ(Q_coordsys1_to_coordsys4) == false)
        {
            return false;
        }
        if (quat_state_coordsys1_to_coordsys4.setOmega(omega_coordsys1_to_coordsys4) == false)
        {
            return false;
        }
        if (quat_state_coordsys1_to_coordsys4.setOmegad(omegad_coordsys1_to_coordsys4) == false)
        {
            return false;
        }
        if (quat_state_coordsys1_to_coordsys4.computeQdAndQdd() == false)
        {
            return false;
        }

        return true;
    }

}
