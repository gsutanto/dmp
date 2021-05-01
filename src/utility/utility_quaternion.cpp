#include "dmp/utility/utility_quaternion.h"

namespace dmp {

bool normalizeQuaternion(Vector4& Q) {
  double norm_Q = Q.norm();
  if (norm_Q <= 0) {
    return false;
  }
  Q = Q / norm_Q;
  return true;
}

bool standardizeNormalizeQuaternion(Vector4& Q) {
  if (Q(0, 0) < 0) {
    Q = -Q;
  }
  return (normalizeQuaternion(Q));
}

Vector4 computeQuaternionComposition(Vector4 Q_a, Vector4 Q_b) {
  double ua, qa1, qa2, qa3;
  ua = Q_a(0, 0);
  qa1 = Q_a(1, 0);
  qa2 = Q_a(2, 0);
  qa3 = Q_a(3, 0);

  Matrix4x4 A;
  A(0, 0) = ua;
  A(1, 0) = qa1;
  A(2, 0) = qa2;
  A(3, 0) = qa3;
  A(0, 1) = -qa1;
  A(1, 1) = ua;
  A(2, 1) = qa3;
  A(3, 1) = -qa2;
  A(0, 2) = -qa2;
  A(1, 2) = -qa3;
  A(2, 2) = ua;
  A(3, 2) = qa1;
  A(0, 3) = -qa3;
  A(1, 3) = qa2;
  A(2, 3) = -qa1;
  A(3, 3) = ua;

  return (A * Q_b);
}

Vector4 computeQuaternionConjugate(const Vector4& Q) {
  Vector4 out_Q = -Q;
  out_Q(0, 0) = Q(0, 0);
  return out_Q;
}

bool computeQd(Vector4 input_Q, const Vector3& input_omega,
               Vector4& output_Qd) {
  if (normalizeQuaternion(input_Q) == false) {
    return false;
  }
  Vector4 temp1;
  temp1(0, 0) = 0;
  temp1.block(1, 0, 3, 1) = input_omega;

  output_Qd = (0.5 * computeQuaternionComposition(temp1, input_Q));

  return true;
}

bool computeQdd(Vector4 input_Q, const Vector4& input_Qd,
                const Vector3& input_omega, const Vector3& input_omegad,
                Vector4& output_Qdd) {
  if (normalizeQuaternion(input_Q) == false) {
    return false;
  }
  Vector4 temp1, temp2, temp3, temp4;
  temp1(0, 0) = 0;
  temp1.block(1, 0, 3, 1) = input_omegad;

  temp2(0, 0) = 0;
  temp2.block(1, 0, 3, 1) = input_omega;

  temp3 = computeQuaternionComposition(temp1, input_Q);
  temp4 = computeQuaternionComposition(temp2, input_Qd);
  output_Qdd = (0.5 * (temp3 + temp4));

  return true;
}

bool computeExpMap_so3_to_SO3(const Vector3& input_r, Vector4& output_exp_r) {
  Vector3 half_input_r = input_r / 2.0;
  double half_norm_r = half_input_r.norm();
  output_exp_r = ZeroVector4;

  if (half_norm_r != 0) {
    output_exp_r(0, 0) = cos(half_norm_r);
    double sin_half_norm_r = sin(half_norm_r);
    if (sin_half_norm_r > 0) {
      output_exp_r.block(1, 0, 3, 1) =
          exp(log(sin_half_norm_r) - log(half_norm_r)) * (half_input_r);
    } else if (sin_half_norm_r < 0) {
      output_exp_r.block(1, 0, 3, 1) =
          exp(log(-sin_half_norm_r) - log(half_norm_r)) * (-half_input_r);
    } else {  // if (sin_half_norm_r == 0)
      output_exp_r(0, 0) = 1.0;
      output_exp_r.block(1, 0, 3, 1) = ZeroVector3;
    }
  } else {  // if (half_norm_r == 0)
    output_exp_r(0, 0) = 1.0;
    output_exp_r.block(1, 0, 3, 1) = ZeroVector3;
  }

  // don't forget to normalize the resulting Quaternion:
  return normalizeQuaternion(output_exp_r);
}

bool computeLogMap_SO3_to_so3(const Vector4& input_Q, Vector3& output_log_Q) {
  Vector4 preprocessed_input_Q = input_Q;
  // normalize the input Quaternion first:
  if (normalizeQuaternion(preprocessed_input_Q) == false) {
    return false;
  }
  double u = preprocessed_input_Q(0, 0);
  if ((u < -1.0) || (u > 1.0)) {
    return false;
  }
  double acos_u = acos(u);
  double sin_acos_u = sin(acos_u);
  Vector3 q = preprocessed_input_Q.block(1, 0, 3, 1);

  if (sin_acos_u != 0) {
    int multiplier_sign = 1;
    double log_multiplier =
        log(acos_u);  // here acos_u is within the range (0, pi] (strictly
                      // positive), so log(acos_u) should be defined.
    if (sin_acos_u > 0) {
      log_multiplier = log_multiplier - log(sin_acos_u);
    } else if (sin_acos_u < 0) {
      multiplier_sign = -multiplier_sign;
      log_multiplier = log_multiplier - log(-sin_acos_u);
    }

    for (uint idx = 0; idx < 3; idx++) {
      if (q(idx, 0) > 0) {
        output_log_Q(idx, 0) =
            multiplier_sign * exp(log_multiplier + log(q(idx, 0)));
      } else if (q(idx, 0) < 0) {
        output_log_Q(idx, 0) =
            (-multiplier_sign) * exp(log_multiplier + log(-q(idx, 0)));
      } else {  // if (q(idx,0) == 0)
        output_log_Q(idx, 0) = 0;
      }
    }
  } else {  // if (sin_acos_u == 0)
    output_log_Q = q;
  }

  output_log_Q *= 2.0;

  return true;
}

bool computeLogQuaternionDifference(Vector4 in_Q1, Vector4 in_Q2,
                                    Vector3& out_result) {
  if (normalizeQuaternion(in_Q1) == false) {
    return false;
  }
  if (normalizeQuaternion(in_Q2) == false) {
    return false;
  }

  Vector4 Q_intermediate =
      computeQuaternionComposition(in_Q1, computeQuaternionConjugate(in_Q2));

  if (standardizeNormalizeQuaternion(Q_intermediate) == false) {
    return false;
  }

  Vector3 so3_variable = ZeroVector3;
  if (computeLogMap_SO3_to_so3(Q_intermediate, so3_variable) == false) {
    return false;
  }

  out_result = so3_variable;

  return true;
}

bool integrateQuaternion(Vector4 in_Q_t, Vector3 in_omega_t, double in_dt,
                         Vector4& out_Q_t_plus_1) {
  Vector3 theta_v = in_omega_t * in_dt;
  Vector4 Q_increment = ZeroVector4;
  if (normalizeQuaternion(in_Q_t) == false) {
    return false;
  }
  if (computeExpMap_so3_to_SO3(theta_v, Q_increment) == false) {
    return false;
  }
  out_Q_t_plus_1 = computeQuaternionComposition(Q_increment, in_Q_t);

  return (normalizeQuaternion(out_Q_t_plus_1));
}

}  // namespace dmp
