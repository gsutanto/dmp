%% Test of Unit Quaternion's Log Mapping
% Author: Giovanni Sutanto
% Date  : Monday, July 17, 2017
% Description:
%    An example of computation of log mapping for 
%    control error signal, for orientation which is 
%    the result of rotation w.r.t. global z-axis.

clear all;
close all;
clc;

addpath('../../../utilities/');
addpath('../');
addpath('../../../utilities/quaternion/');

R_orig              = eye(3);
R_rot_120deg_wrt_z  = rot_mat_wrt_z(2*pi/3.0);
R_rot_240deg_wrt_z  = rot_mat_wrt_z(4*pi/3.0);

% by the way 120 degrees are equivalent to: (in radian)
disp(['120 degrees = 2.0*pi/3.0 = ', num2str(2*pi/3.0), ' radian']);

Q_orig              = quaternion.rotationmatrix(R_orig).e;
Q_rot_120deg_wrt_z  = quaternion.rotationmatrix(R_rot_120deg_wrt_z).e;
Q_rot_240deg_wrt_z  = quaternion.rotationmatrix(R_rot_240deg_wrt_z).e;
minus_Q_rot_120deg_wrt_z    = -Q_rot_120deg_wrt_z;  % theoretically this represent the same orientation as Q_rot_120deg_wrt_z...

% please observe that both Q_rot_120deg_wrt_z and minus_Q_rot_120deg_wrt_z
% are mapped to the same rotation matrix,
% i.e. R_minus_Q_rot_120deg_wrt_z = R_rot_120deg_wrt_z:
R_minus_Q_rot_120deg_wrt_z  = RotationMatrix(quaternion(minus_Q_rot_120deg_wrt_z));

quat_diff_Q_orig_and_Q_rot_120deg_wrt_z = computeQuatProduct(...
                                         	normalizeQuaternion(Q_orig), ...
                                            computeQuatConjugate(Q_rot_120deg_wrt_z));
quat_diff_Q_orig_and_minus_Q_rot_120deg_wrt_z   = computeQuatProduct(...
                                                    normalizeQuaternion(Q_orig), ...
                                                    computeQuatConjugate(minus_Q_rot_120deg_wrt_z));
quat_diff_Q_orig_and_Q_rot_240deg_wrt_z         = computeQuatProduct(...
                                                    normalizeQuaternion(Q_orig), ...
                                                    computeQuatConjugate(Q_rot_240deg_wrt_z));

% 2Xlog(Q1 o Q2*): (NO standardization)
twice_log_no_std_quat_diff_Q_orig_and_Q_rot_120deg_wrt_z        = computeTwiceLogQuatDifference(Q_orig, Q_rot_120deg_wrt_z, 0);
twice_log_no_std_quat_diff_Q_orig_and_minus_Q_rot_120deg_wrt_z  = computeTwiceLogQuatDifference(Q_orig, minus_Q_rot_120deg_wrt_z, 0);
twice_log_no_std_quat_diff_Q_orig_and_Q_rot_240deg_wrt_z        = computeTwiceLogQuatDifference(Q_orig, Q_rot_240deg_wrt_z, 0);

% 2Xlog(standardize(Q1 o Q2*)): (with standardization)
twice_log_std_quat_diff_Q_orig_and_Q_rot_120deg_wrt_z       = computeTwiceLogQuatDifference(Q_orig, Q_rot_120deg_wrt_z, 1);
twice_log_std_quat_diff_Q_orig_and_minus_Q_rot_120deg_wrt_z = computeTwiceLogQuatDifference(Q_orig, minus_Q_rot_120deg_wrt_z, 1);
twice_log_std_quat_diff_Q_orig_and_Q_rot_240deg_wrt_z       = computeTwiceLogQuatDifference(Q_orig, Q_rot_240deg_wrt_z, 1);

% log(R1 * R2^T):
rot_mat_diff_R_orig_and_R_rot_120deg_wrt_z          = computeLogMapRotMat(R_orig * R_rot_120deg_wrt_z.');
rot_mat_diff_R_orig_and_R_minus_Q_rot_120deg_wrt_z  = computeLogMapRotMat(R_orig * R_minus_Q_rot_120deg_wrt_z.');
rot_mat_diff_R_orig_and_R_rot_240deg_wrt_z          = computeLogMapRotMat(R_orig * R_rot_240deg_wrt_z.');

function [R] = rot_mat_wrt_z(theta_radians)
    R   = [cos(theta_radians), -sin(theta_radians), 0;...
           sin(theta_radians), cos(theta_radians),  0;...
           0, 0, 1];
end