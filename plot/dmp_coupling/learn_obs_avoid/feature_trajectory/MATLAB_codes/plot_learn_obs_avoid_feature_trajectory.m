% A MATLAB script to plot the feature trajectory of 
% the obstacle avoidance coupling term, as DMP unrolls.
% Author: Giovanni Sutanto
% Date  : December 21, 2015
clear all;
close all;
clc;

num_obstacle_points     = dlmread('num_obstacle_points.txt');
N_k_phi3_grid           = dlmread('N_k_phi3_grid.txt');

beta_phi1_phi2_grid     = dlmread('beta_phi1_phi2_grid.txt');
beta_phi1_phi2_vector   = dlmread('beta_phi1_phi2_vector.txt');
k_phi1_phi2_grid        = dlmread('k_phi1_phi2_grid.txt');
k_phi1_phi2_vector      = dlmread('k_phi1_phi2_vector.txt');
k_phi3_vector           = dlmread('k_phi3_vector.txt');

param_matrix            = repmat([repmat([beta_phi1_phi2_vector, k_phi1_phi2_vector], num_obstacle_points, 1);
                                  [zeros(N_k_phi3_grid, 1), k_phi3_vector]], 3, 1);

feature_trajectory      = dlmread('feature_trajectory.txt');
feature_weights         = dlmread('learn_obs_avoid_weights_matrix.txt');
Ct_trajectory           = feature_trajectory * feature_weights;

feature_traj_diff       = diff(feature_trajectory,1,1);
[max_abs_diff, max_abs_diff_idx]        = max(abs(feature_traj_diff), [], 1);

endeff_cart_position_local_trajectory   = dlmread('endeff_cart_position_local_trajectory.txt');
endeff_cart_velocity_local_trajectory   = dlmread('endeff_cart_velocity_local_trajectory.txt');

obs_cart_position_local_trajectory      = dlmread('obstacle_cart_position_local_trajectory.txt');
obs_cart_velocity_local_trajectory      = dlmread('obstacle_cart_velocity_local_trajectory.txt');

tau_reproduce           = dlmread('tau_reproduce.txt');

[ r, norm_r, ee_to_obs ]= compute_rotation_axes( obs_cart_position_local_trajectory, endeff_cart_position_local_trajectory, endeff_cart_velocity_local_trajectory );
[ theta ]               = compute_theta( obs_cart_position_local_trajectory, endeff_cart_position_local_trajectory, endeff_cart_velocity_local_trajectory );
[ feature, R_vector ]   = compute_feature( obs_cart_position_local_trajectory, endeff_cart_position_local_trajectory, endeff_cart_velocity_local_trajectory, r, theta, param_matrix(1,1), param_matrix(1,2), tau_reproduce );

figure;
hold on;
h1                      = surf(feature_trajectory);
set(h1,'LineStyle','none');
title('Plot of Obstacle Avoidance Feature Trajectory');
xlabel('Feature');
ylabel('Time');
zlabel('Feature Magnitude');
hold off;

figure;
hold on;
h2                      = surf(feature_traj_diff);
set(h2,'LineStyle','none');
title('Plot of Obstacle Avoidance Feature Trajectory Difference');
xlabel('Feature');
ylabel('Time');
zlabel('Feature Magnitude Difference');
hold off;

figure;
hold on;
h3                      = plot3(Ct_trajectory(:,1), Ct_trajectory(:,2), Ct_trajectory(:,3));
title('Plot of Obstacle Avoidance Coupling Term Trajectory');
xlabel('Ct_x');
ylabel('Ct_y');
zlabel('Ct_z');
hold off;