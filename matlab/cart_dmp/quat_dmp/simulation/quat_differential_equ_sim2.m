% Quaternion Differential Equation Simulation
% 
% Author : Giovanni Sutanto
% Purpose: >> Simulating a Quaternion Differential Equation 
%             with Quaternion State that Converges to the Goal Quaternion
%          >> Showing a Comparison between:
%             (a) using Quaternion Error as the error signal
%             (b) using Log Mapping of Quaternion Difference as the error
%                 signal
%             both using using Log Mapping of Quaternion Difference Integration
% Date   : October 19, 2016

clear all;
close all;
clc;

addpath('../../../utilities/');
addpath('../');

D   = 3;
Dq  = 4;

cart_quat_traj_demo     = dlmread('../../../../data/cart_dmp/cart_dmp_wiggling/sample_quat_traj_recorded_demo_wiggling.txt');

tau         = cart_quat_traj_demo(end,1) - cart_quat_traj_demo(1,1);
traj_length = size(cart_quat_traj_demo,1);
dt          = tau/(traj_length-1);

% Goal Orientation Quaternion:
G_quat  = cart_quat_traj_demo(end,2:5).';
G_quat  = G_quat/norm(G_quat);

% Initial Orientation Quaternion:
quat0   = cart_quat_traj_demo(1,2:5).';
quat0   = quat0/norm(quat0);

% Initial Angular Velocity:
omega0  = zeros(3,1);

% DMP Parameters:
alpha_z = 25.0;
beta_z  = alpha_z/4.0;

% Initializations:
quat    = quat0;
omega   = omega0;

quat_err_from_goal_hist     = zeros(traj_length, 3);
quat_hist                   = zeros(traj_length, 4);
quatd_hist                  = zeros(traj_length, 4);
omega_hist                  = zeros(traj_length, 3);
omegad_hist                 = zeros(traj_length, 3);

% using Quaternion Error as the error signal:
for i=1:traj_length
    quat_err_from_goal  = -computeQuatError(G_quat, quat);
    tau_omegad          = alpha_z * eye(3) * ((beta_z * eye(3) * quat_err_from_goal) - omega);
    omegad              = tau_omegad/tau;
    
    omega_cross_prod_mat= computeCrossProductMatrix(omega);
    tau_quatd           = (1/2) * [0, -omega.'; omega, -omega_cross_prod_mat] * quat;
    quatd               = tau_quatd/tau;
    
    % using Log Mapping of Quaternion Difference Integration:
    quat                = integrateQuat( quat, omega, dt, tau );
    omega               = omega + (dt * omegad);
    
    % Trajectory/History Logging:
    quat_err_from_goal_hist(i,:)    = quat_err_from_goal;
    quat_hist(i,:)                  = quat.';
    quatd_hist(i,:)                 = quatd.';
    omega_hist(i,:)                 = omega.';
    omegad_hist(i,:)                = omegad.';
end

% Initializations:
quat    = quat0;
omega   = omega0;

quat_err_from_goal_hist2    = zeros(traj_length, 3);
quat_hist2                  = zeros(traj_length, 4);
quatd_hist2                 = zeros(traj_length, 4);
omega_hist2                 = zeros(traj_length, 3);
omegad_hist2                = zeros(traj_length, 3);

% using (two times) Log Mapping of Quaternion Difference as the error signal:
for i=1:traj_length
    quat_err_from_goal  = computeTwiceLogQuatDifference(G_quat, quat);
    tau_omegad          = alpha_z * eye(3) * ((beta_z * eye(3) * quat_err_from_goal) - omega);
    omegad              = tau_omegad/tau;
    
    omega_cross_prod_mat= computeCrossProductMatrix(omega);
    tau_quatd           = (1/2) * [0, -omega.'; omega, -omega_cross_prod_mat] * quat;
    quatd               = tau_quatd/tau;
    
    % using Log Mapping of Quaternion Difference Integration:
    quat                = integrateQuat( quat, omega, dt, tau );
    omega               = omega + (dt * omegad);
    
    % Trajectory/History Logging:
    quat_err_from_goal_hist2(i,:)   = quat_err_from_goal;
    quat_hist2(i,:)                 = quat.';
    quatd_hist2(i,:)                = quatd.';
    omega_hist2(i,:)                = omega.';
    omegad_hist2(i,:)               = omegad.';
end

quat_diff_hist          = sqrt(sum(((quat_hist2 - quat_hist).^2),2));

% Plotting:

figure;
axis equal;
for d=1:D
    subplot(D,1,d);
        hold on;
            plot(quat_err_from_goal_hist(:,d), 'r');
            plot(quat_err_from_goal_hist2(:,d), 'b');
            title(['Plot of quat\_err\_from\_goal\_hist', num2str(d), ' trajectory/history']);
            if (d==1)
                legend('using Im[Quaternion Error]', 'using Log Mapping of Quaternion Error');
            end
        hold off;
end

figure;
axis equal;
for d=1:D
    subplot(D,1,d);
        hold on;
            plot(omega_hist(:,d), 'r');
            plot(omega_hist2(:,d), 'b');
            title(['Plot of omega', num2str(d), ' trajectory/history']);
            if (d==1)
                legend('using Im[Quaternion Error]', 'using Log Mapping of Quaternion Error');
            end
        hold off;
end

figure;
axis equal;
for d=1:D
    subplot(D,1,d);
        hold on;
            plot(omegad_hist(:,d), 'r');
            plot(omegad_hist2(:,d), 'b');
            title(['Plot of omegad', num2str(d), ' trajectory/history']);
            if (d==1)
                legend('using Im[Quaternion Error]', 'using Log Mapping of Quaternion Error');
            end
        hold off;
end

figure;
axis equal;
for d=1:Dq
    subplot(Dq,1,d);
        hold on;
            plot(quat_hist(:,d), 'r');
            plot(quat_hist2(:,d), 'b');
            title(['Plot of quat', num2str(d), ' trajectory/history']);
            if (d==1)
                legend('using Im[Quaternion Error]', 'using Log Mapping of Quaternion Error');
            end
        hold off;
end

figure;
axis equal;
for d=1:Dq
    subplot(Dq,1,d);
        hold on;
            plot(quatd_hist(:,d), 'r');
            plot(quatd_hist2(:,d), 'b');
            title(['Plot of quatd', num2str(d), ' trajectory/history']);
            if (d==1)
                legend('using Im[Quaternion Error]', 'using Log Mapping of Quaternion Error');
            end
        hold off;
end

figure;
axis equal;
plot(quat_diff_hist);
title('Plot of quat\_diff\_hist');