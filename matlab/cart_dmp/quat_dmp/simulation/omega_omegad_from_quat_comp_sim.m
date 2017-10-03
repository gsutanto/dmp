% Computation of Omega (Angular Velocity) and Omegad (Angular Acceleration)
% from Quaternion (q), Quaternion Derivative (qd), and 
% Quaternion Double Derivative (qdd) Simulation
% 
% Author : Giovanni Sutanto
% Date   : October 19, 2016

clear all;
close all;
clc;

addpath('../../../utilities/');
addpath('../');

D   = 3;
Dq  = 4;

Wn 	= 0.04;

cart_quat_traj_demo     = dlmread('../../../../data/cart_dmp/cart_dmp_wiggling/sample_quat_traj_recorded_demo_wiggling.txt');

tau         = cart_quat_traj_demo(end,1) - cart_quat_traj_demo(1,1);
traj_length = size(cart_quat_traj_demo,1);
dt          = tau/(traj_length-1);

omega_traj              = zeros(traj_length, 3);
omegad_traj             = zeros(traj_length, 3);
omega_traj_filtered     = zeros(traj_length, 3);
omegad_traj_filtered    = zeros(traj_length, 3);
for i=1:traj_length
    quat        = cart_quat_traj_demo(i,2:5).';
    quatd       = cart_quat_traj_demo(i,6:9).';
    quatdd      = cart_quat_traj_demo(i,10:13).';
    
    quat_conj   = computeQuatConjugate(quat);
    
    quat_omega  = 2*tau*computeQuatProduct(quatd, quat_conj);
    quat_omegad = 2*tau*computeQuatProduct((quatdd - computeQuatProduct(quatd, computeQuatProduct(quat_conj, quatd))), quat_conj);
    
    if(abs(quat_omega(1,1)) > 0.001)
        quat_omega
    end
    omega               = quat_omega(2:4,1);
    omega_traj(i,:)     = omega.';
    
    if(abs(quat_omegad(1,1)) > 0.001)
        quat_omegad
    end
    omegad              = quat_omegad(2:4,1);
    omegad_traj(i,:)    = omegad.';
end

% Use the following if filtering is needed:
[filt_num, filt_den]    = butter(2, Wn);

for d=1:D
    omega_traj_filtered(:,d)    = filtfilt(filt_num, filt_den, omega_traj(:,d));
    omegad_traj_filtered(:,d)   = diffnc(omega_traj_filtered(:,d),dt);
end

figure;
axis equal;
for d=1:Dq
    subplot(Dq,1,d);
        hold on;
            plot(cart_quat_traj_demo(:,d+1));
            title(['Plot of q', num2str(d-1), ' trajectory']);
        hold off;
end

figure;
axis equal;
for d=1:D
    subplot(D,1,d);
        hold on;
            plot(omega_traj(:,d), 'b');
%             plot(omega_traj_filtered(:,d), 'c');
            title(['Plot of omega', num2str(d), ' trajectory']);
        hold off;
end

figure;
axis equal;
for d=1:D
    subplot(D,1,d);
        hold on;
            plot(omegad_traj(:,d), 'g');
%             plot(omegad_traj_filtered(:,d), 'b');
            title(['Plot of omegad', num2str(d), ' trajectory']);
        hold off;
end