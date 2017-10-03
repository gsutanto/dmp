% Generation of dummy Quaternion trajectory demonstrations.
% 
% Author : Giovanni Sutanto
% Date   : December 2016

clear all;
close all;
clc;

% Seed the randomness:
rng(1234);

addpath('../../../utilities/');
addpath('../');

N_demo      = 20;

D   = 3;
Dq  = 4;

data_path   = '../../../../data/cart_dmp/cart_dmp_wiggling/';

% Load Human Pose Quaternion Trajectory Demonstration:
cart_Q_Qd_Qdd_traj_demo         = dlmread([data_path, 'sample_quat_traj_recorded_demo_wiggling.txt']);
cart_Q_omega_omegad_traj_demo   = dlmread([data_path, 'sample_quat_ABGomega_traj_recorded_demo_wiggling.txt']);
time        = cart_Q_Qd_Qdd_traj_demo(:,1)';
QT          = cart_Q_Qd_Qdd_traj_demo(:,2:5)';
QdT         = cart_Q_Qd_Qdd_traj_demo(:,6:9)';
QddT        = cart_Q_Qd_Qdd_traj_demo(:,10:13)';

% extracting/converting omega and omegad (trajectories) 
% from trajectories of Q, Qd, and Qdd
% [ omegaT, omegadT ] = computeOmegaAndOmegaDotTrajectory( QT, QdT, QddT );
omegaT      = cart_Q_omega_omegad_traj_demo(:,6:8)';
omegadT     = cart_Q_omega_omegad_traj_demo(:,9:11)';

tau         = time(1,end) - time(1,1);
traj_length = size(QT,2);
dt          = tau/(traj_length-1);

% Initial Orientation Quaternion:
Q0          = QT(:,1);
Q0          = Q0/norm(Q0);

% Initial Angular Velocity:
omega0      = omegaT(:,1);

omega_omegad_traj_cell  = cell(2, N_demo);
for i=1:N_demo
    % movement duration (stretching)
    new_tau         = normrnd(tau, 0.2*tau);
    if (new_tau < (0.5 * tau))      % some lower bound clipping
        new_tau     = 0.5 * tau;
    elseif (new_tau > (1.5 * tau))  % some upper bound clipping
        new_tau     = 1.5 * tau;
    end
    new_traj_length = round(new_tau / dt) + 1;
    new_tau         = dt * (new_traj_length - 1);
    
    % omega_dot
    scaled_interpolated_omegadT = (tau/new_tau)^2 * stretchTrajectory(omegadT, new_traj_length);    % don't forget to scale!!!
    max_abs_omegad              = max(abs(scaled_interpolated_omegadT),[],2);
    std_omegad_noise            = 0.1 * mean(max_abs_omegad);
    omega_omegad_traj_cell{2,i} = scaled_interpolated_omegadT;
    omega_omegad_traj_cell{2,i} = omega_omegad_traj_cell{2,i} + ...
                                  (normrnd(0, std_omegad_noise, D, new_traj_length) ...
                                   .* repmat(((1:new_traj_length)-1), D, 1) ...
                                   .* repmat(((1:new_traj_length)-new_traj_length), D, 1) ...
                                   * 1.0/(((1+new_traj_length)/2)^2));
    
    % omega
    omega_omegad_traj_cell{1,i} = zeros(D, new_traj_length);
    omega                       = (tau/new_tau) * omega0;
    for j=1:new_traj_length
        omega_omegad_traj_cell{1,i}(:,j)    = omega;
        omegad                  = omega_omegad_traj_cell{2,i}(:,j);
        omega                   = omega + (dt * omegad);
    end
end

% Plot the original omega_dot trajectory as red curve, and 
% plot the perturbed ones as blue curves 
% (stretched and scaled back by squared tau ratio):
figure;
for d=1:D
    subplot(D,1,d);
    hold on;
        plot(omegadT(d,:), 'r');
        for i=1:N_demo
            perturbed_traj_length   = size(omega_omegad_traj_cell{2,i}(d,:), 2);
            %plot(omega_omegad_traj_cell{2,i}(d,:), 'b-.');
            plot(((perturbed_traj_length-1)/(traj_length-1))^2 * stretchTrajectory(omega_omegad_traj_cell{2,i}(d,:), traj_length), 'b-.');
        end
        title(['original vs (stretched and scaled) perturbed omega\_dot trajectories, dimension ', num2str(d)]);
        legend('original', 'perturbed');
    hold off;
end

% Plot the original omega trajectory as red curve, and 
% plot the perturbed ones as blue curves 
% (stretched and scaled back by tau ratio):
figure;
for d=1:D
    subplot(D,1,d);
    hold on;
        plot(omegaT(d,:), 'r');
        for i=1:N_demo
            perturbed_traj_length   = size(omega_omegad_traj_cell{1,i}, 2);
            %plot(omega_omegad_traj_cell{1,i}(d,:), 'b-.');
            plot(((perturbed_traj_length-1)/(traj_length-1)) * stretchTrajectory(omega_omegad_traj_cell{1,i}(d,:), traj_length), 'b-.');
        end
        title(['original vs (stretched and scaled) perturbed omega trajectories, dimension ', num2str(d)]);
        legend('original', 'perturbed');
    hold off;
end

Q_Qd_Qdd_traj_cell              = cell(3, N_demo);
Q_omega_omegad_traj_cell        = cell(3, N_demo);
for i=1:N_demo
    fprintf('Integrating Quaternion Trajectory # %d/%d\n', i, N_demo);
    
    perturbed_traj_length       = size(omega_omegad_traj_cell{1,i}, 2);
    
    % Q
    Q_Qd_Qdd_traj_cell{1,i}     = zeros(Dq, perturbed_traj_length);
    Q                           = Q0;
    for j=1:perturbed_traj_length
        Q_Qd_Qdd_traj_cell{1,i}(:,j)    = Q;
        omega                   = omega_omegad_traj_cell{1,i}(:,j);
        Q                       = integrateQuat( Q, omega, dt, 1.0 );
    end
    
    % Qd and Qdd
    perturbed_QT                = Q_Qd_Qdd_traj_cell{1,i};
    perturbed_omegaT            = omega_omegad_traj_cell{1,i};
    perturbed_omegadT           = omega_omegad_traj_cell{2,i};
    [ perturbed_QdT, ...
      perturbed_QddT ]  = computeQDotAndQDoubleDotTrajectory( perturbed_QT, ...
                                                              perturbed_omegaT, ...
                                                              perturbed_omegadT );
    Q_Qd_Qdd_traj_cell{2,i}     = perturbed_QdT;
    Q_Qd_Qdd_traj_cell{3,i}     = perturbed_QddT;
    
    Q_omega_omegad_traj_cell{1,i}   = perturbed_QT;
    Q_omega_omegad_traj_cell{2,i}   = perturbed_omegaT;
    Q_omega_omegad_traj_cell{3,i}   = perturbed_omegadT;
end

% Plot the original Q trajectory as red curve, and 
% plot the perturbed ones as blue curves 
% (stretched back to the original traj_length):
figure;
for d=1:Dq
    subplot(Dq,1,d);
    hold on;
        plot(QT(d,:), 'r');
        for i=1:N_demo
            perturbed_traj_length   = size(Q_Qd_Qdd_traj_cell{1,i}, 2);
            %plot(Q_Qd_Qdd_traj_cell{1,i}(d,:), 'b-.');
            plot(stretchTrajectory(Q_Qd_Qdd_traj_cell{1,i}(d,:), traj_length), 'b-.');
        end
        title(['original vs perturbed (stretched) Q trajectories, dimension ', num2str(d)]);
        legend('original', 'perturbed');
    hold off;
end

% Plot the original Qd trajectory as red curve, and 
% plot the perturbed ones as blue curves 
% (stretched and scaled back by tau ratio):
figure;
for d=1:Dq
    subplot(Dq,1,d);
    hold on;
        plot(QdT(d,:), 'r');
        for i=1:N_demo
            perturbed_traj_length   = size(Q_Qd_Qdd_traj_cell{2,i}, 2);
            %plot(Q_Qd_Qdd_traj_cell{2,i}(d,:), 'b-.');
            plot(((perturbed_traj_length-1)/(traj_length-1)) * stretchTrajectory(Q_Qd_Qdd_traj_cell{2,i}(d,:), traj_length), 'b-.');
        end
        title(['original vs perturbed (stretched and scaled) Qd trajectories, dimension ', num2str(d)]);
        legend('original', 'perturbed');
    hold off;
end

% Plot the original Qdd trajectory as red curve, and 
% plot the perturbed ones as blue curves 
% (stretched and scaled back by squared tau ratio):
figure;
for d=1:Dq
    subplot(Dq,1,d);
    hold on;
        plot(QddT(d,:), 'r');
        for i=1:N_demo
            perturbed_traj_length   = size(Q_Qd_Qdd_traj_cell{3,i}, 2);
            %plot(Q_Qd_Qdd_traj_cell{3,i}(d,:), 'b-.');
            plot(((perturbed_traj_length-1)/(traj_length-1))^2 * stretchTrajectory(Q_Qd_Qdd_traj_cell{3,i}(d,:), traj_length), 'b-.');
        end
        title(['original vs perturbed (stretched and scaled) Qdd trajectories, dimension ', num2str(d)]);
        legend('original', 'perturbed');
    hold off;
end

fprintf('Writing dummy Quaternion trajectories to files ...\n');
if (exist([data_path, '/dummy_samples_quat_traj_demo_wiggling/'], 'dir') ~= 7) % if directory NOT exist
    mkdir(data_path, 'dummy_samples_quat_traj_demo_wiggling'); % create directory
end
output_dummy_data_path  = [data_path, '/dummy_samples_quat_traj_demo_wiggling/'];
for i=1:N_demo
%     dummy_QT            = Q_Qd_Qdd_traj_cell{1,i};
%     dummy_QdT           = Q_Qd_Qdd_traj_cell{2,i};
%     dummy_QddT          = Q_Qd_Qdd_traj_cell{3,i};
    dummy_QT            = Q_omega_omegad_traj_cell{1,i};
    dummy_omegaT        = Q_omega_omegad_traj_cell{2,i};
    dummy_omegadT       = Q_omega_omegad_traj_cell{3,i};
    dummy_traj_length   = size(dummy_QT, 2);
    dummy_tau           = dt * (dummy_traj_length - 1);
    dummy_time          = 0.0:dt:dummy_tau;
%     dummy_Quat_traj     = [dummy_time', dummy_QT', dummy_QdT', dummy_QddT'];
    dummy_Quat_traj     = [dummy_time', dummy_QT', dummy_omegaT', dummy_omegadT'];
    dlmwrite([output_dummy_data_path, '/', num2str(i), '.txt'], dummy_Quat_traj, 'delimiter', ' ');
end
fprintf('Done.\n');