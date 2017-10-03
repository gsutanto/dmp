% Quaternion Dynamic Movement Primitive (DMP) Simulation
% -Given Baseline Quaternion DMP Primitive, 
%  Simulate Coupling Term (Ct) Extraction from Demo Trajectories-
% 
% Author : Giovanni Sutanto
% Date   : January 2017

clear all;
close all;
clc;

% Seed the randomness:
rng(1234);

addpath('../../../utilities/');
addpath('../');

D   = 3;
Dq  = 4;

data_path           = '../../../../data/cart_dmp/cart_dmp_wiggling/';

%% Quaternion DMP Parameter Setup
global  dcps;

n_rfs   = 25;
c_order = 1;
ID      = 1;

%% Load the Baseline Quaternion DMP Weights
w       = dlmread([data_path, 'weights_quat_dmp_wiggling.txt']);
assert(size(w, 1) == n_rfs, 'Weight matrix size mis-match!');

% Using Unrolling Contexts (Start, Goal, Tau) of Quaternion Baseline
% Trajectory Demonstration:
cart_Q_traj_demo_baseline   = dlmread([data_path, 'sample_quat_traj_recorded_demo_wiggling.txt']);
time_demo_baseline          = cart_Q_traj_demo_baseline(:,1)';
QT_demo_baseline            = cart_Q_traj_demo_baseline(:,2:5)';
QdT_demo_baseline           = cart_Q_traj_demo_baseline(:,6:9)';
QddT_demo_baseline          = cart_Q_traj_demo_baseline(:,10:13)';
[ omegaT_demo_baseline, ...
  omegadT_demo_baseline ]   = computeOmegaAndOmegaDotTrajectory( QT_demo_baseline, ...
                                                                 QdT_demo_baseline, ...
                                                                 QddT_demo_baseline );

Q_omega_omegad_traj_demo_baseline       = cell(3, 1);
Q_omega_omegad_traj_demo_baseline{1,1}  = QT_demo_baseline;
Q_omega_omegad_traj_demo_baseline{2,1}  = omegaT_demo_baseline;
Q_omega_omegad_traj_demo_baseline{3,1}  = omegadT_demo_baseline;

tau_demo_baseline           = time_demo_baseline(1,end) - time_demo_baseline(1,1);
traj_length_demo_baseline   = size(QT_demo_baseline,2);
dt                          = tau_demo_baseline/(traj_length_demo_baseline-1);

% Initial Orientation Quaternion:
Q0_demo_baseline            = QT_demo_baseline(:,1);
Q0_demo_baseline            = Q0_demo_baseline/norm(Q0_demo_baseline);

% Goal Orientation Quaternion:
QG_demo_baseline            = QT_demo_baseline(:,end);
QG_demo_baseline            = QG_demo_baseline/norm(QG_demo_baseline);

%% Generation of Dummy Coupled Demo Trajectories
N_demo              = 5;
Ct_const_multiplier = 50.0;

Q_Qd_Qdd_traj_demo_set          = cell(3, N_demo);
Q_omega_omegad_traj_demo_set    = cell(3, N_demo);
Ft_traj_demo_set                = cell(1, N_demo);
Ct_traj_demo_set                = cell(1, N_demo);

for d_idx = 1:N_demo
    fprintf('Generating (Coupled) Quaternion Demo Trajectory # %d/%d\n', ...
            d_idx, N_demo);
    
    % movement duration (stretching)
    new_tau         = normrnd(tau_demo_baseline, 0.2*tau_demo_baseline);
    if (new_tau < (0.5 * tau_demo_baseline))    % some lower bound clipping
        new_tau     = 0.5 * tau_demo_baseline;
    elseif (new_tau > (1.5 * tau_demo_baseline))% some upper bound clipping
        new_tau     = 1.5 * tau_demo_baseline;
    end
    new_traj_length = round(new_tau / dt) + 1;
    new_tau         = dt * (new_traj_length - 1);
    
    QT              = zeros(Dq, new_traj_length);
    QdT             = zeros(Dq, new_traj_length);
    QddT            = zeros(Dq, new_traj_length);
    omegaT          = zeros(D, new_traj_length);
    omegadT         = zeros(D, new_traj_length);
    FtT             = zeros(D, new_traj_length);
    CtT             = zeros(D, new_traj_length);
    Ct              = zeros(D, 1);
    
    new_omega0      = normrnd(0, 5.0, [3 1]);   % randomized initial angular velocity
    new_omegad0     = normrnd(0, 1.0, [3 1]);   % randomized initial angular acceleration

    dcp_quaternion('init', ID, n_rfs, num2str(ID), c_order);
    dcp_quaternion('reset_state', ID, Q0_demo_baseline, new_omega0, new_omegad0, new_tau);
    dcp_quaternion('set_goal', ID, QG_demo_baseline, 1);
    dcps(ID).w      = w;

    for i=1:new_traj_length
        t               = (i-1) * dt;
        for d = 1:D
            Ct(d,1)     = sin(d * 2*pi*t/new_tau);
            
            % add some Gaussian noise:
            if ((i ~= 1) && (i ~= new_traj_length))
                Ct(d,1) = Ct(d,1) + normrnd(0, 0.2);
            end
        end
        Ct              = Ct_const_multiplier * Ct;
        
        [Q, Qd, Qdd, omega, omegad, f] = dcp_quaternion('run', ID, new_tau, dt, Ct);

        QT(:,i)         = Q;
        QdT(:,i)        = Qd;
        QddT(:,i)       = Qdd;
        omegaT(:,i)     = omega;
        omegadT(:,i)    = omegad;
        FtT(:,i)        = f;
        CtT(:,i)        = Ct;
    end
    
    Q_Qd_Qdd_traj_demo_set{1, d_idx}        = QT;
    Q_Qd_Qdd_traj_demo_set{2, d_idx}        = QdT;
    Q_Qd_Qdd_traj_demo_set{3, d_idx}        = QddT;
    
    Q_omega_omegad_traj_demo_set{1, d_idx}  = QT;
    Q_omega_omegad_traj_demo_set{2, d_idx}  = omegaT;
    Q_omega_omegad_traj_demo_set{3, d_idx}  = omegadT;
    
    Ft_traj_demo_set{1, d_idx}              = FtT;
    Ct_traj_demo_set{1, d_idx}              = CtT;
end

%% Extract the Coupling Term Trajectories from the Demonstrations
Quat_dmp_baseline_params.n_rfs          = n_rfs;
Quat_dmp_baseline_params.c_order        = c_order;
Quat_dmp_baseline_params.dt             = dt;
Quat_dmp_baseline_params.w              = w;
Quat_dmp_baseline_params.fit_mean_Q0    = Q0_demo_baseline;
Quat_dmp_baseline_params.fit_mean_QG    = QG_demo_baseline;

% [Ct_extracted, Ct_set_extracted, Ft_extracted] = computeQuatDMPCtTarget(Q_Qd_Qdd_traj_demo_set, Quat_dmp_baseline_params);
[Ct_extracted, Ct_set_extracted, Ft_extracted] = computeQuatDMPCtTarget(Q_omega_omegad_traj_demo_set, Quat_dmp_baseline_params);

%% Evaluation: Measure Difference between Extracted Ct and Gold/Ground Truth Ct
Ct_gold     = cell2mat(Ct_traj_demo_set)';
rmse_Ct     = sqrt(mean(mean((Ct_gold-Ct_extracted).^2)));
fprintf('RMSE Ct = %f\n', rmse_Ct);

Ft_gold     = cell2mat(Ft_traj_demo_set)';
rmse_Ft     = sqrt(mean(mean((Ft_gold-Ft_extracted).^2)));
fprintf('RMSE Ft = %f\n', rmse_Ft);

%% Plotting

% Plot the coupled demonstrations as blue curves vs
% the baseline demonstrations as red curves
% (stretched to have equal lengths):
for d_idx=1:N_demo
    for order=1:3
        if (order==1)
            state_var   = 'Quaternion';
            dimension   = Dq;
        elseif (order==2)
            state_var   = 'omega';
            dimension   = D;
        elseif (order==3)
            state_var   = 'omegad';
            dimension   = D;
        end
        
        figure;
        for d=1:dimension
            subplot(dimension,1,d);
            hold on;
                plot(Q_omega_omegad_traj_demo_baseline{order,1}(d,:), 'r');
                plot(stretchTrajectory(Q_omega_omegad_traj_demo_set{order,d_idx}(d,:), traj_length_demo_baseline), 'b');
                title(['baseline vs coupled ', state_var, ' demo #', num2str(d_idx), ', dimension ', num2str(d)]);
                legend('baseline', 'coupled');
            hold off;
        end
    end
end

% Plot the all Ct ground-truth trajectories as blue curves vs
% extracted Ct trajectories as green curves
% (stretched to have equal lengths):
figure;
for d=1:D
    subplot(D,1,d);
    hold on;
        for d_idx=1:N_demo
            plot(stretchTrajectory(Ct_traj_demo_set{1,d_idx}(d,:), traj_length_demo_baseline), 'b-.');
            plot(stretchTrajectory(Ct_set_extracted{d_idx,1}(:,d)', traj_length_demo_baseline), 'g*');
        end
        title(['(stretched) ground-truth vs extracted Coupling Term trajectories, dimension ', num2str(d)]);
        legend('ground-truth', 'extracted');
    hold off;
end

% Plot the per-demonstration Ct ground-truth trajectory as blue curve vs
% extracted Ct trajectory as green curve:
for d_idx=1:N_demo
    figure;
    for d=1:D
        subplot(D,1,d);
        hold on;
            plot(Ct_traj_demo_set{1,d_idx}(d,:), 'b-.');
            plot(Ct_set_extracted{d_idx,1}(:,d)', 'g*');
            title(['ground-truth vs extracted Coupling Term trajectory #', num2str(d_idx), ', dimension ', num2str(d)]);
            legend('ground-truth', 'extracted');
        hold off;
    end
end