% Quaternion Dynamic Movement Primitive (DMP) Simulation
% -Given Baseline Quaternion DMP Primitive, 
%  Simulate Fitted Coupling Term (Ct) Unrolling-
% 
% Author : Giovanni Sutanto
% Date   : February 2017

% clear all;
% close all;
clc;

rel_dir_path        = '../../';

addpath([rel_dir_path, '../../utilities/']);
addpath([rel_dir_path, '../../cart_dmp/cart_coord_dmp/']);
addpath([rel_dir_path, '../../cart_dmp/quat_dmp/']);
addpath([rel_dir_path, '../../dmp_multi_dim/']);

D   = 3;
Dq  = 4;

%% Quaternion DMP Parameter Setup
global      dcps;

ID          = 1;
setting_num = 2;
num_trial = 10;
%% Load the Baseline Quaternion DMP Weights and the Ct's to be Unrolled
load([rel_dir_path,'dmp_baseline_params_scraping.mat']);
load([rel_dir_path,'data_demo_scraping.mat']);
load([rel_dir_path,'dataset_Ct_tactile_asm_scraping.mat']);
%1:1:size(dataset_Ct_tactile_asm.sub_X{setting_num},1)
    load([num2str(setting_num),'_', num2str(num_trial), '.mat']);
    % load([num2str(setting_num),'_1.mat']);

    cf = 5;
    Fs = 300;
    [b, a] = butter(2, cf/(Fs/2));
    Ct_target_trial = filtfilt(b, a, Ct_target_trial);
%     Ct_target_trial = dataset_Ct_tactile_asm.sub_Ct_target{1, setting_num}{num_trial,1};
    [~, nmse_1_1]    = computeNMSE(Ct_fit_trial, Ct_target_trial);
    disp(['nmse_', num2str(setting_num) '_', num2str(num_trial), ' = ', num2str(nmse_1_1)]);
    % Ct_fit_trial = filtfilt(b, a, Ct_fit_trial);
    figure; plot(Ct_fit_trial);
    figure; plot(Ct_target_trial);
    figure; 
    subplot(3,1,1);plot(Ct_fit_trial(:,4));hold on; plot(Ct_target_trial(:,4));
    subplot(3,1,2);plot(Ct_fit_trial(:,5));hold on; plot(Ct_target_trial(:,5));
    subplot(3,1,3);plot(Ct_fit_trial(:,6));hold on; plot(Ct_target_trial(:,6));

    %% Generation of Dummy Coupled Demo Trajectories
    dt                  = dmp_baseline_params.Quat{1,1}.dt;
    n_rfs               = dmp_baseline_params.Quat{1,1}.n_rfs;
    c_order             = dmp_baseline_params.Quat{1,1}.c_order;
    fit_mean_Q0_baseline= dmp_baseline_params.Quat{1,1}.fit_mean_Q0;
    fit_mean_QG_baseline= dmp_baseline_params.Quat{1,1}.fit_mean_QG;
    w                   = dmp_baseline_params.Quat{1,1}.w;
    traj_length_unroll  = size(Ct_fit_trial, 1);
    tau_unroll          = dt * (traj_length_unroll - 1);

    N_mode              = 2;

    Q_omega_omegad_traj_unroll  = cell(3, N_mode);

    for mode = 1:N_mode
        if (mode == 1)
            Ct_traj     = Ct_fit_trial(:,4:6).';
        else
    %         Ct_traj     = Ct_target_trial(:,4:6).';
            Ct_traj     = stretchTrajectory(Ct_target_trial(:,4:6).',traj_length_unroll);

    %         Ct_traj     = stretchTrajectory(dataset_Ct_tactile_asm.sub_Ct_target{1,setting_num}{num_trial,1}(:,4:6).',traj_length_unroll);
        end
        dcp_quaternion('init', ID, n_rfs, num2str(ID), c_order);
    %     dcp_quaternion('reset_state', ID, fit_mean_Q0_baseline);
        dcp_quaternion('reset_state', ID, data_demo.coupled{1,setting_num}{num_trial,3}(1,1:4)');
        dcp_quaternion('set_goal', ID, fit_mean_QG_baseline, 1);
        dcps(ID).w      = w;

        QT              = zeros(Dq, traj_length_unroll);
        QdT             = zeros(Dq, traj_length_unroll);
        QddT            = zeros(Dq, traj_length_unroll);
        omegaT          = zeros(D, traj_length_unroll);
        omegadT         = zeros(D, traj_length_unroll);
        FtT             = zeros(D, traj_length_unroll);
        CtT             = zeros(D, traj_length_unroll);

        for i=1:traj_length_unroll
            [Q, Qd, Qdd, omega, omegad, f] = dcp_quaternion('run', ID, tau_unroll, dt, Ct_traj(:,i));

            QT(:,i)         = Q;
            QdT(:,i)        = Qd;
            QddT(:,i)       = Qdd;
            omegaT(:,i)     = omega;
            omegadT(:,i)    = omegad;
            FtT(:,i)        = f;
        end

        Q_omega_omegad_traj_unroll{1, mode} = QT;
        Q_omega_omegad_traj_unroll{2, mode} = omegaT;
        Q_omega_omegad_traj_unroll{3, mode} = omegadT;
    end

    %% Plotting
    figure;
    axis equal;
    for d=1:Dq
        subplot(Dq,1,d);
        hold on;
            for mode=1:N_mode
                plot(Q_omega_omegad_traj_unroll{1, mode}(d,:));
            end
            plot(stretchTrajectory(data_demo.coupled{1,setting_num}{num_trial,3}(:,d)',traj_length_unroll));
            if (d==1)
                title('Quaternion');
            end
%             ylim([-1 1])
            legend('fit', 'target', 'demo');
        hold off;
    end

    figure;
    for d=1:D
        subplot(D,1,d);
        hold on;
            for mode=1:N_mode
                plot(Q_omega_omegad_traj_unroll{2, mode}(d,:));
            end
            plot(stretchTrajectory(data_demo.coupled{1,setting_num}{num_trial,3}(:,d+4)',traj_length_unroll));
            if (d==1)
                title('omega');
            end
            legend('fit', 'target', 'demo');
        hold off;
    end

    figure;
    for d=1:D
        subplot(D,1,d);
        hold on;
            for mode=1:N_mode
                plot(Q_omega_omegad_traj_unroll{3, mode}(d,:));
            end
            plot(stretchTrajectory(data_demo.coupled{1,setting_num}{num_trial,3}(:,d+7)',traj_length_unroll));
            if (d==1)
                title('omegad');
            end
            legend('fit', 'target', 'demo');
        hold off;
    end
