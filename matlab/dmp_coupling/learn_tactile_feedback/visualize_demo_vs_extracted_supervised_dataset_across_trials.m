% Author        : Giovanni Sutanto
% Date          : March 2017
% Description   :
%   Visualize the trajectories, as well as extracted supervised dataset
%   (target X (sensor input) - Ct/coupling term (output) pairs) 
%   overlayed together across trials in each setting to observe inconsistency.

clear all;
close all;
clc;

is_visualizing_Ctt_X                                                = 0;
is_visualizing_Ctt_traj_CartCoord_Quat_all_trials_per_setting       = 1;
    is_visualizing_Ctt_traj_CartCoord_all_trials_per_setting        = 0;
    is_visualizing_Ctt_traj_Quat_all_trials_per_setting             = 1;
is_covisualizing_traj                                               = 0;
is_visualizing_traj_only                                            = 0;
is_visualizing_traj_vs_smoothed_traj                                = 0;
is_visualizing_Ctt_traj_Quat_per_trial                              = 0;
is_visualizing_smoothed_coupled_vs_baseline_traj                    = 0;

N_sensor_dimension  = 38;
max_num_demo_display= 30;

rel_dir_path        = './';

addpath([rel_dir_path, '../../utilities/']);

task_type   = 'scraping';
load(['data_demo_',task_type,'.mat']);
load(['smoothed_data_demo_',task_type,'.mat']);
load(['dataset_Ct_tactile_asm_',task_type,'_augmented.mat']);

N_correction_types_per_plot     = 2;    % 1st is for position correction, 2nd is for orientation correction
N_modalities_per_plot           = 3;    % # of sensory modalities per plot
N_mode                          = 2;    % 1st: against modalities; 2nd: against raw coupled trajectories
N_subplot_per_plot              = 3 + N_modalities_per_plot;    % 3 top subplots are for the target coupling term, the rest are for the modalities
traj_length_disp                = 1500;

% Low-Pass Filter Setup:
% fc      = 5.0;                  % cutoff frequency   = 5 Hz
% fs      = 300.0;                % sampling frequency = 300 Hz
% N_order = 2;
% [b, a] 	= butter(N_order, fc/(fs/2)); % low-pass N_order-th-order Butterworth filter

[ linespec_codes ]  = generateLinespecCodes();

N_primitive         = size(dataset_Ct_tactile_asm.sub_Ct_target, 1);
N_settings          = size(dataset_Ct_tactile_asm.sub_Ct_target, 2);

% for np = 1:1:N_primitive
for np = 2
%     for ns = 1:1:N_settings
    for ns = 1
        N_demos = min(size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}, 1), max_num_demo_display);

        Ctt_N_dim                   = size(dataset_Ct_tactile_asm.sub_Ct_target{1,1}{1,1}, 2);
        m_N_dim                     = size(dataset_Ct_tactile_asm.sub_X{1,1}{1,1}, 2);

        stretched_Ctt_traj_cell     = cell(Ctt_N_dim, N_demos);
        stretched_m_traj_cell       = cell(  m_N_dim, N_demos);

        for ndm = 1:1:N_demos
            for Ctt_dim = 1:1:Ctt_N_dim
                Ct_target_demo      = dataset_Ct_tactile_asm.sub_Ct_target{np,ns}{dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}(ndm),1}(:,Ctt_dim);
                stretched_Ctt_traj  = stretchTrajectory( Ct_target_demo.', traj_length_disp );
    %             stretched_Ctt_traj  = stretchTrajectory( filtfilt(b, a, Ct_target_demo.'), traj_length_disp );
                stretched_Ctt_traj_cell{Ctt_dim, ndm}   = stretched_Ctt_traj.';
            end

            for m_dim = 1:1:m_N_dim
                modality_demo       = dataset_Ct_tactile_asm.sub_X{np,ns}{dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}(ndm),1}(:,m_dim);
                stretched_m_traj    = stretchTrajectory( modality_demo.', traj_length_disp );
    %             stretched_m_traj    = stretchTrajectory( filtfilt(b, a, modality_demo.'), traj_length_disp );
                stretched_m_traj_cell{m_dim, ndm}       = stretched_m_traj.';
            end
        end

        if (is_visualizing_Ctt_X)
            close all;
            for nc = 1:N_correction_types_per_plot
                for npl=1:ceil(size(dataset_Ct_tactile_asm.sub_X{np,ns}{1,1}, 2)/N_modalities_per_plot)
                    figure;
                    axis equal;
                    for nsp=1:N_subplot_per_plot
                        subplot(N_subplot_per_plot,1,nsp)
                        hold on;
                        if (nsp <= 3)
                            Ctt_dim = ((nc-1) * 3) + nsp;   % Ct target dimension
                            for ndm = 1:1:N_demos
                                plot(stretched_Ctt_traj_cell{Ctt_dim, ndm}, linespec_codes{1,ndm});
                            end
                            if (nc == 1)
                                special_remark      = ['Position    #', num2str(Ctt_dim)];
                            else
                                special_remark      = ['Orientation #', num2str(Ctt_dim-3)];
                            end
                            title(['Ct target dimension #',num2str(Ctt_dim),...
                                   ' (',special_remark,')',', Primitive #',num2str(np),', Setting #',num2str(ns)]);
                        else
                            m_dim   = ((npl-1) * N_modalities_per_plot) + (nsp-3);  % modality dimension
                            for ndm = 1:1:N_demos
                                plot(stretched_m_traj_cell{m_dim, ndm}, linespec_codes{1,ndm});
                            end
                            if (m_dim <= 38)
                                special_remark      = ['Finger Electrode #', num2str(m_dim)];
                            else
                                special_remark      = ['Joint Position   #', num2str(m_dim-38)];
                            end
                            title(['modality error signal dimension #',num2str(m_dim),...
                                   ' (',special_remark,')',', Primitive #',num2str(np),', Setting #',num2str(ns)]);
                        end
                        hold off;
                    end
                end
            end
            keyboard;
        end

        if (is_visualizing_Ctt_traj_CartCoord_Quat_all_trials_per_setting)
            if (is_visualizing_Ctt_traj_CartCoord_all_trials_per_setting)
                figure;
                axis equal;
                if (is_covisualizing_traj)
                    Nsp = 6;
                else
                    Nsp = 3;
                end
                for nsp=1:Nsp
                    subplot(Nsp,1,nsp)
                    hold on;
                    if (nsp <= 3)
                        Ctt_dim = nsp;      % Ct target (CartCoord) dimension
                        for ndm = 1:1:N_demos
                            plot(stretched_Ctt_traj_cell{Ctt_dim, ndm}, linespec_codes{1,ndm});
                        end
                        special_remark          = ['Cartesian Coordinate #', num2str(Ctt_dim)];
                        title(['Ct target dimension #',num2str(Ctt_dim),...
                               ' (',special_remark,')',', Primitive #',num2str(np),', Setting #',num2str(ns)]);
                    else
                        CartCoord_dim   = (nsp-3);  % modality dimension
                        for ndm = 1:1:N_demos
                            CartCoord_demo      = data_demo.coupled{np,ns}{dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}(ndm),2}(:,CartCoord_dim);
                            stretched_CC_traj   = stretchTrajectory( CartCoord_demo.', traj_length_disp );
                            plot(stretched_CC_traj, linespec_codes{1,ndm});
                        end
                        title(['Cartesian Coordinate dimension #',num2str(CartCoord_dim),...
                               ', Primitive #',num2str(np),', Setting #',num2str(ns)]);
                    end
                    hold off;
                end
            end

            if (is_visualizing_Ctt_traj_Quat_all_trials_per_setting)
                figure;
                axis equal;
                if (is_covisualizing_traj)
                    Nsp = 7;
                else
                    Nsp = 3;
                end
                for nsp=1:Nsp
                    subplot(Nsp,1,nsp)
                    hold on;
                    if (nsp <= 3)
                        Ctt_dim = 3 + nsp;   % Ct target (Quat) dimension
                        for ndm = 1:1:N_demos
                            plot(stretched_Ctt_traj_cell{Ctt_dim, ndm}, linespec_codes{1,ndm});
                        end
                        special_remark          = ['Orientation #', num2str(Ctt_dim-3)];
                        title(['Ct target dimension #',num2str(Ctt_dim),...
                               ' (',special_remark,')',', Primitive #',num2str(np),', Setting #',num2str(ns)]);
                    else
                        Quat_dim    = (nsp-3);  % modality dimension
                        for ndm = 1:1:N_demos
                            Quat_demo   = data_demo.coupled{np,ns}{dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}(ndm),3}(:,Quat_dim);
                            stretched_Q_traj    = stretchTrajectory( Quat_demo.', traj_length_disp );
                            plot(stretched_Q_traj, linespec_codes{1,ndm});
                        end
                        title(['Quaternion dimension #',num2str(Quat_dim),...
                               ', Primitive #',num2str(np),', Setting #',num2str(ns)]);
                    end
                    hold off;
                end
%                 keyboard;
            end
        end

        if (is_visualizing_traj_only)
            figure;
            axis equal;
            Nsp = 3;
            for nsp=1:Nsp
                subplot(Nsp,1,nsp)
                hold on;
                    CartCoord_dim   = nsp;  % modality dimension
                    for ndm = 1:1:N_demos
                        CartCoord_demo      = data_demo.coupled{np,ns}{dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}(ndm),2}(:,CartCoord_dim);
                        stretched_CC_traj   = stretchTrajectory( CartCoord_demo.', traj_length_disp );
                        plot(stretched_CC_traj, linespec_codes{1,ndm});
                    end
                    title(['Cartesian Coordinate dimension #',num2str(CartCoord_dim),...
                           ', Primitive #',num2str(np),', Setting #',num2str(ns)]);
                hold off;
            end

            figure;
            axis equal;
            Nsp = 4;
            for nsp=1:Nsp
                subplot(Nsp,1,nsp)
                hold on;
                    Quat_dim    = nsp;  % modality dimension
                    for ndm = 1:1:N_demos
                        Quat_demo           = data_demo.coupled{np,ns}{dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}(ndm),3}(:,Quat_dim);
                        stretched_Q_traj    = stretchTrajectory( Quat_demo.', traj_length_disp );
                        plot(stretched_Q_traj, linespec_codes{1,ndm});
                    end
                    title(['Quaternion dimension #',num2str(Quat_dim),...
                           ', Primitive #',num2str(np),', Setting #',num2str(ns)]);
                hold off;
            end
            keyboard;
        end

        if (is_visualizing_traj_vs_smoothed_traj)
            figure;
            axis equal;
            Nsp = 3;
            for nsp=1:Nsp
                subplot(Nsp,1,nsp)
                hold on;
                    CartCoord_dim   = nsp;  % modality dimension
                    for ndm = 1:1:N_demos
                        % original trajectory
                        CartCoord_demo      = data_demo.coupled{np,ns}{dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}(ndm),2}(:,CartCoord_dim);
                        stretched_CC_traj   = stretchTrajectory( CartCoord_demo.', traj_length_disp );
                        plot(stretched_CC_traj, 'b');

                        % smoothed trajectory
                        CartCoord_demo      = smoothed_data_demo.coupled{np,ns}{dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}(ndm),2}(:,CartCoord_dim);
                        stretched_CC_traj   = stretchTrajectory( CartCoord_demo.', traj_length_disp );
                        plot(stretched_CC_traj, 'g');
                    end
                    legend('original', 'smoothed');
                    title(['Cartesian Coordinate dimension #',num2str(CartCoord_dim),...
                           ', Primitive #',num2str(np),', Setting #',num2str(ns)]);
                hold off;
            end

            figure;
            axis equal;
            Nsp = 4;
            for nsp=1:Nsp
                subplot(Nsp,1,nsp)
                hold on;
                    Quat_dim    = nsp;  % modality dimension
                    for ndm = 1:1:N_demos
                        % original trajectory
                        Quat_demo           = data_demo.coupled{np,ns}{dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}(ndm),3}(:,Quat_dim);
                        stretched_Q_traj    = stretchTrajectory( Quat_demo.', traj_length_disp );
                        plot(stretched_Q_traj, 'b');

                        % smoothed trajectory
                        Quat_demo           = smoothed_data_demo.coupled{np,ns}{dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}(ndm),3}(:,Quat_dim);
                        stretched_Q_traj    = stretchTrajectory( Quat_demo.', traj_length_disp );
                        plot(stretched_Q_traj, 'g');
                    end
                    legend('original', 'smoothed');
                    title(['Quaternion dimension #',num2str(Quat_dim),...
                           ', Primitive #',num2str(np),', Setting #',num2str(ns)]);
                hold off;
            end
            keyboard;
        end

        if (is_visualizing_Ctt_traj_Quat_per_trial)
            for ndm = 1:1:N_demos
                figure;
                axis equal;
                for nsp=1:7
                    subplot(7,1,nsp)
                    hold on;
                    if (nsp <= 3)
                        Ctt_dim = 3 + nsp;   % Ct target dimension
                        plot(stretched_Ctt_traj_cell{Ctt_dim, ndm}, linespec_codes{1,ndm});
                        special_remark          = ['Orientation #', num2str(Ctt_dim-3)];
                        title(['Ct target dimension #',num2str(Ctt_dim),...
                               ' (',special_remark,')',', Primitive #',num2str(np),', Setting #',num2str(ns),', Demo #',num2str(ndm)]);
                    else
                        Quat_dim    = (nsp-3);  % modality dimension
                        Quat_c_demo = data_demo.coupled{np,ns}{dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}(ndm),3}(:,Quat_dim);   % coupled trajectory
                        stretched_Qc_traj   = stretchTrajectory( Quat_c_demo.', traj_length_disp );
                        plot(stretched_Qc_traj, linespec_codes{1,ndm});
                        Quat_b_demo = data_demo.baseline{1,1}{5,3}(:,Quat_dim);     % baseline trajectory
                        stretched_Qb_traj   = stretchTrajectory( Quat_b_demo.', traj_length_disp );
                        plot(stretched_Qb_traj, 'r', 'Linewidth', 3);
                        title(['Quaternion dimension #',num2str(Quat_dim),...
                               ', Primitive #',num2str(np),', Setting #',num2str(ns),', Demo #',num2str(ndm)]);
                    end
                    hold off;
                end
            end
        end

        if (is_visualizing_smoothed_coupled_vs_baseline_traj)
            figure;
            axis equal;
            Nsp = 3;
            for nsp=1:Nsp
                subplot(Nsp,1,nsp)
                hold on;
                    CartCoord_dim   = nsp;  % modality dimension
                    
                    N_demos_baseline= length(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,N_settings});
                    for ndmb = 1:1:N_demos_baseline
                        % smoothed baseline trajectory
                        CartCoord_demo      = smoothed_data_demo.baseline{np,1}{dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,N_settings}(ndmb),2}(:,CartCoord_dim);
                        stretched_CC_traj   = stretchTrajectory( CartCoord_demo.', traj_length_disp );
                        pbCC                = plot(stretched_CC_traj, 'r');
                    end
                    
                    N_demos_coupled = length(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns});
                    for ndmc = 1:1:N_demos_coupled
                        % smoothed coupled trajectory
                        CartCoord_demo      = smoothed_data_demo.coupled{np,ns}{dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}(ndmc),2}(:,CartCoord_dim);
                        stretched_CC_traj   = stretchTrajectory( CartCoord_demo.', traj_length_disp );
                        pcCC                = plot(stretched_CC_traj, 'g');
                    end
                    
                    legend([pbCC, pcCC], 'baseline', 'coupled');
                    title(['Cartesian Coordinate dimension #',num2str(CartCoord_dim),...
                           ', Primitive #',num2str(np),', Setting #',num2str(ns)]);
                hold off;
            end

            figure;
            axis equal;
            Nsp = 4;
            for nsp=1:Nsp
                subplot(Nsp,1,nsp)
                hold on;
                    Quat_dim    = nsp;  % modality dimension
                    
                    N_demos_baseline= length(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,N_settings});
                    for ndmb = 1:1:N_demos_baseline
                        % smoothed baseline trajectory
                        Quat_demo           = smoothed_data_demo.baseline{np,1}{dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,N_settings}(ndmb),3}(:,Quat_dim);
                        stretched_Q_traj    = stretchTrajectory( Quat_demo.', traj_length_disp );
                        pbQ                 = plot(stretched_Q_traj, 'r');
                    end
                    
                    N_demos_coupled = length(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns});
                    for ndmc = 1:1:N_demos_coupled
                        % smoothed coupled trajectory
                        Quat_demo           = smoothed_data_demo.coupled{np,ns}{dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}(ndmc),3}(:,Quat_dim);
                        stretched_Q_traj    = stretchTrajectory( Quat_demo.', traj_length_disp );
                        pcQ                 = plot(stretched_Q_traj, 'g');
                    end
                    
                    legend([pbQ, pcQ], 'baseline', 'coupled');
                    title(['Quaternion dimension #',num2str(Quat_dim),...
                           ', Primitive #',num2str(np),', Setting #',num2str(ns)]);
                hold off;
            end
            keyboard;
        end
    end
end