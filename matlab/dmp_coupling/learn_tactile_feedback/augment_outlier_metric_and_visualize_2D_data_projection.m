% Author        : Giovanni Sutanto
% Date          : March 2017
% Description   :
%   Perform outlier metric computation and ranking of 
%   each individual sample per setting, 
%   based on one of the following criteria:
%   [1] non-linear dimensionality reduction 
%       for each setting of the dataset into 2D, such that it becomes clear 
%       which data/demonstration is most likely to be an outlier, 
%       and which one is not.
%       Also visualize the 2D representation of the dataset for inspection.
%   [2] based on primitive 2's orientation coupling term in 2nd dimension
%       i.e. rotation w.r.t. y-axis (beta); we manually identify whether 
%       the coupling term should be positive or negative in this dimension.

clear all;
close all;
clc;

outlier_metric_criteria = 1;    % based on non-linear dimensionality reduction
% outlier_metric_criteria = 2;    % based on primitive 2's orientation coupling term in 2nd dimension (beta)

% for (outlier_metric_criteria == 1):
is_visualizing      = 0;
distance_metric_mode= 1;    % using Ct_target component only for the distance metric
% distance_metric_mode= 2;  % using X component only for the distance metric
% distance_metric_mode= 3;  % using both Ct_target and X components in the distance metric

% for (outlier_metric_criteria == 2):
% p2_and_p3_orientation_ct_beta_polarity  = [+1; ...  % setting  1
%                                        +1; ...  % setting  2
%                                        +1; ...  % setting  3
%                                        +1; ...  % setting  4
%                                        +1; ...  % setting  5
%                                        +1; ...  % setting  6
%                                        +1; ...  % setting  7
%                                        -1; ...  % setting  8
%                                        -1; ...  % setting  9
%                                        -1; ...  % setting 10
%                                        -1; ...  % setting 11
%                                        -1; ...  % setting 12
%                                        -1; ...  % setting 13
%                                        -1; ...  % setting 14
%                                        -1; ...  % setting 15
%                                        -1; ...  % setting 16
%                                        0;  ...  % setting 17
%                                        ];
% for scraping_wo_tool:
% p2_and_p3_orientation_ct_beta_polarity  = [0; ...  % setting  1
%                                            0; ...  % setting  2
%                                            0; ...  % setting  3
%                                            0; ...  % setting  4
%                                            0; ...  % setting  5
%                                            0; ...  % setting  6
%                                            0; ...  % setting  7
%                                            0; ...  % setting  8
%                                            0; ...  % setting  9
%                                           ];
% for scraping_w_tool:
p2_and_p3_orientation_ct_beta_polarity  = [0; ...  % setting  1
                                           0; ...  % setting  2
                                           0; ...  % setting  3
                                           0; ...  % setting  4
                                           0; ...  % setting  5
%                                            0; ...  % setting  6
%                                            0; ...  % setting  7
%                                            0; ...  % setting  8
%                                            0; ...  % setting  9
                                          ];
abs_max_prim2_orientation_ct_beta_fraction_retain   = 0.75;

% only considers tactile electrodes' signals for similarity metric:
N_sensor_dimension  = 38;

rel_dir_path        = './';

addpath([rel_dir_path, '../../utilities/']);

task_type           = 'scraping';
load(['data_demo_',task_type,'.mat']);
load(['dataset_Ct_tactile_asm_',task_type,'_augmented.mat']);

traj_length_disp                = 1500;

[ linespec_codes ]  = generateLinespecCodes();

N_primitive         = size(dataset_Ct_tactile_asm.sub_Ct_target, 1);
N_settings          = size(dataset_Ct_tactile_asm.sub_Ct_target, 2);

assert(N_settings == size(p2_and_p3_orientation_ct_beta_polarity, 1), 'Incorrect manual specification of p2_and_p3_orientation_ct_beta_polarity!');

%% Add Excluded Trials/Demos Indices

dataset_Ct_tactile_asm.exclude          = cell(N_primitive, N_settings);
dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion   = cell(N_primitive, N_settings);

%% Add Manual Exclusion of Trials (Manually Picked from Observation 
%  from Execution of visualize_demo_vs_extracted_supervised_dataset_across_trials.m Script, 
%  e.g. due to Inconsistencies, etc.)

for np=1:1:N_primitive
    dataset_Ct_tactile_asm.exclude{np, 1}   = [];
    dataset_Ct_tactile_asm.exclude{np, 2}   = [];
    dataset_Ct_tactile_asm.exclude{np, 3}   = [];
    dataset_Ct_tactile_asm.exclude{np, 4}   = [];
    dataset_Ct_tactile_asm.exclude{np, 5}   = [];
    dataset_Ct_tactile_asm.exclude{np, 6}   = [];
    dataset_Ct_tactile_asm.exclude{np, 7}   = [];
    dataset_Ct_tactile_asm.exclude{np, 8}   = [];
end

%% Add Outlier Metric

dataset_Ct_tactile_asm.outlier_metric   = cell(N_primitive, N_settings);
dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric   = cell(N_primitive, N_settings);
dataset_Ct_tactile_asm.abs_max_prim2_orientation_ct_beta    = cell(N_primitive, N_settings);

for ns = 1:1:N_settings
    N_demos = size(dataset_Ct_tactile_asm.sub_Ct_target{1,ns}, 1);

    if (outlier_metric_criteria == 1)   % based on non-linear dimensionality reduction
        Ctt_N_dim                   = size(dataset_Ct_tactile_asm.sub_Ct_target{1,1}{1,1}, 2);
        m_N_dim                     = size(dataset_Ct_tactile_asm.sub_X{1,1}{1,1}, 2);

        stretched_Ctt_traj_cell     = cell(Ctt_N_dim, N_demos);
        stretched_m_traj_cell       = cell(  m_N_dim, N_demos);

        for ndm = 1:1:N_demos
            for Ctt_dim = 1:1:Ctt_N_dim
                stretched_Ctt_traj_prim_cell            = cell(1, N_primitive);
                for np=1:1:N_primitive
                    Ct_target_demo                      = dataset_Ct_tactile_asm.sub_Ct_target{np,ns}{ndm,1}(:,Ctt_dim);
    %                 stretched_Ctt_traj                  = stretchTrajectory( filtfilt(b, a, Ct_target_demo.'), traj_length_disp );
                    stretched_Ctt_traj_prim_cell{1, np} = stretchTrajectory( Ct_target_demo.', traj_length_disp );
                end
                stretched_Ctt_traj                      = cell2mat(stretched_Ctt_traj_prim_cell);
                stretched_Ctt_traj_cell{Ctt_dim, ndm}   = stretched_Ctt_traj.';
            end

            for m_dim = 1:1:m_N_dim
                stretched_m_traj_prim_cell              = cell(1, N_primitive);
                for np=1:1:N_primitive
                    modality_demo                       = dataset_Ct_tactile_asm.sub_X{np,ns}{ndm,1}(:,m_dim);
    %                 stretched_m_traj                    = stretchTrajectory( filtfilt(b, a, modality_demo.'), traj_length_disp );
                    stretched_m_traj_prim_cell{1, np}   = stretchTrajectory( modality_demo.', traj_length_disp );
                end
                stretched_m_traj                        = cell2mat(stretched_m_traj_prim_cell);
                stretched_m_traj_cell{m_dim, ndm}       = stretched_m_traj.';
            end
        end

        stretched_Ctt_traj_concat                   = cell2mat(stretched_Ctt_traj_cell);
        stretched_m_tactile_electrodes_traj_concat  = cell2mat(stretched_m_traj_cell(1:N_sensor_dimension,:));

        %% Perform Isomap-based Pruning to Prune Outlier Demonstrations

        % normalize the data first:
        stretched_Ctt_traj_concat_mean          = mean(stretched_Ctt_traj_concat, 2);
        stretched_Ctt_traj_concat_std           = std(stretched_Ctt_traj_concat, 0, 2);
        stretched_Ctt_traj_concat_normalized    = (stretched_Ctt_traj_concat - repmat(stretched_Ctt_traj_concat_mean, 1, N_demos)) ./ ...
                                                  repmat(stretched_Ctt_traj_concat_std, 1, N_demos);

        stretched_m_tactile_electrodes_traj_concat_mean         = mean(stretched_m_tactile_electrodes_traj_concat, 2);
        stretched_m_tactile_electrodes_traj_concat_std          = std(stretched_m_tactile_electrodes_traj_concat, 0, 2);
        stretched_m_tactile_electrodes_traj_concat_std(find(stretched_m_tactile_electrodes_traj_concat_std == 0), 1)    = 1;
        stretched_m_tactile_electrodes_traj_concat_normalized   = (stretched_m_tactile_electrodes_traj_concat - repmat(stretched_m_tactile_electrodes_traj_concat_mean, 1, N_demos)) ./ ...
                                                                  repmat(stretched_m_tactile_electrodes_traj_concat_std, 1, N_demos);

        if (sum(isnan(stretched_Ctt_traj_concat_normalized)) > 0)
            keyboard;
        end
        if (sum(isnan(stretched_m_tactile_electrodes_traj_concat_normalized)) > 0)
            keyboard;
        end

        if (distance_metric_mode == 1)
            strt_repl_Ctt_m_tactile_electrodes_traj_concat_normalized   = stretched_Ctt_traj_concat_normalized;
        elseif (distance_metric_mode == 2)
            strt_repl_Ctt_m_tactile_electrodes_traj_concat_normalized   = stretched_m_tactile_electrodes_traj_concat_normalized;
        elseif (distance_metric_mode == 3)
            % replicate Ctt and m_tactile_electrodes data, 
            % based on their greatest common divisor, 
            % such that they both contribute equally to the distance measure:
            gcd_Ctt_m_tactile_electrodes        = gcd(Ctt_N_dim, N_sensor_dimension);
            N_Ctt_replication                   = N_sensor_dimension/gcd_Ctt_m_tactile_electrodes;
            N_m_tactile_electrodes_replication  = Ctt_N_dim/gcd_Ctt_m_tactile_electrodes;

            strt_repl_Ctt_m_tactile_electrodes_traj_concat_normalized   = [repmat(stretched_Ctt_traj_concat_normalized, N_Ctt_replication, 1); ...
                                                                           repmat(stretched_m_tactile_electrodes_traj_concat_normalized, N_m_tactile_electrodes_replication, 1)];
        end

        D   = L2_distance(strt_repl_Ctt_m_tactile_electrodes_traj_concat_normalized, ...
                          strt_repl_Ctt_m_tactile_electrodes_traj_concat_normalized, 1);

        % Computing the solution of MDS (Multi-Dimensional Scaling):
        [data_mds,eigvals]  = cmdscale(D);

        % data indices
        data_ID     = cell(0);
        for ndm = 1:1:N_demos
            data_ID{1, ndm} = num2str(ndm);
        end

        mean_data_mds       = mean(data_mds, 1);
        std_data_mds        = std(data_mds, 0, 1);

    %     if (is_visualizing)
            % Plotting the unnormalized "coordinate" of the data points and its ID number:
    %         figure;
    %         axis equal;
    %         hold on;
    %             plot(data_mds(:,1),data_mds(:,2),'.');
    %             text(data_mds(:,1)+25,data_mds(:,2),data_ID);
    %             xlabel('x');
    %             ylabel('y');
    %             title(['Unnormalized MDS (Multi-Dimensional Scaling) Coordinates, Setting #',num2str(ns)]);
    % 
    %             plot( mean_data_mds(1,1), mean_data_mds(1,2), 'k*' );
    %             plot_ellipse( mean_data_mds(1,1), mean_data_mds(1,2), std_data_mds(1,1), std_data_mds(1,2) );
    %             plot_ellipse( mean_data_mds(1,1), mean_data_mds(1,2), 2*std_data_mds(1,1), 2*std_data_mds(1,2) );
    %             plot_ellipse( mean_data_mds(1,1), mean_data_mds(1,2), 3*std_data_mds(1,1), 3*std_data_mds(1,2) );
    %         hold off;
    %     end

        % data indices
        data_ID     = cell(0);
        for ndm = 1:1:N_demos
            data_ID{1, ndm} = num2str(ndm);
        end

        data_mds_normalized = (data_mds - repmat(mean_data_mds, size(data_mds, 1), 1)) ./ repmat(std_data_mds, size(data_mds, 1), 1);

        data_mds_normalized_2D_norm = sqrt(sum((data_mds_normalized(:,1:2) .^ 2), 2));
    elseif (outlier_metric_criteria == 2)   % based on primitive 2's orientation coupling term in 2nd dimension (beta)
        mean_p2_ori_ct_beta_depolarized     = zeros(N_demos, 1);
        max_p2_ori_ct_beta_depolarized      = zeros(N_demos, 1);
        min_p2_ori_ct_beta_depolarized      = zeros(N_demos, 1);
        min_p3_ori_ct_beta_depolarized      = zeros(N_demos, 1);
        for ndm = 1:1:N_demos
            p2_ori_ct_beta 	= dataset_Ct_tactile_asm.sub_Ct_target{2,ns}{ndm,1}(:,5);
            p2_ori_ct_beta_depolarized = p2_and_p3_orientation_ct_beta_polarity(ns, 1) * p2_ori_ct_beta;
            mean_p2_ori_ct_beta_depolarized(ndm,1)  = mean(p2_ori_ct_beta_depolarized);
            max_p2_ori_ct_beta_depolarized(ndm,1)   = max(p2_ori_ct_beta_depolarized);
            min_p2_ori_ct_beta_depolarized(ndm,1)   = min(p2_ori_ct_beta_depolarized);
            
            p3_ori_ct_beta 	= dataset_Ct_tactile_asm.sub_Ct_target{3,ns}{ndm,1}(:,5);
            p3_ori_ct_beta_depolarized = p2_and_p3_orientation_ct_beta_polarity(ns, 1) * p3_ori_ct_beta;
            min_p3_ori_ct_beta_depolarized(ndm,1)   = min(p3_ori_ct_beta_depolarized);
        end
        max_max_p2_ori_ct_beta_depolarized  = max(max_p2_ori_ct_beta_depolarized);
        demos_to_be_excluded= find(max_p2_ori_ct_beta_depolarized < abs_max_prim2_orientation_ct_beta_fraction_retain * max_max_p2_ori_ct_beta_depolarized);
        demos_to_be_excluded= union(demos_to_be_excluded, find(min_p2_ori_ct_beta_depolarized < 0));
%         demos_to_be_excluded= union(demos_to_be_excluded, find(min_p3_ori_ct_beta_depolarized < 0));
        [~, demo_rank]      = sort(mean_p2_ori_ct_beta_depolarized, 1, 'descend');
    end

    for np=1:1:N_primitive
        if (outlier_metric_criteria == 1)       % based on non-linear dimensionality reduction
            dataset_Ct_tactile_asm.outlier_metric{np,ns}    = data_mds_normalized_2D_norm;
        elseif (outlier_metric_criteria == 2)   % based on primitive 2's orientation coupling term in 2nd dimension (beta)
            dataset_Ct_tactile_asm.outlier_metric{np,ns}    = demo_rank;
            dataset_Ct_tactile_asm.exclude{np,ns}           = union(dataset_Ct_tactile_asm.exclude{np,ns}, demos_to_be_excluded);
            dataset_Ct_tactile_asm.abs_max_prim2_orientation_ct_beta{np,ns} = max_max_p2_ori_ct_beta_depolarized;
        end
        [~, dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric{np,ns}]       = sort(dataset_Ct_tactile_asm.outlier_metric{np,ns});
        dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}= setdiff(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric{np,ns},...
                                                                                              dataset_Ct_tactile_asm.exclude{np,ns},'stable');
    end

    if ((outlier_metric_criteria == 1) && is_visualizing)
        % Plotting the normalized "coordinate" of the data points and its ID number:
        figure;
        axis equal;
        hold on;
            plot(data_mds_normalized(:,1),data_mds_normalized(:,2),'.');
            text(data_mds_normalized(:,1)+0.2,data_mds_normalized(:,2),data_ID);
            xlabel('x');
            ylabel('y');
            title(['Normalized MDS (Multi-Dimensional Scaling) Coordinates, Setting #',num2str(ns)]);

            plot( 0, 0, 'k*' );
            plot_ellipse( 0, 0, 1, 1 );
            plot_ellipse( 0, 0, 2, 2 );
            plot_ellipse( 0, 0, 3, 3 );
        hold off;

        keyboard;
    end
end

save(['dataset_Ct_tactile_asm_',task_type,'_augmented.mat'],'dataset_Ct_tactile_asm');