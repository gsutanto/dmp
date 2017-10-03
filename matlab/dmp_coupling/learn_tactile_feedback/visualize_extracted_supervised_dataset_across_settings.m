% Author: Giovanni Sutanto
% Date  : July 2017
% Description   :
%   Visualize the extracted supervised dataset
%   (target X (sensor input) - Ct/coupling term (output) pairs) 
%   overlayed together across settings to observe how these dataset
%   transitions from one setting to another.

clear all;
close all;
clc;

rel_dir_path        = './';

addpath([rel_dir_path, '../../utilities/']);
addpath([rel_dir_path, '../../cart_dmp/cart_coord_dmp/']);
addpath([rel_dir_path, '../../cart_dmp/quat_dmp/']);
addpath([rel_dir_path, '../../dmp_multi_dim/']);

task_type           = 'scraping';
load(['dataset_Ct_tactile_asm_',task_type,'_augmented.mat']);

is_plotting_position_ct_target      = 1;
is_plotting_orientation_ct_target   = 1;

is_plotting_ct_target_separately    = 0;
is_plotting_ct_target_overlayed     = 1;
is_plotting_deltaX_overlayed        = 1;

is_averaging_per_setting            = 1;
if (is_averaging_per_setting)
    plot_desc       = 'averaged';
else
    plot_desc       = 'N-best';
end

N_demo_display      = 3;

D                   = 3;

traj_stretch_length = 1500;

prim_no             = 2;

N_settings          = size(dataset_Ct_tactile_asm.sub_Ct_target, 2);

zero_horizon_line   = zeros(traj_stretch_length, 1);

[ linespec_codes ]  = generateLinespecCodes();

%% Plotting Target Coupling Term (Ct_target) Separately for each Setting

if (is_plotting_ct_target_separately)
    for ns=1:N_settings
        N_demo  = size(dataset_Ct_tactile_asm.sub_Ct_target{prim_no,ns}, 1);

        if (is_plotting_position_ct_target)
            figure;
            act_type_string     = 'position';
            for d=1:D
                subplot(D,1,d);
                hold on;
                    if (d == 1)
                        dim_string  = 'x';
                    elseif (d == 2)
                        dim_string  = 'y';
                    elseif (d == 3)
                        dim_string  = 'z';
                    end
                    plot(zero_horizon_line, 'g', 'LineWidth', 2);
                    for nd=1:min(N_demo_display, size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{prim_no,ns}, 1))
                        demo_idx            = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{prim_no,ns}(nd,1);
                        demo_ct_target_dim  = dataset_Ct_tactile_asm.sub_Ct_target{prim_no,ns}{demo_idx,1}(:,d);
                        stretched_demo_ct_target_dim    = stretchTrajectory(demo_ct_target_dim.', traj_stretch_length).';
                        plot(stretched_demo_ct_target_dim, 'b');
                    end
                    if (d == 1)
                        title([act_type_string, ' coupling term, prim #', num2str(prim_no), ', setting #', num2str(ns)]);
                        legend('zero-line', 'demos');
                    end
                hold off;
            end
        end

        if (is_plotting_orientation_ct_target)
            figure;
            act_type_string     = 'orientation';
            for d=1:D
                subplot(D,1,d);
                hold on;
                    if (d == 1)
                        dim_string  = 'alpha';
                    elseif (d == 2)
                        dim_string  = 'beta';
                    elseif (d == 3)
                        dim_string  = 'gamma';
                    end
                    plot(zero_horizon_line, 'g', 'LineWidth', 2);
                    for nd=1:min(N_demo_display, size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{prim_no,ns}, 1))
                        demo_idx            = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{prim_no,ns}(nd,1);
                        demo_ct_target_dim  = dataset_Ct_tactile_asm.sub_Ct_target{prim_no,ns}{demo_idx,1}(:,D+d);
                        stretched_demo_ct_target_dim    = stretchTrajectory(demo_ct_target_dim.', traj_stretch_length).';
                        plot(stretched_demo_ct_target_dim, 'b');
                    end
                    if (d == 1)
                        title([act_type_string, ' coupling term, prim #', num2str(prim_no), ', setting #', num2str(ns)]);
                        legend('zero-line', 'demos');
                    end
                hold off;
            end
        end
    end
end

% end of Plotting Target Coupling Term (Ct_target) Separately for each Setting

%% Plotting Target Coupling Term (Ct_target) Overlayed All Settings

if (is_plotting_ct_target_overlayed)
    if (is_plotting_position_ct_target)
        figure;
        act_type_string     = 'position';
        for d=1:D
            subplot(D,1,d);
            hold on;
                if (d == 1)
                    dim_string  = 'x';
                elseif (d == 2)
                    dim_string  = 'y';
                elseif (d == 3)
                    dim_string  = 'z';
                end
                pz  = plot(zero_horizon_line, 'g', 'LineWidth', 2);
                ps  = cell(N_settings, 1);  % cell of plot handles
                ls  = cell(N_settings, 1);  % cell of plot legends
                
                for ns=1:N_settings
                    if (is_averaging_per_setting == 0)
                        for nd=1:min(N_demo_display, size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{prim_no,ns}, 1))
                            demo_idx            = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{prim_no,ns}(nd,1);
                            demo_ct_target_dim  = dataset_Ct_tactile_asm.sub_Ct_target{prim_no,ns}{demo_idx,1}(:,d);
                            stretched_demo_ct_target_dim    = stretchTrajectory(demo_ct_target_dim.', traj_stretch_length).';
                            ps{ns}              = plot(stretched_demo_ct_target_dim, linespec_codes{1,ns});
                        end
                    else
                        N_demo                  = size(dataset_Ct_tactile_asm.sub_Ct_target{prim_no,ns}, 1);
                        accum_stretched_demo_ct_target_dim  = zeros(traj_stretch_length, 1);
                        for nd=1:N_demo
                            demo_ct_target_dim  = dataset_Ct_tactile_asm.sub_Ct_target{prim_no,ns}{nd,1}(:,d);
                            stretched_demo_ct_target_dim    = stretchTrajectory(demo_ct_target_dim.', traj_stretch_length).';
                            accum_stretched_demo_ct_target_dim  = accum_stretched_demo_ct_target_dim + stretched_demo_ct_target_dim;
                        end
                        accum_stretched_demo_ct_target_dim  = accum_stretched_demo_ct_target_dim/N_demo;
                      	ps{ns}                  = plot(accum_stretched_demo_ct_target_dim, linespec_codes{1,ns});
                    end
                    
                    ls{ns}                      = num2str(ns);
                end
                
                if (d == 1)
                    title([act_type_string, ' coupling term, prim #', num2str(prim_no), ' (', plot_desc, ')']);
                    legend([pz, ps{:}], 'zero-line', ls{:});
                end
            hold off;
        end
    end

    if (is_plotting_orientation_ct_target)
        figure;
        act_type_string     = 'orientation';
        for d=1:D
            subplot(D,1,d);
            hold on;
                if (d == 1)
                    dim_string  = 'alpha';
                elseif (d == 2)
                    dim_string  = 'beta';
                elseif (d == 3)
                    dim_string  = 'gamma';
                end
                pz  = plot(zero_horizon_line, 'g', 'LineWidth', 2);
                ps  = cell(N_settings, 1);  % cell of plot handles
                ls  = cell(N_settings, 1);  % cell of plot legends
                
                for ns=1:N_settings
                    if (is_averaging_per_setting == 0)
                        for nd=1:min(N_demo_display, size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{prim_no,ns}, 1))
                            demo_idx            = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{prim_no,ns}(nd,1);
                            demo_ct_target_dim  = dataset_Ct_tactile_asm.sub_Ct_target{prim_no,ns}{demo_idx,1}(:,D+d);
                            stretched_demo_ct_target_dim    = stretchTrajectory(demo_ct_target_dim.', traj_stretch_length).';
                            ps{ns}              = plot(stretched_demo_ct_target_dim, linespec_codes{1,ns});
                        end
                    else
                        N_demo                  = size(dataset_Ct_tactile_asm.sub_Ct_target{prim_no,ns}, 1);
                        accum_stretched_demo_ct_target_dim  = zeros(traj_stretch_length, 1);
                        for nd=1:N_demo
                            demo_ct_target_dim  = dataset_Ct_tactile_asm.sub_Ct_target{prim_no,ns}{nd,1}(:,D+d);
                            stretched_demo_ct_target_dim    = stretchTrajectory(demo_ct_target_dim.', traj_stretch_length).';
                            accum_stretched_demo_ct_target_dim  = accum_stretched_demo_ct_target_dim + stretched_demo_ct_target_dim;
                        end
                        accum_stretched_demo_ct_target_dim  = accum_stretched_demo_ct_target_dim/N_demo;
                      	ps{ns}                  = plot(accum_stretched_demo_ct_target_dim, linespec_codes{1,ns});
                    end
                    
                    ls{ns}                      = num2str(ns);
                end
                
                if (d == 1)
                    title([act_type_string, ' coupling term, prim #', num2str(prim_no), ' (', plot_desc, ')']);
                    legend([pz, ps{:}], 'zero-line', ls{:});
                end
            hold off;
        end
    end
end

% end of Plotting Target Coupling Term (Ct_target) Overlayed All Settings

%% Plotting Sensor Trace Deviation (Delta X) Overlayed All Settings

X_plot_group    = {[1:19], [20:38]};
plot_titles     = {'R\_LF\_electrodes', 'R\_RF\_electrodes'};

for nxplot=1:size(X_plot_group, 2)
    figure;
    X_plot_indices  = X_plot_group{1, nxplot};
    D               = length(X_plot_indices);
    for d=1:D
        data_idx    = X_plot_indices(d);
        N_plot_cols = ceil(D/5);
        subplot(ceil(D/N_plot_cols),N_plot_cols,d);
        if (d==2)
            title([plot_titles{1,nxplot},', Primitive #',num2str(prim_no), ' (', plot_desc, ')']);
        end
        hold on;
            ps  = cell(N_settings, 1);  % cell of plot handles
            ls  = cell(N_settings, 1);  % cell of plot legends
            
           	for ns=1:N_settings
                if (is_averaging_per_setting == 0)
                    for nd=1:min(N_demo_display, size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{prim_no,ns}, 1))
                        demo_idx                    = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{prim_no,ns}(nd,1);
                        X_dim_demo_traj         	= dataset_Ct_tactile_asm.sub_X{prim_no,ns}{demo_idx,1}(:,data_idx);
                        stretched_X_dim_demo_traj   = stretchTrajectory( X_dim_demo_traj', traj_stretch_length )';
                        ps{ns}                      = plot(stretched_X_dim_demo_traj, linespec_codes{1,ns});
                    end
                else
                    N_demo  = size(dataset_Ct_tactile_asm.sub_X{prim_no,ns}, 1);
                    accum_stretched_X_dim_demo_traj = zeros(traj_stretch_length, 1);
                    for nd=1:N_demo
                        X_dim_demo_traj         	= dataset_Ct_tactile_asm.sub_X{prim_no,ns}{nd,1}(:,data_idx);
                        stretched_X_dim_demo_traj   = stretchTrajectory( X_dim_demo_traj', traj_stretch_length )';
                        accum_stretched_X_dim_demo_traj = accum_stretched_X_dim_demo_traj + stretched_X_dim_demo_traj;
                    end
                    accum_stretched_X_dim_demo_traj = accum_stretched_X_dim_demo_traj/N_demo;
                    ps{ns}                        	= plot(accum_stretched_X_dim_demo_traj, linespec_codes{1,ns});
                end
                    
                ls{ns}                              = num2str(ns);
            end
            
            if (d==1)
                legend([ps{:}], ls{:});
            end
            xlabel(['Electrode #', num2str(d)]);
        hold off;
    end
end

% end of Plotting Sensor Trace Deviation (Delta X) Overlayed All Settings