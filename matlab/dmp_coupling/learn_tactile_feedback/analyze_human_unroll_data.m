% Author: Giovanni Sutanto
% Date  : July 2017

clear  	all;
close   all;
clc;

addpath('../../utilities/');
addpath('../../cart_dmp/cart_coord_dmp/');
addpath('../../cart_dmp/quat_dmp/');
addpath('../../dmp_multi_dim/');

task_type               = 'scraping';
date                    = '20170720';

in_data_root_dir_path   = ['~/Desktop/dmp_robot_unroll_results/',task_type,'/',date,'/human/'];
% in_data_dir_name        = 'negative_5';
in_data_dir_name        = 'negative_10';
demo_types              = {'baseline', 'coupled', 'prevdemo'};
legend_types            = {'noncorrected', 'corrected', 'prevdemo'};
display_time_offset     = {{101}, {317}, {907}};
N_DT                    = size(demo_types, 2);

N_points_ave_init_offset= 5;

%% Data Loading

X_sense_cell            = cell(0);

for ndt = 1:size(demo_types, 2)
    in_data_dir_path    = [in_data_root_dir_path, in_data_dir_name, '/', demo_types{1, ndt}, '/'];
    
    X_sense_cell{1, ndt}= cell(0);
    
    data_files          = dir([in_data_dir_path,'/d*']);
    data_file_count  	= 1;
    for data_file = data_files'
        in_data_file_path   = [in_data_dir_path,'/',data_file.name];
        [D,vars,freq]       = clmcplot_convert(in_data_file_path);
        dt                  = 1.0/freq;

        traj_R_LF_electrodes    = clmcplot_getvariables(D, vars, getDataNamesWithNumericIndex('R_LF_','E',[1:19]));
        traj_R_RF_electrodes    = clmcplot_getvariables(D, vars, getDataNamesWithNumericIndex('R_RF_','E',[1:19]));
        traj_joint_positions    = clmcplot_getvariables(D, vars, ...
                                                        getDataNamesFromCombinations('R_', ...
                                                            {'SFE_', 'SAA_', 'HR_', 'EB_', 'WR_', 'WFE_', 'WAA_'}, ...
                                                            {'th'}));
        X_sense             = [traj_R_LF_electrodes, traj_R_RF_electrodes, traj_joint_positions];

        % Clipping Points Determination
        traj_is_non_zero    = zeros(size(X_sense,1), 1);
        for dim=1:size(X_sense, 2)
            traj_is_non_zero= traj_is_non_zero | (X_sense(:,dim) ~= 0);
        end
        [ start_clipping_idx, end_clipping_idx ] = getNullClippingIndex( traj_is_non_zero );
        
        % Subtracting BioTac Electrodes Sensor Traces Offset
        X_sense             = X_sense(start_clipping_idx+display_time_offset{1, ndt}{1,1}:end_clipping_idx, :);
%         X_sense             = X_sense(start_clipping_idx+display_time_offset{1, ndt}{1,1}:1000+display_time_offset{1, ndt}{1,1}, :);
        
        BT_electrode_offset = (1.0/N_points_ave_init_offset) * sum(X_sense(1:N_points_ave_init_offset,1:38),1);
        X_sense(:,1:38)     = X_sense(:,1:38) - repmat(BT_electrode_offset, size(X_sense, 1), 1);
        
        X_sense_cell{1, ndt}{1, data_file_count} = X_sense;

        data_file_count     = data_file_count + 1;
    end
end

% end of Data Loading

%% Plotting

X_plot_group    = {[1:19], [20:38], [39:45]};
plot_titles     = {'R\_LF\_electrodes', 'R\_RF\_electrodes', ...
                   'Joint Positions (Proprioception)'};

for nxplot=1:size(X_plot_group, 2)
    figure;
    X_plot_indices  = X_plot_group{1, nxplot};
    D               = length(X_plot_indices);
    for d=1:D
        data_idx    = X_plot_indices(d);
        N_plot_cols = ceil(D/5);
        subplot(ceil(D/N_plot_cols),N_plot_cols,d);
        if (d==1)
            title([plot_titles{1,nxplot}]);
        end
        hold on;
            for ndt = 1:size(demo_types, 2)
                for nt = 1:size(X_sense_cell{1,ndt}, 2)
                    if (ndt == 1)
                        pnc     = plot(X_sense_cell{1,ndt}{1,nt}(:,data_idx), 'b','LineWidth',3);
                    elseif (ndt == 2)
                        pc      = plot(X_sense_cell{1,ndt}{1,nt}(:,data_idx), 'r','LineWidth',3);
                    elseif (ndt == 3)
                        ppd     = plot(X_sense_cell{1,ndt}{1,nt}(:,data_idx), 'm');
                    end
                end
            end
            ylabel(['d = ', num2str(d)]);

            if (d==1)
                legend([pnc, pc, ppd], legend_types);
            end
        hold off;
    end
end

% end of Plotting