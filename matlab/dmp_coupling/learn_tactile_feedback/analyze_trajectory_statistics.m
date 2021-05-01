clear all;
close all;
clc;

rel_dir_path        = './';

addpath([rel_dir_path, '../../utilities/']);
addpath([rel_dir_path, '../../cart_dmp/cart_coord_dmp/']);
addpath([rel_dir_path, '../../cart_dmp/quat_dmp/']);
addpath([rel_dir_path, '../../dmp_multi_dim/']);

generic_task_type   = 'scraping';

scraping_data_root_dir_path = [rel_dir_path, '../../../data/dmp_coupling/learn_tactile_feedback/',generic_task_type,'/'];

% tau_multiplier      = 4.0;
tau_multiplier      = 1.0;
task_servo_rate     = 300.0;
dt                  = 1.0/task_servo_rate;

modality_names          = {'position', 'orientation', 'sense_R_LF', 'sense_R_RF', 'sense_proprio'};
modality_dimensionality = {9,9,19,19,7};
modality_mapping        = {2,3,6,14,22};

%% Loading the Demonstrations

load([rel_dir_path, 'dmp_baseline_params_',generic_task_type,'.mat']);
load([rel_dir_path, 'data_demo_',generic_task_type,'.mat']);

N_prims             = size(data_demo.coupled, 1);

% end of Loading the Demonstrations

%% Test Setting and Trial Number Particular Selection

outdata_root_dir_path	= [scraping_data_root_dir_path, 'unroll_test_dataset/all_prims/'];

setting_no  = 2;
trial_no    = 3;

trial_outdata_dir_path  = [outdata_root_dir_path, 'setting_', num2str(setting_no), '_trial_', num2str(trial_no), '/'];

% we will unroll with similar length (up to constant multiplier) 
% as selected setting and trial above
unroll_traj_length  = zeros(N_prims, 1);
unroll_tau          = zeros(N_prims, 1);
for np=1:N_prims
    unroll_traj_length(np, 1)   = round(tau_multiplier * size(data_demo.coupled{np, setting_no}{trial_no, 1}, 1));
    unroll_tau(np, 1)           = (unroll_traj_length(np, 1) - 1) * dt;
end

% end of Test Setting and Trial Number Particular Selection

%% Data Reorganization (Grouping), Conversion, and Stretching

N_settings  = size(data_demo.coupled, 2);
N_modality  = size(modality_dimensionality,2);

data_dist_from_prim_goal_per_dims   = cell(N_prims, N_settings+1);  % columns 1:N_settings belongs to coupled demonstrations, column (N_settings+1) belongs to baseline demonstration

for np=1:N_prims
    for ns=1:N_settings+1
        data_dist_from_prim_goal_per_dims{np,ns}	= cell(1,N_modality);
        if (ns <= N_settings)
            N_demo  = size(data_demo.coupled{np,ns},1);
        else
            N_demo  = size(data_demo.baseline{np,1},1);
        end
        for nm=1:N_modality
            N_MD    = modality_dimensionality{1,nm};
            data_dist_from_prim_goal_per_dims{np,ns}{1,nm}	= cell(N_MD,1);
            for nmd=1:N_MD
                data_dist_from_prim_goal_per_dims{np,ns}{1,nm}{nmd,1}   = zeros(unroll_traj_length(np, 1),N_demo);
                
                % CartCoord Position or ANY Sensing Modalities (treated as position signal as well)
                if (((nm == 1) && (nmd <= 3)) || (nm >= 3))
                    if (nm == 1)
                        goal    = dmp_baseline_params.cart_coord{np,1}.mean_goal_global(nmd, 1);
                    elseif ((nm == 3) || (nm == 4))
                        goal    = dmp_baseline_params.BT_electrode{np,nm-2}.mean_goal(nmd, 1);
                    else
                        goal    = dmp_baseline_params.joint_sense{np,1}.mean_goal(nmd, 1);
                    end
                    
                    for nd=1:N_demo
                        if (ns <= N_settings)
                            traj    = data_demo.coupled{np,ns}{nd,modality_mapping{1,nm}}(:,nmd);
                        else
                            traj    = data_demo.baseline{np,1}{nd,modality_mapping{1,nm}}(:,nmd);
                        end
                        goal_minus_traj     = goal - traj;
                        data_dist_from_prim_goal_per_dims{np,ns}{1,nm}{nmd,1}(:,nd) = stretchTrajectory(goal_minus_traj.',unroll_traj_length(np, 1)).';
                    end
                % CartCoord Velocity OR Acceleration OR 
                % Orientation Angular 
                elseif (((nm == 1) && (nmd >= 4)) || ((nm == 2) && (nmd >= 4)))
                    for nd=1:N_demo
                        if (nm == 1)    % CartCoord
                            if (ns <= N_settings)
                                traj    = data_demo.coupled{np,ns}{nd,modality_mapping{1,nm}}(:,nmd);
                            else
                                traj    = data_demo.baseline{np,1}{nd,modality_mapping{1,nm}}(:,nmd);
                            end
                        else            % Orientation
                            if (ns <= N_settings)
                                traj    = data_demo.coupled{np,ns}{nd,modality_mapping{1,nm}}(:,nmd+1);
                            else
                                traj    = data_demo.baseline{np,1}{nd,modality_mapping{1,nm}}(:,nmd+1);
                            end
                        end
                        data_dist_from_prim_goal_per_dims{np,ns}{1,nm}{nmd,1}(:,nd) = stretchTrajectory(traj.',unroll_traj_length(np, 1)).';
                    end
                % Orientation Quaternion
                elseif ((nm == 2) && (nmd <= 3))
                    QG  = dmp_baseline_params.Quat{np,1}.fit_mean_QG;

                    for nd=1:N_demo
                        if (ns <= N_settings)
                            Qtraj   = data_demo.coupled{np,ns}{nd,modality_mapping{1,nm}}(:,1:4).';
                        else
                            Qtraj   = data_demo.baseline{np,1}{nd,modality_mapping{1,nm}}(:,1:4).';
                        end
                        QGtraj  = repmat(QG, 1, size(Qtraj, 2));
                        log_quat_diff_from_goal   = computeLogQuatDifference(QGtraj, Qtraj);
                        stretched_log_quat_diff_from_goal = stretchTrajectory(log_quat_diff_from_goal(nmd,:),unroll_traj_length(np, 1)).';

                        data_dist_from_prim_goal_per_dims{np,ns}{1,nm}{nmd,1}(:,nd)	= stretched_log_quat_diff_from_goal;
                    end
                else
                    error('Should NEVER reach here!!!');
                end
            end
        end
    end
end

% end of Data Reorganization (Grouping), Conversion, and Stretching

%% Computation of Trajectory (Time-Series) Statistics

N_stat_types                = 2;    % the 1st one is the mean trajectory, and 
                                    % the 2nd one is the standard deviation trajectory
precision_string            = '%.20f';

%%% across ALL settings:
data_demo_time_stats    	= cell(N_prims, 1);

stat_pos_outdata_dir_path   = [outdata_root_dir_path, 'pos/'];
recreateDir(stat_pos_outdata_dir_path);

mean_pos_outdata_dir_path   = [stat_pos_outdata_dir_path, 'mean/'];
recreateDir(mean_pos_outdata_dir_path);

std_pos_outdata_dir_path    = [stat_pos_outdata_dir_path, 'std/'];
recreateDir(std_pos_outdata_dir_path);

stat_ori_outdata_dir_path   = [outdata_root_dir_path, 'ori/'];
recreateDir(stat_ori_outdata_dir_path);

mean_ori_outdata_dir_path   = [stat_ori_outdata_dir_path, 'mean/'];
recreateDir(mean_ori_outdata_dir_path);

std_ori_outdata_dir_path    = [stat_ori_outdata_dir_path, 'std/'];
recreateDir(std_ori_outdata_dir_path);

% across all demonstrations, for each primitive:
for np=1:N_prims
    data_demo_time_stats{np,1}  = cell(1,N_modality);
    for nm=1:N_modality
        N_MD                                = modality_dimensionality{1,nm};
        data_demo_time_stats{np,1}{1,nm}	= cell(N_stat_types,1);
        for nst=1:N_stat_types
            data_demo_time_stats{np,1}{1,nm}{nst,1}	= zeros(unroll_traj_length(np, 1), N_MD);
        end
        
        for nmd=1:N_MD
            data_dist_from_prim_goal_per_dims_all_settings_cell	= cell(1,N_settings+1);
            for ns=1:N_settings+1
                data_dist_from_prim_goal_per_dims_all_settings_cell{1,ns}	= data_dist_from_prim_goal_per_dims{np,ns}{1,nm}{nmd,1};
            end
            data_dist_from_prim_goal_per_dims_all_settings  = cell2mat(data_dist_from_prim_goal_per_dims_all_settings_cell);
            
            for nst=1:N_stat_types
                if (nst == 1)       % mean trajectory
                    data_demo_time_stats{np,1}{1,nm}{nst,1}(:,nmd)	= mean(data_dist_from_prim_goal_per_dims_all_settings, 2);
                elseif (nst == 2)   % standard deviation (std) trajectory
                    data_demo_time_stats{np,1}{1,nm}{nst,1}(:,nmd)  = std(data_dist_from_prim_goal_per_dims_all_settings, 0, 2);
                else
                    error('Should NEVER reach here!!!');
                end
            end
        end
        
        for nst=1:N_stat_types
            if ((nm == 1) || (nm == 2))
                if (nst == 1)       % mean trajectory
                    mean_traj   = data_demo_time_stats{np,1}{1,nm}{nst,1};
                    time_idx    = dt * [1:1:size(mean_traj, 1)].';
                    mean_traj   = [time_idx, mean_traj];
                    if (nm == 1)
                        dlmwrite([mean_pos_outdata_dir_path, num2str(np), '.txt'], mean_traj, 'delimiter', ' ', 'precision', precision_string);
                    elseif (nm == 2)
                        dlmwrite([mean_ori_outdata_dir_path, num2str(np), '.txt'], mean_traj, 'delimiter', ' ', 'precision', precision_string);
                    end
                    clear mean_traj;
                elseif (nst == 2)   % standard deviation (std) trajectory
                    std_traj    = data_demo_time_stats{np,1}{1,nm}{nst,1};
                    time_idx    = dt * [1:1:size(std_traj, 1)].';
                    std_traj    = [time_idx, std_traj];
                    if (nm == 1)
                        dlmwrite([std_pos_outdata_dir_path, num2str(np), '.txt'], std_traj, 'delimiter', ' ', 'precision', precision_string);
                    elseif (nm == 2)
                        dlmwrite([std_ori_outdata_dir_path, num2str(np), '.txt'], std_traj, 'delimiter', ' ', 'precision', precision_string);
                    end
                    clear std_traj;
                end
            end
        end
    end
end

%%% each setting (specified by setting_no):
data_demo_time_stats_per_setting    = cell(N_prims, N_settings+1);  % columns 1:N_settings belongs to coupled demonstrations, column (N_settings+1) belongs to baseline demonstration

stat_pos_setting_trial_outdata_dir_path   = [trial_outdata_dir_path, 'pos/'];
recreateDir(stat_pos_setting_trial_outdata_dir_path);

mean_pos_setting_trial_outdata_dir_path   = [stat_pos_setting_trial_outdata_dir_path, 'mean/'];
recreateDir(mean_pos_setting_trial_outdata_dir_path);

std_pos_setting_trial_outdata_dir_path    = [stat_pos_setting_trial_outdata_dir_path, 'std/'];
recreateDir(std_pos_setting_trial_outdata_dir_path);

stat_ori_setting_trial_outdata_dir_path   = [trial_outdata_dir_path, 'ori/'];
recreateDir(stat_ori_setting_trial_outdata_dir_path);

mean_ori_setting_trial_outdata_dir_path   = [stat_ori_setting_trial_outdata_dir_path, 'mean/'];
recreateDir(mean_ori_setting_trial_outdata_dir_path);

std_ori_setting_trial_outdata_dir_path    = [stat_ori_setting_trial_outdata_dir_path, 'std/'];
recreateDir(std_ori_setting_trial_outdata_dir_path);

for np=1:N_prims
    for ns=1:N_settings+1
        data_demo_time_stats_per_setting{np,ns} = cell(1,N_modality);
        for nm=1:N_modality
            N_MD        = modality_dimensionality{1,nm};
            data_demo_time_stats_per_setting{np,ns}{1,nm}   = cell(N_stat_types,1);
            
            for nst=1:N_stat_types
                data_demo_time_stats_per_setting{np,ns}{1,nm}{nst,1}    = zeros(unroll_traj_length(np, 1), N_MD);
                
                for nmd=1:N_MD
                    if (nst == 1)       % mean trajectory
                        data_demo_time_stats_per_setting{np,ns}{1,nm}{nst,1}(:,nmd) = mean(data_dist_from_prim_goal_per_dims{np,ns}{1,nm}{nmd,1}, 2);
                    elseif (nst == 2)   % standard deviation (std) trajectory
                        data_demo_time_stats_per_setting{np,ns}{1,nm}{nst,1}(:,nmd) = std(data_dist_from_prim_goal_per_dims{np,ns}{1,nm}{nmd,1}, 0, 2);
                    else
                        error('Should NEVER reach here!!!');
                    end
                end
                
                if (ns == setting_no)
                    if ((nm == 1) || (nm == 2))
                        if (nst == 1)       % mean trajectory
                            mean_traj   = data_demo_time_stats_per_setting{np,ns}{1,nm}{nst,1};
                            time_idx    = dt * [1:1:size(mean_traj, 1)].';
                            mean_traj   = [time_idx, mean_traj];
                            if (nm == 1)
                                dlmwrite([mean_pos_setting_trial_outdata_dir_path, num2str(np), '.txt'], mean_traj, 'delimiter', ' ', 'precision', precision_string);
                            elseif (nm == 2)
                                dlmwrite([mean_ori_setting_trial_outdata_dir_path, num2str(np), '.txt'], mean_traj, 'delimiter', ' ', 'precision', precision_string);
                            end
                            clear mean_traj;
                        elseif (nst == 2)   % standard deviation (std) trajectory
                            std_traj    = data_demo_time_stats_per_setting{np,ns}{1,nm}{nst,1};
                            time_idx    = dt * [1:1:size(std_traj, 1)].';
                            std_traj    = [time_idx, std_traj];
                            if (nm == 1)
                                dlmwrite([std_pos_setting_trial_outdata_dir_path, num2str(np), '.txt'], std_traj, 'delimiter', ' ', 'precision', precision_string);
                            elseif (nm == 2)
                                dlmwrite([std_ori_setting_trial_outdata_dir_path, num2str(np), '.txt'], std_traj, 'delimiter', ' ', 'precision', precision_string);
                            end
                            clear std_traj;
                        end
                    end
                end
            end
        end
    end
end

% end of Computation of Trajectory (Time-Series) Statistics

%% Plotting

D                           = 3;
std_tolerance_factor_pos    = 6.0;
std_tolerance_factor_vel    = 8.0;
std_tolerance_factor_acc    = 6.0;

for np=1:N_prims
    figure; % position
    for d=1:D
        subplot(D,1,d);
        hold on;
            mean_traj   = data_demo_time_stats_per_setting{np,setting_no}{1,1}{1,1}(:,d);
            std_traj    = data_demo_time_stats_per_setting{np,setting_no}{1,1}{2,1}(:,d);
            upper_bound = mean_traj + (std_tolerance_factor_pos * std_traj);
            lower_bound = mean_traj - (std_tolerance_factor_pos * std_traj);
            plot(mean_traj, 'k-.');
            plot(upper_bound, 'k-');
            plot(lower_bound, 'k-');
            
            N_demo  = size(data_demo.coupled{np,setting_no},1);
            for nd=1:N_demo
                plot(data_dist_from_prim_goal_per_dims{np,setting_no}{1,1}{d,1}(:,nd), 'g');
            end
            
            if (d == 1)
                title(['position metric: (G - x), primitive #', num2str(np)]);
                legend('mean', 'upper-bound', 'lower-bound', 'demo');
            end
        hold off;
    end
    
    figure; % orientation
    for d=1:D
        subplot(D,1,d);
        hold on;
            mean_traj   = data_demo_time_stats_per_setting{np,setting_no}{1,2}{1,1}(:,d);
            std_traj    = data_demo_time_stats_per_setting{np,setting_no}{1,2}{2,1}(:,d);
            upper_bound = mean_traj + (std_tolerance_factor_pos * std_traj);
            lower_bound = mean_traj - (std_tolerance_factor_pos * std_traj);
            plot(mean_traj, 'k-.');
            plot(upper_bound, 'k-');
            plot(lower_bound, 'k-');
            
            N_demo  = size(data_demo.coupled{np,setting_no},1);
            for nd=1:N_demo
                plot(data_dist_from_prim_goal_per_dims{np,setting_no}{1,2}{d,1}(:,nd), 'g');
            end
            
            if (d == 1)
                title(['orientation metric: 2 log (QG compose Q*), primitive #', num2str(np)]);
                legend('mean', 'upper-bound', 'lower-bound', 'demo');
            end
        hold off;
    end
    
    figure; % velocity
    for d=1:D
        subplot(D,1,d);
        hold on;
            mean_traj   = data_demo_time_stats_per_setting{np,setting_no}{1,1}{1,1}(:,d+D);
            std_traj    = data_demo_time_stats_per_setting{np,setting_no}{1,1}{2,1}(:,d+D);
            upper_bound = mean_traj + (std_tolerance_factor_vel * std_traj);
            lower_bound = mean_traj - (std_tolerance_factor_vel * std_traj);
            plot(mean_traj, 'k-.');
            plot(upper_bound, 'k-');
            plot(lower_bound, 'k-');
            
            N_demo  = size(data_demo.coupled{np,setting_no},1);
            for nd=1:N_demo
                plot(data_dist_from_prim_goal_per_dims{np,setting_no}{1,1}{d+D,1}(:,nd), 'g');
            end
            
            if (d == 1)
                title(['velocity, primitive #', num2str(np)]);
                legend('mean', 'upper-bound', 'lower-bound', 'demo');
            end
        hold off;
    end
    
    figure; % angular velocity
    for d=1:D
        subplot(D,1,d);
        hold on;
            mean_traj   = data_demo_time_stats_per_setting{np,setting_no}{1,2}{1,1}(:,d+D);
            std_traj    = data_demo_time_stats_per_setting{np,setting_no}{1,2}{2,1}(:,d+D);
            upper_bound = mean_traj + (std_tolerance_factor_vel * std_traj);
            lower_bound = mean_traj - (std_tolerance_factor_vel * std_traj);
            plot(mean_traj, 'k-.');
            plot(upper_bound, 'k-');
            plot(lower_bound, 'k-');
            
            N_demo  = size(data_demo.coupled{np,setting_no},1);
            for nd=1:N_demo
                plot(data_dist_from_prim_goal_per_dims{np,setting_no}{1,2}{d+D,1}(:,nd), 'g');
            end
            
            if (d == 1)
                title(['angular velocity, primitive #', num2str(np)]);
                legend('mean', 'upper-bound', 'lower-bound', 'demo');
            end
        hold off;
    end
    
    figure; % acceleration
    for d=1:D
        subplot(D,1,d);
        hold on;
            mean_traj   = data_demo_time_stats_per_setting{np,setting_no}{1,1}{1,1}(:,d+D+D);
            std_traj    = data_demo_time_stats_per_setting{np,setting_no}{1,1}{2,1}(:,d+D+D);
            upper_bound = mean_traj + (std_tolerance_factor_acc * std_traj);
            lower_bound = mean_traj - (std_tolerance_factor_acc * std_traj);
            plot(mean_traj, 'k-.');
            plot(upper_bound, 'k-');
            plot(lower_bound, 'k-');
            
            N_demo  = size(data_demo.coupled{np,setting_no},1);
            for nd=1:N_demo
                plot(data_dist_from_prim_goal_per_dims{np,setting_no}{1,1}{d+D+D,1}(:,nd), 'g');
            end
            
            if (d == 1)
                title(['acceleration, primitive #', num2str(np)]);
                legend('mean', 'upper-bound', 'lower-bound', 'demo');
            end
        hold off;
    end
    
    figure; % angular acceleration
    for d=1:D
        subplot(D,1,d);
        hold on;
            mean_traj   = data_demo_time_stats_per_setting{np,setting_no}{1,2}{1,1}(:,d+D+D);
            std_traj    = data_demo_time_stats_per_setting{np,setting_no}{1,2}{2,1}(:,d+D+D);
            upper_bound = mean_traj + (std_tolerance_factor_acc * std_traj);
            lower_bound = mean_traj - (std_tolerance_factor_acc * std_traj);
            plot(mean_traj, 'k-.');
            plot(upper_bound, 'k-');
            plot(lower_bound, 'k-');
            
            N_demo  = size(data_demo.coupled{np,setting_no},1);
            for nd=1:N_demo
                plot(data_dist_from_prim_goal_per_dims{np,setting_no}{1,2}{d+D+D,1}(:,nd), 'g');
            end
            
            if (d == 1)
                title(['angular acceleration, primitive #', num2str(np)]);
                legend('mean', 'upper-bound', 'lower-bound', 'demo');
            end
        hold off;
    end
end

% end of Plotting
