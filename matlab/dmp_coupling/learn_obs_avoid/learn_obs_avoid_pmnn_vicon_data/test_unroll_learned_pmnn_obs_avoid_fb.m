% Author: Giovanni Sutanto
% Date  : October 2017
% Description:
%   Test unrolling 
%   the learned PMNN obstacle avoidance feedback/coupling term model.

clear  	all;
close   all;
clc;

addpath('../utilities/');
addpath('../vicon/');
addpath('../../../cart_dmp/cart_coord_dmp/');

[   ~, ~, ...
    loa_feat_methods_to_be_evaluated, ...
    feat_constraint_mode, ...
    loa_feat_methods, ...
    max_num_trajs_per_setting, ...
    D, n_rfs, c_order ]                 = getConfigParams();

%% Demo Dataset Loading

load('data_multi_demo_vicon_static_global_coord.mat');

% end of Demo Dataset Loading

%% Baseline Primitive Loading

load('dmp_baseline_params_obs_avoid.mat');

% end of Baseline Primitive Loading

%% Supervised Obstacle Avoidance Feedback Dataset Loading

load('dataset_Ct_obs_avoid.mat');

% end of Supervised Obstacle Avoidance Feedback Dataset Loading

%% Loading of Parameters

load('learning_param_obs_avoid.mat');
load('unrolling_param_obs_avoid.mat');
load('loa_feat_param_obs_avoid.mat');

learn_fb_task                   = 'learn_obs_avoid';
amd_clmc_dmp_root_dir_path      = '../../../../';
data_learn_fb_task_subdir_path  = [learn_fb_task, '/static_obs/'];
python_learn_fb_task_TF_models_prefix_subdir_path   = 'tf/';
PMNN_name                       = 'my_PMNN_obs_avoid_fb';
np                              = 1;

data_root_dir_path          = [amd_clmc_dmp_root_dir_path, 'data/'];
matlab_root_dir_path        = [amd_clmc_dmp_root_dir_path, 'matlab/'];
python_root_dir_path        = [amd_clmc_dmp_root_dir_path, 'python/'];

data_learn_fb_task_dir_path                     = [data_root_dir_path, 'dmp_coupling/', data_learn_fb_task_subdir_path];
data_learn_fb_task_PMNN_dir_path                = [data_learn_fb_task_dir_path, 'neural_nets/pmnn/'];
data_learn_fb_task_PMNN_python_models_dir_path  = [data_learn_fb_task_PMNN_dir_path, 'python_models/'];

python_learn_fb_task_dir_path           = [python_root_dir_path, 'dmp_coupling/', learn_fb_task, '/'];
python_learn_fb_task_TF_models_dir_path = [python_learn_fb_task_dir_path, python_learn_fb_task_TF_models_prefix_subdir_path, 'models/'];

addpath([matlab_root_dir_path, 'utilities/']);
addpath([matlab_root_dir_path, 'neural_nets/feedforward/pmnn/']);

reinit_selection_idx= dlmread([python_learn_fb_task_TF_models_dir_path, 'reinit_selection_idx.txt']);
TF_max_train_iters  = dlmread([python_learn_fb_task_TF_models_dir_path, 'TF_max_train_iters.txt']);

D_input             = size(dataset_Ct_obs_avoid.sub_X{1,1}{1,1}, 2);
regular_NN_hidden_layer_topology = dlmread([python_learn_fb_task_TF_models_dir_path, 'regular_NN_hidden_layer_topology.txt']);
N_phaseLWR_kernels  = size(dataset_Ct_obs_avoid.sub_normalized_phase_PSI_mult_phase_V{1,1}{1,1}, 2);
D_output            = size(dataset_Ct_obs_avoid.sub_Ct_target{1,1}{1,1}, 2);

regular_NN_hidden_layer_activation_func_list = readStringsToCell([python_learn_fb_task_TF_models_dir_path, 'regular_NN_hidden_layer_activation_func_list.txt']);

NN_info.name                = PMNN_name;
NN_info.topology            = [D_input, regular_NN_hidden_layer_topology, N_phaseLWR_kernels, D_output];
NN_info.activation_func_list= {'identity', regular_NN_hidden_layer_activation_func_list{:}, 'identity', 'identity'};
NN_info.filepath            = [data_learn_fb_task_PMNN_python_models_dir_path, 'prim_', num2str(np), '_params_reinit_',num2str(reinit_selection_idx(1,np)),'_step_',num2str(TF_max_train_iters,'%07d'),'.mat'];

learning_param.pmnn         = NN_info;

% end of Loading of Parameters

%% Unrolling Test

N_settings  = size(data_global_coord.obs_avoid, 1);
disp('Unrolling on Trained Obstacle Settings');

unroll_dataset_learned_Ct_obs_avoid.sub_Ct_target   = cell(1, N_settings);
unroll_dataset_learned_Ct_obs_avoid.sub_X           = cell(1, N_settings);
global_traj_unroll_setting_cell                     = cell(N_settings, 1);

normed_closest_distance_to_obs_traj_human_1st_demo_setting  = cell(N_settings, 1);
normed_closest_distance_to_obs_human_1st_demo_setting       = zeros(N_settings, 1);
normed_closest_distance_to_obs_human_1st_demo_setting_idx   = zeros(N_settings, 1);
final_distance_to_goal_vector_human_1st_demo_setting        = zeros(N_settings, 1);

normed_closest_distance_to_obs_traj_per_trained_setting_cell= cell(N_settings, 1);
normed_closest_distance_to_obs_per_trained_setting_cell    	= cell(N_settings, 1);
normed_closest_distance_to_obs_per_trained_setting_idx_cell = cell(N_settings, 1);
final_distance_to_goal_vector_per_trained_setting_cell     	= cell(N_settings, 1);
normed_closest_distance_to_obs_overall_train_demos_per_setting 	= zeros(N_settings, 1);
normed_closest_distance_to_obs_overall_train_demos_per_sett_idx = zeros(N_settings, 1);
worst_final_distance_to_goal_per_trained_setting        = zeros(N_settings, 1);
worst_final_distance_to_goal_per_trained_setting_idx    = zeros(N_settings, 1);
for i=1:N_settings
    if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
        N_demo_this_setting = 1;
    else
        N_demo_this_setting = min(max_num_trajs_per_setting, size(data_global_coord.obs_avoid{i,2}, 2));
    end
    
    unroll_dataset_learned_Ct_obs_avoid.sub_Ct_target{1,i}  = cell(N_demo_this_setting, 1);
    unroll_dataset_learned_Ct_obs_avoid.sub_X{1,i}          = cell(N_demo_this_setting, 1);
    global_traj_unroll_setting_cell{i,1}                    = cell(3, N_demo_this_setting);
    
    normed_closest_distance_to_obs_traj_per_trained_setting_cell{i,1}   = cell(N_demo_this_setting, 1);
    normed_closest_distance_to_obs_per_trained_setting_cell{i,1}    	= zeros(N_demo_this_setting, 1);
    normed_closest_distance_to_obs_per_trained_setting_idx_cell{i,1}    = zeros(N_demo_this_setting, 1);
    final_distance_to_goal_vector_per_trained_setting_cell{i,1}     	= zeros(N_demo_this_setting, 1);
    for j=1:N_demo_this_setting
        tic
        disp(['   Setting #', num2str(i), ...
              ' (', num2str(i), '/', num2str(N_settings), '), Demo #', num2str(j), '/', num2str(N_demo_this_setting)]);

        if (j == 1)
            is_measuring_demo_performance_metric    = 1;
        else
            is_measuring_demo_performance_metric    = 0;
        end

        [ unroll_dataset_learned_Ct_obs_avoid.sub_Ct_target{1,i}{j,1}, ...
          global_traj_unroll_setting_cell{i,1}(1:3,j), ...
          ~, ...
          buf_normed_closest_distance_to_obs_traj_human_1st_demo_setting, ...
          buf_final_distance_to_goal_vector_human_1st_demo_setting, ...
          base_normed_closest_distance_to_obs_traj_human_1st_demo_setting, ...
          base_final_distance_to_goal_vector_human_1st_demo_setting, ...
          normed_closest_distance_to_obs_traj_per_trained_setting_cell{i,1}{j,1}, ...
          final_distance_to_goal_vector_per_trained_setting_cell{i,1}(j,1), ...
          unroll_dataset_learned_Ct_obs_avoid.sub_X{1,i}{j,1}, ...
          learning_param.pmnn ] = unrollObsAvoidViconTraj(  data_global_coord.baseline(:,1),...
                                                            data_global_coord.obs_avoid{i,2}(:,j), ...
                                                            data_global_coord.obs_avoid{i,1}, ...
                                                            data_global_coord.dt, ...
                                                            dmp_baseline_params.cart_coord{1,1}, ...
                                                            loa_feat_methods, ...
                                                            loa_feat_param, ...
                                                            learning_param, ...
                                                            unrolling_param, ...
                                                            is_measuring_demo_performance_metric );
        if ((j == 1) && (isempty(buf_normed_closest_distance_to_obs_traj_human_1st_demo_setting) == 0) && ...
            (isnan(buf_final_distance_to_goal_vector_human_1st_demo_setting) == 0))

            normed_closest_distance_to_obs_traj_baseline_demo_setting{i,1}  = base_normed_closest_distance_to_obs_traj_human_1st_demo_setting;
            final_distance_to_goal_vector_baseline_demo_setting(i,1)        = base_final_distance_to_goal_vector_human_1st_demo_setting;

            normed_closest_distance_to_obs_traj_human_1st_demo_setting{i,1} = buf_normed_closest_distance_to_obs_traj_human_1st_demo_setting;
            final_distance_to_goal_vector_human_1st_demo_setting(i,1)       = buf_final_distance_to_goal_vector_human_1st_demo_setting;
        end

        [ normed_closest_distance_to_obs_per_trained_setting_cell{i,1}(j,1), ...
          normed_closest_distance_to_obs_per_trained_setting_idx_cell{i,1}(j,1) ]   = min(normed_closest_distance_to_obs_traj_per_trained_setting_cell{i,1}{j,1});

        toc
    end
    [normed_closest_distance_to_obs_baseline_demo_setting(i,1), ...
     normed_closest_distance_to_obs_baseline_demo_setting_idx(i,1)]    = min(normed_closest_distance_to_obs_traj_baseline_demo_setting{i,1});

    [normed_closest_distance_to_obs_human_1st_demo_setting(i,1), ...
     normed_closest_distance_to_obs_human_1st_demo_setting_idx(i,1)]    = min(normed_closest_distance_to_obs_traj_human_1st_demo_setting{i,1});

    [normed_closest_distance_to_obs_overall_train_demos_per_setting(i,1), ...
     normed_closest_distance_to_obs_overall_train_demos_per_sett_idx(i,1)]    = min(normed_closest_distance_to_obs_per_trained_setting_cell{i,1});
    [worst_final_distance_to_goal_per_trained_setting(i,1), ...
     worst_final_distance_to_goal_per_trained_setting_idx(i,1)] = max(final_distance_to_goal_vector_per_trained_setting_cell{i,1});
end


[normed_closest_dist_to_obs_human_1st_demo_over_all_settings, ...
 normed_closest_dist_to_obs_human_1st_demo_over_all_settings_idx]   = min(normed_closest_distance_to_obs_human_1st_demo_setting);
[worst_final_dist_to_goal_vector_human_1st_demo_overall_settings, ...
 worst_final_dist_to_goal_vector_human_1st_demo_overall_sett_idx]   = min(final_distance_to_goal_vector_human_1st_demo_setting);

[normed_closest_distance_to_obs_over_all_trained_settings, ...
 normed_closest_distance_to_obs_over_all_trained_settings_idx]  = min(normed_closest_distance_to_obs_overall_train_demos_per_setting);
[worst_final_distance_to_goal_over_all_trained_settings, ...
 worst_final_distance_to_goal_over_all_trained_setting_idx] = max(worst_final_distance_to_goal_per_trained_setting);

save(['unroll_dataset_learned_Ct_obs_avoid.mat'],'unroll_dataset_learned_Ct_obs_avoid');
save(['global_traj_unroll_setting_cell.mat'],'global_traj_unroll_setting_cell');

% end of Unrolling Test