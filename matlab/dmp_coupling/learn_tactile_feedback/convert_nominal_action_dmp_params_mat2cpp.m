clear all;
close all;
clc;

specific_task_type    	= 'scraping_w_tool';

dmp_root_dir_path  = '../../../';

data_root_dir_path      = [dmp_root_dir_path, 'data/'];
matlab_root_dir_path  	= [dmp_root_dir_path, 'matlab/'];

data_learn_tactile_fb_specific_task_type_dir_path = [data_root_dir_path, 'dmp_coupling/learn_tactile_feedback/',specific_task_type,'/'];

matlab_learn_tactile_fb_dir_path    = [matlab_root_dir_path, 'dmp_coupling/learn_tactile_feedback/'];

addpath([matlab_root_dir_path, 'utilities/']);

N_prims                 = 3;

precision_string     	= '%.20f';

%% Copy Nominal DMPs Parameters (*.mat File)

copyfile([matlab_learn_tactile_fb_dir_path,'action_dmp_baseline_params_',specific_task_type,'.mat'],...
         data_learn_tactile_fb_specific_task_type_dir_path);

%% Convert Nominal DMPs Parameters into *.txt Files (for Loading by C++ Programs)

load([data_learn_tactile_fb_specific_task_type_dir_path, 'action_dmp_baseline_params_',specific_task_type,'.mat']);

% param logging for C++ synchronization:
prims_param_root_dir_path   = [data_learn_tactile_fb_specific_task_type_dir_path, 'learned_prims_params/'];
recreateDir(prims_param_root_dir_path);

% position (CartCoordDMP)
prims_param_type_dir_path       = [prims_param_root_dir_path, 'position/'];
recreateDir(prims_param_type_dir_path);
for n_prim = 1:N_prims
    prims_param_type_prim_dir_path   = [prims_param_type_dir_path, 'prim', num2str(n_prim), '/'];
    recreateDir(prims_param_type_prim_dir_path);
    dlmwrite([prims_param_type_prim_dir_path, 'w'],                     action_dmp_baseline_params.cart_coord{n_prim, 1}.w.',                  'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'A_learn'],               action_dmp_baseline_params.cart_coord{n_prim, 1}.dG.',                 'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'start_global'],          action_dmp_baseline_params.cart_coord{n_prim, 1}.mean_start_global,    'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'goal_global'],           action_dmp_baseline_params.cart_coord{n_prim, 1}.mean_goal_global,     'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'start_local'],           action_dmp_baseline_params.cart_coord{n_prim, 1}.mean_start_local,     'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'goal_local'],            action_dmp_baseline_params.cart_coord{n_prim, 1}.mean_goal_local,      'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'T_local_to_global_H'],   action_dmp_baseline_params.cart_coord{n_prim, 1}.T_local_to_global_H, 	'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'T_global_to_local_H'],   action_dmp_baseline_params.cart_coord{n_prim, 1}.T_global_to_local_H, 	'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'tau'],                   action_dmp_baseline_params.cart_coord{n_prim, 1}.mean_tau,             'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'canonical_sys_order'],   action_dmp_baseline_params.cart_coord{n_prim, 1}.c_order+1,            'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'ctraj_local_coordinate_frame_selection'],	action_dmp_baseline_params.cart_coord{n_prim, 1}.ctraj_local_coordinate_frame_selection,   'delimiter', ' ', 'precision', precision_string);
end

% orientation (QuaternionDMP)
prims_param_type_dir_path       = [prims_param_root_dir_path, 'orientation/'];
recreateDir(prims_param_type_dir_path);
for n_prim = 1:N_prims
    prims_param_type_prim_dir_path   = [prims_param_type_dir_path, 'prim', num2str(n_prim), '/'];
    recreateDir(prims_param_type_prim_dir_path);
    dlmwrite([prims_param_type_prim_dir_path, 'w'],                     action_dmp_baseline_params.Quat{n_prim, 1}.w.',            'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'A_learn'],               action_dmp_baseline_params.Quat{n_prim, 1}.dG,             'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'start'],                 action_dmp_baseline_params.Quat{n_prim, 1}.fit_mean_Q0,    'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'goal'],                  action_dmp_baseline_params.Quat{n_prim, 1}.fit_mean_QG,    'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'tau'],                   action_dmp_baseline_params.Quat{n_prim, 1}.fit_mean_tau,   'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'canonical_sys_order'],	action_dmp_baseline_params.Quat{n_prim, 1}.c_order+1,      'delimiter', ' ', 'precision', precision_string);
end