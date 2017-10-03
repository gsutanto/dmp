clear all;
close all;
clc;

task_type                   = 'scraping';

amd_clmc_dmp_root_dir_path  = '../../../../';

data_root_dir_path          = [amd_clmc_dmp_root_dir_path, 'data/'];
matlab_root_dir_path        = [amd_clmc_dmp_root_dir_path, 'matlab/'];
python_root_dir_path        = [amd_clmc_dmp_root_dir_path, 'python/'];

data_learn_tactile_fb_task_type_dir_path             = [data_root_dir_path, 'dmp_coupling/learn_tactile_feedback/',task_type,'/'];
data_LTacFB_task_type_PRBFN_dir_path                 = [data_learn_tactile_fb_task_type_dir_path, 'neural_nets/FFNNFinalPhaseLWRLayerPerDims/'];
data_LTacFB_task_type_PRBFN_unroll_test_dir_path     = [data_LTacFB_task_type_PRBFN_dir_path, 'unroll_test_dataset/'];
data_LTacFB_task_type_PRBFN_python_models_dir_path   = [data_LTacFB_task_type_PRBFN_dir_path, 'python_models/'];
data_LTacFB_task_type_PRBFN_cpp_models_dir_path      = [data_LTacFB_task_type_PRBFN_dir_path, 'cpp_models/'];

matlab_learn_tactile_fb_dir_path	= [matlab_root_dir_path, 'dmp_coupling/learn_tactile_feedback/'];

python_learn_tactile_fb_dir_path            = [python_root_dir_path, 'dmp_coupling/learn_tactile_feedback/'];
python_learn_tactile_fb_task_type_dir_path 	= [python_learn_tactile_fb_dir_path, task_type, '/'];
python_learn_tactile_fb_models_dir_path     = [python_learn_tactile_fb_dir_path, 'models/'];

addpath([matlab_root_dir_path, 'utilities/']);
addpath([matlab_root_dir_path, 'neural_nets/feedforward/with_final_phaseLWR_layer/per_dimensions/FFNNFinalPhaseLWRLayerPerDims/']);

recreateDir(data_LTacFB_task_type_PRBFN_unroll_test_dir_path);
recreateDir(data_LTacFB_task_type_PRBFN_python_models_dir_path);
recreateDir(data_LTacFB_task_type_PRBFN_cpp_models_dir_path);

reinit_selection_idx    = dlmread([python_learn_tactile_fb_models_dir_path, 'reinit_selection_idx.txt']);
TF_max_train_iters      = dlmread([python_learn_tactile_fb_models_dir_path, 'TF_max_train_iters.txt']);
N_prims                 = size(reinit_selection_idx, 2);

precision_string     	= '%.20f';

%% Copy Nominal DMPs Parameters (*.mat File)

copyfile([matlab_learn_tactile_fb_dir_path,'dmp_baseline_params_',task_type,'.mat'],...
         data_learn_tactile_fb_task_type_dir_path);

%% Convert Nominal DMPs Parameters into *.txt Files (for Loading by C++ Programs)

load([data_learn_tactile_fb_task_type_dir_path, 'dmp_baseline_params_',task_type,'.mat']);

% param logging for C++ synchronization:
prims_param_root_dir_path   = [data_learn_tactile_fb_task_type_dir_path, 'learned_prims_params/'];
recreateDir(prims_param_root_dir_path);

% position (CartCoordDMP)
prims_param_type_dir_path       = [prims_param_root_dir_path, 'position/'];
recreateDir(prims_param_type_dir_path);
for n_prim = 1:N_prims
    prims_param_type_prim_dir_path   = [prims_param_type_dir_path, 'prim', num2str(n_prim), '/'];
    recreateDir(prims_param_type_prim_dir_path);
    dlmwrite([prims_param_type_prim_dir_path, 'w'],                     dmp_baseline_params.cart_coord{n_prim, 1}.w.',                  'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'A_learn'],               dmp_baseline_params.cart_coord{n_prim, 1}.dG.',                 'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'start_global'],          dmp_baseline_params.cart_coord{n_prim, 1}.mean_start_global,    'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'goal_global'],           dmp_baseline_params.cart_coord{n_prim, 1}.mean_goal_global,     'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'start_local'],           dmp_baseline_params.cart_coord{n_prim, 1}.mean_start_local,     'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'goal_local'],            dmp_baseline_params.cart_coord{n_prim, 1}.mean_goal_local,      'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'T_local_to_global_H'],   dmp_baseline_params.cart_coord{n_prim, 1}.T_local_to_global_H, 	'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'T_global_to_local_H'],   dmp_baseline_params.cart_coord{n_prim, 1}.T_global_to_local_H, 	'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'tau'],                   dmp_baseline_params.cart_coord{n_prim, 1}.mean_tau,             'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'canonical_sys_order'],   dmp_baseline_params.cart_coord{n_prim, 1}.c_order+1,            'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'ctraj_local_coordinate_frame_selection'],	dmp_baseline_params.cart_coord{n_prim, 1}.ctraj_local_coordinate_frame_selection,   'delimiter', ' ', 'precision', precision_string);
end

% orientation (QuaternionDMP)
prims_param_type_dir_path       = [prims_param_root_dir_path, 'orientation/'];
recreateDir(prims_param_type_dir_path);
for n_prim = 1:N_prims
    prims_param_type_prim_dir_path   = [prims_param_type_dir_path, 'prim', num2str(n_prim), '/'];
    recreateDir(prims_param_type_prim_dir_path);
    dlmwrite([prims_param_type_prim_dir_path, 'w'],                     dmp_baseline_params.Quat{n_prim, 1}.w.',            'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'A_learn'],               dmp_baseline_params.Quat{n_prim, 1}.dG,             'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'start'],                 dmp_baseline_params.Quat{n_prim, 1}.fit_mean_Q0,    'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'goal'],                  dmp_baseline_params.Quat{n_prim, 1}.fit_mean_QG,    'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'tau'],                   dmp_baseline_params.Quat{n_prim, 1}.fit_mean_tau,   'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'canonical_sys_order'],	dmp_baseline_params.Quat{n_prim, 1}.c_order+1,      'delimiter', ' ', 'precision', precision_string);
end

% sense_R_LF
prims_param_type_dir_path       = [prims_param_root_dir_path, 'sense_R_LF/'];
recreateDir(prims_param_type_dir_path);
for n_prim = 1:N_prims
    prims_param_type_prim_dir_path   = [prims_param_type_dir_path, 'prim', num2str(n_prim), '/'];
    recreateDir(prims_param_type_prim_dir_path);
    dlmwrite([prims_param_type_prim_dir_path, 'w'],                     dmp_baseline_params.BT_electrode{n_prim, 1}.w.',        'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'A_learn'],               zeros(19, 1),                                           'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'start'],                 dmp_baseline_params.BT_electrode{n_prim, 1}.mean_start, 'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'goal'],                  dmp_baseline_params.BT_electrode{n_prim, 1}.mean_goal,  'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'tau'],                   dmp_baseline_params.BT_electrode{n_prim, 1}.mean_tau,   'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'canonical_sys_order'],	dmp_baseline_params.BT_electrode{n_prim, 1}.c_order+1,  'delimiter', ' ', 'precision', precision_string);
end

% sense_R_RF
prims_param_type_dir_path       = [prims_param_root_dir_path, 'sense_R_RF/'];
recreateDir(prims_param_type_dir_path);
for n_prim = 1:N_prims
    prims_param_type_prim_dir_path   = [prims_param_type_dir_path, 'prim', num2str(n_prim), '/'];
    recreateDir(prims_param_type_prim_dir_path);
    dlmwrite([prims_param_type_prim_dir_path, 'w'],                     dmp_baseline_params.BT_electrode{n_prim, 2}.w.',        'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'A_learn'],               zeros(19, 1),                                           'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'start'],                 dmp_baseline_params.BT_electrode{n_prim, 2}.mean_start, 'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'goal'],                  dmp_baseline_params.BT_electrode{n_prim, 2}.mean_goal,  'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'tau'],                   dmp_baseline_params.BT_electrode{n_prim, 2}.mean_tau,   'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'canonical_sys_order'],	dmp_baseline_params.BT_electrode{n_prim, 2}.c_order+1,  'delimiter', ' ', 'precision', precision_string);
end

% sense_proprio
prims_param_type_dir_path       = [prims_param_root_dir_path, 'sense_proprio/'];
recreateDir(prims_param_type_dir_path);
for n_prim = 1:N_prims
    prims_param_type_prim_dir_path   = [prims_param_type_dir_path, 'prim', num2str(n_prim), '/'];
    recreateDir(prims_param_type_prim_dir_path);
    dlmwrite([prims_param_type_prim_dir_path, 'w'],                     dmp_baseline_params.joint_sense{n_prim, 1}.w.',         'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'A_learn'],               zeros(7, 1),                                            'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'start'],                 dmp_baseline_params.joint_sense{n_prim, 1}.mean_start,  'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'goal'],                  dmp_baseline_params.joint_sense{n_prim, 1}.mean_goal,   'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'tau'],                   dmp_baseline_params.joint_sense{n_prim, 1}.mean_tau,    'delimiter', ' ', 'precision', precision_string);
    dlmwrite([prims_param_type_prim_dir_path, 'canonical_sys_order'],	dmp_baseline_params.joint_sense{n_prim, 1}.c_order+1,   'delimiter', ' ', 'precision', precision_string);
end

%% Convert Unroll Test Dataset for Primitive 1 (for Comparing between Prediction Made in Python TensorFlow versus in MATLAB)

load([python_learn_tactile_fb_task_type_dir_path, 'test_unroll_prim_1_X_raw_',task_type,'.mat']);
load([python_learn_tactile_fb_task_type_dir_path, 'test_unroll_prim_1_normalized_phase_PSI_mult_phase_V_',task_type,'.mat']);
load([python_learn_tactile_fb_task_type_dir_path, 'test_unroll_prim_1_Ct_target_',task_type,'.mat']);

dlmwrite([data_LTacFB_task_type_PRBFN_unroll_test_dir_path, 'test_unroll_prim_1_X_raw_',task_type,'.txt'], X, 'delimiter', ' ', 'precision', precision_string);
dlmwrite([data_LTacFB_task_type_PRBFN_unroll_test_dir_path, 'test_unroll_prim_1_normalized_phase_PSI_mult_phase_V_',task_type,'.txt'], normalized_phase_PSI_mult_phase_V, 'delimiter', ' ', 'precision', precision_string);
dlmwrite([data_LTacFB_task_type_PRBFN_unroll_test_dir_path, 'test_unroll_prim_1_Ct_target_',task_type,'.txt'], Ct_target, 'delimiter', ' ', 'precision', precision_string);

%% Copy and Convert FFNNFinalPhaseLWRPerDims (or PRBFN) Learned Parameters from *.mat (Python TensorFlow Result) to *.txt (for Loading by C++ Programs) Files

NN_name             = 'my_ffNNphaseLWR';

for np=1:N_prims
    param_filename  = ['prim_',num2str(np),'_params_reinit_',num2str(reinit_selection_idx(1,np)),'_step_',num2str(TF_max_train_iters,'%07d'),'.mat'];
    nmse_filename   = ['prim_',num2str(np),'_nmse_reinit_',num2str(reinit_selection_idx(1,np)),'_step_',num2str(TF_max_train_iters,'%07d'),'.mat'];
    var_gt_filename = ['prim_',num2str(np),'_var_ground_truth.mat'];

    % Copy FFNNFinalPhaseLWRPerDims (or PRBFN) Learned Parameters:
    copyfile([python_learn_tactile_fb_models_dir_path, param_filename], ...
             data_LTacFB_task_type_PRBFN_python_models_dir_path);

    % Copy FFNNFinalPhaseLWRPerDims (or PRBFN) Learning NMSEs:
    copyfile([python_learn_tactile_fb_models_dir_path, nmse_filename], ...
             data_LTacFB_task_type_PRBFN_python_models_dir_path);

    % Copy FFNNFinalPhaseLWRPerDims (or PRBFN) Ground-Truth Variance:
    copyfile([python_learn_tactile_fb_models_dir_path, var_gt_filename], ...
             data_LTacFB_task_type_PRBFN_python_models_dir_path);
    
    % Convert FFNNFinalPhaseLWRPerDims (or PRBFN) Learned Parameters:
    mat_filepath= [data_LTacFB_task_type_PRBFN_python_models_dir_path, param_filename];
    out_dirpath = [data_LTacFB_task_type_PRBFN_cpp_models_dir_path, 'prim', num2str(np)];
    convertFFNNFinalPhaseLWRPerDimsParamsFromMatFile2TxtFiles( NN_name, 6, mat_filepath, out_dirpath );
end

%% Test Dataset (with Particular Selection of Setting and Trial Number) Logging into *.txt Files (for Comparison between MATLAB and C++ Execution)

load([matlab_learn_tactile_fb_dir_path, 'data_demo_',task_type,'.mat']);

outdata_root_dir_path	= [data_learn_tactile_fb_task_type_dir_path, 'unroll_test_dataset/all_prims/'];

setting_no  = 5;
trial_no    = 3;

trial_outdata_dir_path  = [outdata_root_dir_path, 'setting_', num2str(setting_no), '_trial_', num2str(trial_no), '/'];
createDirIfNotExist(trial_outdata_dir_path);

unroll_test_data.position   = cell(N_prims, 1);
unroll_test_data.orientation= cell(N_prims, 1);
unroll_test_data.sense      = cell(N_prims, 3);

% data binning and logging for C++ synchronization:
for n_prim = 1:N_prims
    trial_prim_outdata_dir_path             = [trial_outdata_dir_path, 'prim', num2str(n_prim), '/'];
    recreateDir(trial_prim_outdata_dir_path);
    unroll_test_data.position{n_prim, 1}	= data_demo.coupled{n_prim, setting_no}{trial_no,  2};
    dlmwrite([trial_prim_outdata_dir_path, '/position'], unroll_test_data.position{n_prim, 1}, 'delimiter', ' ', 'precision', precision_string);
    unroll_test_data.orientation{n_prim, 1}	= data_demo.coupled{n_prim, setting_no}{trial_no,  3};
    dlmwrite([trial_prim_outdata_dir_path, '/orientation'], unroll_test_data.orientation{n_prim, 1}, 'delimiter', ' ', 'precision', precision_string);
    unroll_test_data.sense{n_prim, 1}       = data_demo.coupled{n_prim, setting_no}{trial_no,  6};  % 1==BioTac R_LF_electrodes (Tactile Right Finger)
    dlmwrite([trial_prim_outdata_dir_path, '/sense_R_LF'], unroll_test_data.sense{n_prim, 1}, 'delimiter', ' ', 'precision', precision_string);
    unroll_test_data.sense{n_prim, 2}       = data_demo.coupled{n_prim, setting_no}{trial_no, 14};  % 2==BioTac R_RF_electrodes (Tactile Right Finger)
    dlmwrite([trial_prim_outdata_dir_path, '/sense_R_RF'], unroll_test_data.sense{n_prim, 2}, 'delimiter', ' ', 'precision', precision_string);
    unroll_test_data.sense{n_prim, 3}       = data_demo.coupled{n_prim, setting_no}{trial_no, 22};  % 3==Joint Positions/Coordinates (Proprioception)
    dlmwrite([trial_prim_outdata_dir_path, '/sense_proprio'], unroll_test_data.sense{n_prim, 3}, 'delimiter', ' ', 'precision', precision_string);
end