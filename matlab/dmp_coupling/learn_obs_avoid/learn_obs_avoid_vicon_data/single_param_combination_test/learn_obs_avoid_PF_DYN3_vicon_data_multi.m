% Author        : Giovanni Sutanto
% Date          : August 01, 2016
% Description   : evaluation on multiple (selected) 
%                 obstacle avoidance demonstrations

clear  	all;
close   all;
clc;

addpath('../../utilities/');
addpath('../');

unrolling_param.is_comparing_with_cpp_implementation    = 0;

% learning_param.max_cond_number = 1000.0;   % Recommendation from Stefan (for Robotic applications)
learning_param.max_cond_number              = 5e3;

learning_param.feature_variance_threshold   = 1e-4;

learning_param.max_abs_ard_weight_threshold = 7.5e3;

unrolling_param.is_plot_unrolling  	= 0;

% possible loa_feat_method:
% loa_feat_method == 0: Akshara's Humanoids'14 features (_CONSTRAINED_)
% loa_feat_method == 1: Potential Field 2nd Dynamic Obst Avoid features
% loa_feat_method == 2: Giovanni's Kernelized General Features v01
% loa_feat_method == 3: Franzi's Kernelized General Features v01
% loa_feat_method == 4: Potential Field 3rd Dynamic Obst Avoid features (have some sense of KGF too...)
% loa_feat_method == 5: Akshara's Humanoids'14 features (_UNCONSTRAINED_)
% (enclose in a cell if you want to use multiple feature types, like in the following example below ...)
% loa_feat_methods    = {2, 4};
loa_feat_methods    = 4;

% This feat_constraint_mode is for loa_feat_method 0, 1, 2, and 4.
% For loa_feat_method 3 and 5, this value setting for
% feat_constraint_mode is ignored,
% and overridden with feat_constraint_mode = '_UNCONSTRAINED_'.
% (in utilities/initializeAllInvolvedLOAparams.m
%  or utilities/initializeFM_KGFv01LearnObsAvoidFeatParam.m)
feat_constraint_mode                = '_CONSTRAINED_';

learning_param.N_iter_ard        	= 200;

% if ((strcmp(feat_constraint_mode, '_CONSTRAINED_') == 1) && ~(isMemberLOAFeatMethods(3, loa_feat_methods)))
%     learning_param.learning_constraint_mode   = '_PER_AXIS_';
% else
%     learning_param.learning_constraint_mode   = '_NONE_';
% end
learning_param.learning_constraint_mode     = '_NONE_';

D               = 3;

n_rfs           = 25;   % Number of basis functions used to represent the forcing term of DMP
c_order         = 1;    % DMP is using 2nd order canonical system

%% Data Creation/Loading

data_filepath                       = '../../data/data_multi_demo_vicon_static_global_coord.mat';
unrolling_param.cpp_impl_dump_path  = '../../../../../plot/dmp_coupling/learn_obs_avoid/feature_trajectory/static_obs/single_baseline/multi_demo_vicon/';

% if input file is not exist yet, then create one (convert from C++ textfile format):
if (~exist(data_filepath, 'file'))
    convert_loa_vicon_data_to_mat_format;
end

load(data_filepath);

% end of Data Creation/Loading

%% Baseline Primitive Learning

disp('Processing Local Coordinate System for Demonstrated Baseline Trajectories ...');
[ cart_coord_dmp_baseline_params, ...
  unrolling_param.cart_coord_dmp_baseline_unroll_global_traj ] = learnCartPrimitiveMultiOnLocalCoord(data_global_coord.baseline, data_global_coord.dt, n_rfs, c_order);

% end of Baseline Primitive Learning

%% Learning and Unrolling (and Evaluating Performance Metric)

is_resetting_performance_evaluation = 0;

% loa_feat_methods_to_be_evaluated    = [0, 2, 3, 4];
loa_feat_methods_to_be_evaluated    = [4];
feature_threshold_mode              = 8;

N_feat_methods  = size(loa_feat_methods_to_be_evaluated, 2);
eval_performance_metric_multi_settings  = cell(1, N_feat_methods);

% selected_obs_avoid_setting_numbers  = [19, 23, 26, 47, 49, 56, 63, ...
%                                        85, 93, 98, 119, 121, 137, 145, ...
%                                        150, 175, 190, 210, 211, 212, 217];
%     selected_obs_avoid_setting_numbers  = [26, 47, 85, 98, 119, 190, 210, 217];
selected_obs_avoid_setting_numbers  = [7];
max_num_trajs_per_setting           = 5;
figure_path                         = '/home/gsutanto/Desktop/CLMC/Publications/Humanoids16/data/single_param_combination_test/multi_settings_learning_plot/';

for j=1:N_feat_methods
    disp(['>>>Feature Method #', num2str(loa_feat_methods_to_be_evaluated(1, j)), ' (', num2str(j), '/', num2str(N_feat_methods), ') ...']);
    [ performance_metric, learning_unrolling_variables ] = learnAndUnrollObsAvoidPF_DYN3ViconDataSetting( data_global_coord, ...
                                                                                                          cart_coord_dmp_baseline_params, ...
                                                                                                          loa_feat_methods_to_be_evaluated(1, j), ...
                                                                                                          feat_constraint_mode, ...
                                                                                                          learning_param, ...
                                                                                                          unrolling_param, ...
                                                                                                          1, 0.0, 0.0, ...
                                                                                                          1, 0.01, 0.01, ...
                                                                                                          1, 0.5, 0.5, ...
                                                                                                          1.0, 1.0, ...
                                                                                                          selected_obs_avoid_setting_numbers, ...
                                                                                                          max_num_trajs_per_setting, ...
                                                                                                          figure_path, 0, ...
                                                                                                          feature_threshold_mode );
    eval_performance_metric_multi_settings{1, j}   = performance_metric;
end

save('eval_performance_metric_multi_settings.mat', 'eval_performance_metric_multi_settings');

copyfile('eval_performance_metric_multi_settings.mat', ...
         '/home/gsutanto/Desktop/CLMC/Publications/Humanoids16/data/single_param_combination_test/');

% end of Learning and Unrolling (and Evaluating Performance Metric)