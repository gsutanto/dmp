% Author: Giovanni Sutanto
% Date  : August 01, 2016

% clear  	all;
close   all;
clc;

rng(1234)

addpath('../utilities/');

root_figure_path = 'figures_single';
% possible performance_metric_evaluation_mode:
% performance_metric_evaluation_mode == 0: no evaluation; run just as
%                                          regular execution with selected parameters
% performance_metric_evaluation_mode == 1: evaluation on each single
%                                          obstacle avoidance demonstration
% performance_metric_evaluation_mode == 2: evaluation on multiple (selected)
%                                          obstacle avoidance demonstrations
performance_metric_evaluation_mode                                  = 1;
unrolling_param.is_comparing_with_cpp_implementation                = 0;
unrolling_param.is_unrolling_only_1st_demo_each_trained_settings    = 1;

% possible debugging_X_and_Ct_mode:
% debugging_X_and_Ct_mode == 0: no debugging
% debugging_X_and_Ct_mode == 1: debugging Ct only
% debugging_X_and_Ct_mode == 2: debugging both X and Ct
debugging_X_and_Ct_mode                  	= 0;

% learning_param.max_cond_number = 1000.0;   % Recommendation from Stefan (for Robotic applications)
learning_param.max_cond_number              = 5e3;
learning_param.feature_variance_threshold   = 1e-4;
learning_param.max_abs_ard_weight_threshold = 7.5e3;

unrolling_param.is_plot_unrolling  	= 1;

% This feat_constraint_mode is for loa_feat_method 0, 1, 2, and 4.
% For loa_feat_method 3 and 5, this value setting for
% feat_constraint_mode is ignored,
% and overridden with feat_constraint_mode = '_UNCONSTRAINED_'.
% (in utilities/initializeAllInvolvedLOAparams.m
%  or utilities/initializeFM_KGFv01LearnObsAvoidFeatParam.m)
feat_constraint_mode                = '_CONSTRAINED_';
learning_param.learning_constraint_mode     = '_NONE_';

learning_param.N_iter_ard        	= 200;
max_num_trajs_per_setting           = 500;
% possible loa_feat_method:
% loa_feat_method == 0: Akshara's Humanoids'14 features (_CONSTRAINED_)
% loa_feat_method == 5: Akshara's Humanoids'14 features (_UNCONSTRAINED_)
% loa_feat_method == 7: Neural Network
loa_feat_methods_to_be_evaluated = 7;

D               = 3;

n_rfs           = 25;   % Number of basis functions used to represent the forcing term of DMP
c_order         = 1;    % DMP is using 2nd order canonical system

%% Data Creation/Loading

data_filepath                       = '../data/data_multi_demo_vicon_static_global_coord.mat';
load(data_filepath);

% end of Data Creation/Loading
%% Baseline Primitive Learning
disp('Processing Local Coordinate System for Demonstrated Baseline Trajectories ...');
[ cart_coord_dmp_baseline_params, ...
    unrolling_param.cart_coord_dmp_baseline_unroll_global_traj ] = learnCartPrimitiveMultiOnLocalCoord(data_global_coord.baseline, data_global_coord.dt, n_rfs, c_order);

% end of Baseline Primitive Learning

%% Learning and Unrolling (and Evaluating Performance Metric)
is_resetting_performance_evaluation     = 0;
feature_threshold_mode              = 7;
N_feat_methods  = size(loa_feat_methods_to_be_evaluated, 2);

settings = [8, 10,26, 27, 28,29,30,31,44,45,46,47,83,84,98,99,100,102,...
    116,117,118,119,120,122,136,137,157,172,173,174,175,176,177,178,...
    190,193,194,195,197,209,210];
% N_settings  = size(data_global_coord.obs_avoid, 1);
%settings = idx_start_setting:N_settings;
for i=settings
    disp(['>>Evaluating Performance Metric for Setting #', num2str(i), '/', num2str(length(settings)), ' ...']);
    selected_obs_avoid_setting_numbers  = [i];
    max_num_trajs_per_setting           = 10;
    
    for j=1:N_feat_methods
        disp(['>>>>>Feature Method #', num2str(loa_feat_methods_to_be_evaluated(1, j)), ' (', num2str(j), '/', num2str(N_feat_methods), ') ...']);
        [ performance_metric, ...
            learning_unrolling_variables, ...
            learning_param ]  = learnAndUnrollObsAvoidViconDataSetting( data_global_coord, ...
            cart_coord_dmp_baseline_params, ...
            loa_feat_methods_to_be_evaluated(1, j), ...
            feat_constraint_mode, ...
            learning_param, ...
            unrolling_param, ...
            selected_obs_avoid_setting_numbers, ...
            max_num_trajs_per_setting, ...
            root_figure_path, 0, ...
            feature_threshold_mode );
        eval_performance_metric_each_single_setting{i, j}   = performance_metric;
    end
    
    if ((mod(i,3) == 0) || (i == length(settings)))  % save data everytime 3 settings of evaluations are completed
        save('eval_performance_metric_each_single_setting.mat', 'eval_performance_metric_each_single_setting');
    end
end

% end of Learning and Unrolling (and Evaluating Performance Metric)