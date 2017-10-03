% Author: Giovanni Sutanto
% Date  : August 01, 2016

% clear  	all;
close   all;
clc;

rng(1234)

addpath('../utilities/');

[~, name] = system('hostname');
if (strcmp(name(1,1:end-1), 'gsutanto-ThinkPad-T430s') == 1)
    root_figure_path    = '/home/gsutanto/Desktop/CLMC/Publications/Humanoids16/data/';
else
    root_figure_path    = 'figures_multi/';
end

% possible performance_metric_evaluation_mode:
% performance_metric_evaluation_mode == 0: no evaluation; run just as
%                                          regular execution with selected parameters
% performance_metric_evaluation_mode == 1: evaluation on each single
%                                          obstacle avoidance demonstration
% performance_metric_evaluation_mode == 2: evaluation on multiple (selected) 
%                                          obstacle avoidance demonstrations
% performance_metric_evaluation_mode == 3: learned NN obstacle avoidance
%                                          inference replication test
performance_metric_evaluation_mode                                  = 0;

unrolling_param.is_comparing_with_cpp_implementation                = 0;

unrolling_param.is_unrolling_only_1st_demo_each_trained_settings    = 1;

% possible debugging_X_and_Ct_mode:
% debugging_X_and_Ct_mode == 0: no debugging
% debugging_X_and_Ct_mode == 1: debugging Ct only
% debugging_X_and_Ct_mode == 2: debugging both X and Ct
debugging_X_and_Ct_mode                  	= 2;

% learning_param.max_cond_number = 1000.0;   % Recommendation from Stefan (for Robotic applications)
learning_param.max_cond_number              = 5e3;
learning_param.feature_variance_threshold   = 1e-4;
learning_param.max_abs_ard_weight_threshold = 7.5e3;

if (unrolling_param.is_comparing_with_cpp_implementation)
    unrolling_param.is_unrolling_only_1st_demo_each_trained_settings    = 0;
    unrolling_param.is_plot_unrolling   = 0;
    loa_feat_methods                    = 2;
    feat_constraint_mode                = '_CONSTRAINED_';
    learning_param.N_iter_ard          	= 1;
    selected_obs_avoid_setting_numbers  = [27, 119];
    max_num_trajs_per_setting           = 2;
    debugging_X_and_Ct_mode             = 0;
else
    unrolling_param.is_plot_unrolling  	= 1;
    
    % possible loa_feat_method:
    % loa_feat_method == 0: Akshara's Humanoids'14 features (_CONSTRAINED_)
    % loa_feat_method == 1: Potential Field 2nd Dynamic Obst Avoid features
    % loa_feat_method == 2: Giovanni's Kernelized General Features v01
    % loa_feat_method == 3: Franzi's Kernelized General Features v01
    % loa_feat_method == 4: Potential Field 3rd Dynamic Obst Avoid features (have some sense of KGF too...)
    % loa_feat_method == 5: Akshara's Humanoids'14 features (_UNCONSTRAINED_)
    % loa_feat_method == 6: Potential Field 4th Dynamic Obst Avoid features (have some sense of KGF too...)
    % loa_feat_method == 7: Neural Network
    % (enclose in a cell if you want to use multiple feature types, like in the following example below ...)
%     loa_feat_methods    = {2, 4};
    loa_feat_methods    = 7;
    
    % This feat_constraint_mode is for loa_feat_method 0, 1, 2, and 4.
    % For loa_feat_method 3 and 5, this value setting for
    % feat_constraint_mode is ignored,
    % and overridden with feat_constraint_mode = '_UNCONSTRAINED_'.
    % (in utilities/initializeAllInvolvedLOAparams.m
    %  or utilities/initializeFM_KGFv01LearnObsAvoidFeatParam.m)
    feat_constraint_mode                = '_CONSTRAINED_';
    
    learning_param.N_iter_ard        	= 200;

    selected_obs_avoid_setting_numbers  = [194];
    max_num_trajs_per_setting           = 500;
end

unrolling_param.verify_NN_inference    	= 0;

learning_param.learning_constraint_mode = '_NONE_';

D               = 3;

n_rfs           = 25;   % Number of basis functions used to represent the forcing term of DMP
c_order         = 1;    % DMP is using 2nd order canonical system

%% Data Conversion/Loading

data_filepath                       = '../data/data_multi_demo_vicon_static_global_coord.mat';
unrolling_param.cpp_impl_dump_path  = '../../../../plot/dmp_coupling/learn_obs_avoid/feature_trajectory/static_obs/single_baseline/multi_demo_vicon/';

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
is_resetting_performance_evaluation     = 0;
N_settings  = size(data_global_coord.obs_avoid, 1);

if (performance_metric_evaluation_mode == 2) % evaluation on multiple (selected) demonstrations
%     loa_feat_methods_to_be_evaluated    = [0, 2, 3, 4];
    loa_feat_methods_to_be_evaluated    = [7];
    feature_threshold_mode              = 7;
    settings_selection_mode             = 1;
    is_unrolling_on_unseen_settings     = 1;
    
    N_feat_methods  = size(loa_feat_methods_to_be_evaluated, 2);
    eval_performance_metric_multi_settings  = cell(1, N_feat_methods);
    
    if (settings_selection_mode == 0) % all
        selected_obs_avoid_setting_numbers  = [1:N_settings];
    elseif (settings_selection_mode == 1) % hand-picked:
        selected_obs_avoid_setting_numbers = [10];
       % Cube : [28 31 33 34 35 36 39 42 43 46 47 48]
%         selected_obs_avoid_setting_numbers  = [19, 23, 26, 47, 49, 56, 63, ...
%                                                85, 93, 98, 119, 121, 137, 145, ...
%                                                150, 175, 190, 210, 211, 212, 217];
% Akshara's hand-selected sphere demonstrations:
        sets = [1 2 5 6 7 9 11 12 13 14 15 16 17 18 19 20 23 24 26 27 28 29 30 31 33 35 37 39 41 42 43 45 46 47 50 51 53 54 56 57 58 59 62 63 65 68];
        sets = [sets 77,79,80,83,87,88,90,92,94,97,98,100,101,102,103,104,105,107,110,119,120,121,123,134,137,138,143,152,153,154,156,160,164,172,174,175,176,177,179,182,189,193,194,195,196,197,200,212,213];

        selected_obs_avoid_setting_numbers = sets; %[1 2 5 6 7 9 11 12 13 14 15 16 17 18 19 20 23 24 26 27 28 29 30 31 33 35 37 39 41 42 43 45 46 47 50 51 53 54 56 57 58 59 62 63 65 68];
    elseif (settings_selection_mode == 2) % based on annotated obstacle avoidance's dominant axis:
        data_annotation_obs_avoid_dominant_axis = cell2mat(data_global_coord.obs_avoid(:,3));
        idx_no_dom_axis = find(data_annotation_obs_avoid_dominant_axis == 0);
        idx_y_dom_axis  = find(data_annotation_obs_avoid_dominant_axis == 2);
        idx_z_dom_axis  = find(data_annotation_obs_avoid_dominant_axis == 3);

        idx_w_dom_axis  = union(idx_y_dom_axis, idx_z_dom_axis);
    %     num_no_dom_axis_demo    = 5;
    %     idx_randperm_no_dom_axis= randperm(length(idx_no_dom_axis));
    %     selected_idx_no_dom_axis= idx_no_dom_axis(idx_randperm_no_dom_axis(1,1:num_no_dom_axis_demo),:);
        selected_idx_no_dom_axis = [];

        selected_obs_avoid_setting_numbers  = union(idx_w_dom_axis, selected_idx_no_dom_axis).';
    elseif (settings_selection_mode == 3) % based on annotated obstacle avoidance demonstration's consistency:
        data_annotation_obs_avoid_consistency   = cell2mat(data_global_coord.obs_avoid(:,4));
        selected_obs_avoid_setting_numbers  = find(data_annotation_obs_avoid_consistency == 1).';
    end
    max_num_trajs_per_setting   = 10;
    if (strcmp(name(1,1:end-1), 'gsutanto-ThinkPad-T430s') == 1)
        figure_path            	= [root_figure_path, '/multi_settings_learning_plot/'];
    else
        figure_path            	= 'figures_multi/';
    end

    for j=1:N_feat_methods
        disp(['>>>Feature Method #', num2str(loa_feat_methods_to_be_evaluated(1, j)), ' (', num2str(j), '/', num2str(N_feat_methods), ') ...']);
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
                                                                      figure_path, is_unrolling_on_unseen_settings, ...
                                                                      feature_threshold_mode );
        eval_performance_metric_multi_settings{1, j}   = performance_metric;
    end

    save('eval_performance_metric_multi_settings.mat', 'eval_performance_metric_multi_settings');

    copyfile('eval_performance_metric_multi_settings.mat', ...
             root_figure_path);
elseif (performance_metric_evaluation_mode == 1) % evaluation on each single demonstration
%     loa_feat_methods_to_be_evaluated    = [0, 2, 3, 4];
    loa_feat_methods_to_be_evaluated    = [7];
    feature_threshold_mode              = 7;
    
    N_feat_methods  = size(loa_feat_methods_to_be_evaluated, 2);
    if ((is_resetting_performance_evaluation) || (exist('eval_performance_metric_each_single_setting.mat', 'file') == 0))
        eval_performance_metric_each_single_setting = cell(N_settings, N_feat_methods);
        idx_start_setting               = 1;
    else
        load('eval_performance_metric_each_single_setting.mat');
        idx_start_setting               = 1;
        while (isempty(eval_performance_metric_each_single_setting{idx_start_setting,1}) == 0)
            idx_start_setting           = idx_start_setting + 1;
            if (idx_start_setting > N_settings)
                break;
            end
        end
        disp(['eval_performance_metric_each_single_setting.mat file exists. Resuming from Setting #', num2str(idx_start_setting), '...']);
    end
    
    if (strcmp(name(1,1:end-1), 'gsutanto-ThinkPad-T430s') == 1)
        figure_path = [root_figure_path, '/single_setting_learning_plot/'];
    else
        figure_path = 'figures_single/';
    end
    
    test_settings  = [8, 10, 26, 27, 28,29,30,31,44,45,46,47,83,84,98,99,100,102,116,117,118,119,120,122,136,137,157,172,173,174,175,176,177,178,190,193,194,195,197,209,210]; %[26, 47, 85, 98, 119, 190, 210, 217];

%     for i=idx_start_setting:N_settings
    for idx=1:length(test_settings)
        i = test_settings(idx);
        disp(['>>Evaluating Performance Metric for Setting #', num2str(i), '/', num2str(N_settings), ' ...']);
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
                                                                          figure_path, 0, ...
                                                                          feature_threshold_mode );
            eval_performance_metric_each_single_setting{i, j}   = performance_metric;
        end
        
        if ((mod(i,3) == 0) || (i == N_settings))  % save data everytime 3 settings of evaluations are completed
            save('eval_performance_metric_each_single_setting.mat', 'eval_performance_metric_each_single_setting');
            
            if ((mod(i,15) == 0) || (i == N_settings))
                copyfile('eval_performance_metric_each_single_setting.mat', ...
                         root_figure_path);
            end
        end
    end
elseif (performance_metric_evaluation_mode == 0) % regular execution
    if (unrolling_param.is_comparing_with_cpp_implementation == 1)
        feature_threshold_mode  = 3;
    else
        feature_threshold_mode  = 7;
    end
    [ performance_metric, ...
      learning_unrolling_variables, ...
      learning_param ] = learnAndUnrollObsAvoidViconDataSetting( data_global_coord, ...
                                                                 cart_coord_dmp_baseline_params, ...
                                                                 loa_feat_methods, ...
                                                                 feat_constraint_mode, ...
                                                                 learning_param, ...
                                                                 unrolling_param, ...
                                                                 selected_obs_avoid_setting_numbers, ...
                                                                 max_num_trajs_per_setting, ...
                                                                 '', 0, ...
                                                                 feature_threshold_mode, ...
                                                                 debugging_X_and_Ct_mode );
elseif (performance_metric_evaluation_mode == 3) % learned NN obstacle avoidance inference replication test
    unrolling_param.is_unrolling_only_1st_demo_each_trained_settings    = 1;
    unrolling_param.is_plot_unrolling   = 0;
    unrolling_param.verify_NN_inference = 1;
    loa_feat_methods                    = 7;
    feat_constraint_mode                = '_UNCONSTRAINED_';
    feature_threshold_mode              = 7;
    selected_obs_avoid_setting_numbers  = [8];
    max_num_trajs_per_setting           = 10;
    
    [ performance_metric, ...
      learning_unrolling_variables, ...
      learning_param ] = learnAndUnrollObsAvoidViconDataSetting( data_global_coord, ...
                                                                 cart_coord_dmp_baseline_params, ...
                                                                 loa_feat_methods, ...
                                                                 feat_constraint_mode, ...
                                                                 learning_param, ...
                                                                 unrolling_param, ...
                                                                 selected_obs_avoid_setting_numbers, ...
                                                                 max_num_trajs_per_setting, ...
                                                                 '', 0, ...
                                                                 feature_threshold_mode, ...
                                                                 debugging_X_and_Ct_mode );
end

% end of Learning and Unrolling (and Evaluating Performance Metric)