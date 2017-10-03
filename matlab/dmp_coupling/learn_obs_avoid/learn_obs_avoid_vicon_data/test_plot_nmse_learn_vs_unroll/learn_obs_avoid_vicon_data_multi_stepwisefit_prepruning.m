% Author: Giovanni Sutanto
% Date  : August 01, 2016

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

if (unrolling_param.is_comparing_with_cpp_implementation)
    unrolling_param.is_plot_unrolling   = 0;
    loa_feat_methods                    = 2;
    feat_constraint_mode                = '_CONSTRAINED_';
    learning_param.N_iter_ard          	= 1;
else
    unrolling_param.is_plot_unrolling  	= 0;
    
    % possible loa_feat_method:
    % loa_feat_method == 0: Akshara's Humanoids'14 features (_CONSTRAINED_)
    % loa_feat_method == 1: Potential Field 2nd Dynamic Obst Avoid features
    % loa_feat_method == 2: Giovanni's Kernelized General Features v01
    % loa_feat_method == 3: Franzi's Kernelized General Features v01
    % loa_feat_method == 4: Potential Field 3rd Dynamic Obst Avoid features (have some sense of KGF too...)
    % loa_feat_method == 5: Akshara's Humanoids'14 features (_UNCONSTRAINED_)
    % (enclose in a cell if you want to use multiple feature types, like in the following example below ...)
%     loa_feat_methods    = {2, 4};
    loa_feat_methods    = 2;
    
    % This feat_constraint_mode is for loa_feat_method 0, 1, 2, and 4.
    % For loa_feat_method 3 and 5, this value setting for
    % feat_constraint_mode is ignored,
    % and overridden with feat_constraint_mode = '_UNCONSTRAINED_'.
    % (in utilities/initializeAllInvolvedLOAparams.m
    %  or utilities/initializeFM_KGFv01LearnObsAvoidFeatParam.m)
    feat_constraint_mode                = '_CONSTRAINED_';
    
    learning_param.N_iter_ard        	= 200;
end

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

% loa_feat_methods_to_be_evaluated    = [0, 2, 3, 4];
loa_feat_methods_to_be_evaluated    = [4];

N_feat_methods  = size(loa_feat_methods_to_be_evaluated, 2);
eval_performance_metric_multi_settings  = cell(1, N_feat_methods);

data_annotation_obs_avoid_dominant_axis = cell2mat(data_global_coord.obs_avoid(:,3));
idx_no_dom_axis = find(data_annotation_obs_avoid_dominant_axis == 0);
idx_y_dom_axis  = find(data_annotation_obs_avoid_dominant_axis == 2);
idx_z_dom_axis  = find(data_annotation_obs_avoid_dominant_axis == 3);

idx_w_dom_axis  = union(idx_y_dom_axis, idx_z_dom_axis);
% num_no_dom_axis_demo    = 5;
% idx_randperm_no_dom_axis= randperm(length(idx_no_dom_axis));
% selected_idx_no_dom_axis= idx_no_dom_axis(idx_randperm_no_dom_axis(1,1:num_no_dom_axis_demo),:);
selected_idx_no_dom_axis = [];

% selected_obs_avoid_setting_numbers  = [19, 23, 26, 47, 49, 56, 63, ...
%                                        85, 93, 98, 119, 121, 137, 145, ...
%                                        150, 175, 190, 210, 211, 212, 217];
% selected_obs_avoid_setting_numbers  = [26, 47, 85, 98, 119, 190, 210, 217];
selected_obs_avoid_setting_numbers  = union(idx_w_dom_axis, selected_idx_no_dom_axis).';
max_num_trajs_per_setting           = 5;
figure_path                         = '/home/gsutanto/Desktop/CLMC/Publications/Humanoids16/data/test_plot_nmse_learn_vs_unroll/';

for j=1:N_feat_methods
    disp(['>>>Feature Method #', num2str(loa_feat_methods_to_be_evaluated(1, j)), ' (', num2str(j), '/', num2str(N_feat_methods), ') ...']);
    [ performance_metric_cell ] = learnAndUnrollObsAvoidViconDataSettingStepwisefitPrepruning( data_global_coord, ...
                                                                                               cart_coord_dmp_baseline_params, ...
                                                                                               loa_feat_methods_to_be_evaluated(1, j), ...
                                                                                               feat_constraint_mode, ...
                                                                                               learning_param, ...
                                                                                               unrolling_param, ...
                                                                                               selected_obs_avoid_setting_numbers, ...
                                                                                               max_num_trajs_per_setting, ...
                                                                                               figure_path, 1 );
    eval_performance_metric_multi_settings{1, j}   = performance_metric_cell;
end

save('eval_performance_metric_multi_settings.mat', 'eval_performance_metric_multi_settings');

copyfile('eval_performance_metric_multi_settings.mat', ...
         '/home/gsutanto/Desktop/CLMC/Publications/Humanoids16/data/');

% end of Learning and Unrolling (and Evaluating Performance Metric)

N_trial     = 5;
x_axis      = round(linspace(1, 76, N_trial));
nmse_learn  = zeros(1, N_trial);
nmse_unroll = zeros(1, N_trial);
for i=1:size(performance_metric_cell,1)
    nmse_learn(1,i)  = performance_metric_cell{i, 1}.nmse_learning;
    nmse_unroll(1,i) = performance_metric_cell{i, 1}.nmse_unroll;
end
plot(x_axis, nmse_learn, 'b', x_axis, nmse_unroll, 'r');
legend('NMSE Learning', 'NMSE Unrolling');