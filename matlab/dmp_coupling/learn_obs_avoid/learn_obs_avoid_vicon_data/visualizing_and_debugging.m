% Author: Giovanni Sutanto
% Date  : August 01, 2016

clear  	all;
close   all;
clc;

addpath('../utilities/');

% possible performance_metric_evaluation_mode:
% performance_metric_evaluation_mode == 0: no evaluation; run just as
%                                          regular execution with selected parameters
% performance_metric_evaluation_mode == 1: evaluation on each single
%                                          obstacle avoidance demonstration
% performance_metric_evaluation_mode == 2: evaluation on multiple (selected) 
%                                          obstacle avoidance demonstrations
performance_metric_evaluation_mode                      = 0;

unrolling_param.is_comparing_with_cpp_implementation    = 0;

is_debugging_X_and_Ct                       = 1;

% learning_param.max_cond_number = 1000.0;   % Recommendation from Stefan (for Robotic applications)
learning_param.max_cond_number              = 5e3;

% possible learning_method:
% learning_param.learning_method == 1: ARD
learning_param.learning_method              = 1;

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
    % loa_feat_method == 6: Potential Field 4th Dynamic Obst Avoid features (have some sense of KGF too...)
    % (enclose in a cell if you want to use multiple feature types, like in the following example below ...)
%     loa_feat_methods    = {2, 4};
loa_feat_methods    = 4;
feat_constraint_mode                = '_CONSTRAINED_';

learning_param.N_iter_ard        	= 200;
selected_obs_avoid_setting_numbers  = [27];
max_num_trajs_per_setting           = 500;

learning_param.learning_constraint_mode     = '_NONE_';

D               = 3;

n_rfs           = 25;   % Number of basis functions used to represent the forcing term of DMP
c_order         = 1;    % DMP is using 2nd order canonical system

%% Data Creation/Loading

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

%% visualize data of this setting
vicon_marker_radius             = 15.0/2000.0;  % in meter
critical_position_marker_radius = 30/1000.0;    % in meter

setting_no = selected_obs_avoid_setting_numbers(1);
obs_avoid_demos_global = data_global_coord.obs_avoid{setting_no, 2}(1,:);
figure, hold on
n_demos = length(obs_avoid_demos_global);
for i=1:size(data_global_coord.baseline,2)
    plot3(data_global_coord.baseline{1,i}(:,1),...
        data_global_coord.baseline{1,i}(:,2),...
        data_global_coord.baseline{1,i}(:,3),...
        'b');
end

for i = 1:n_demos
    plot3(obs_avoid_demos_global{i}(:,1),...
        obs_avoid_demos_global{i}(:,2), ...
        obs_avoid_demos_global{i}(:,3), 'r')
end

for op=1:size(data_global_coord.obs_avoid{setting_no,1},1)
    plot_sphere(vicon_marker_radius,...
        data_global_coord.obs_avoid{setting_no,1}(op,1),...
        data_global_coord.obs_avoid{setting_no,1}(op,2),...
        data_global_coord.obs_avoid{setting_no,1}(op,3));
end
axis equal
hold off

figure, hold on
for i=1:size(data_global_coord.baseline,2)
    [demo_local] = convertCTrajAtOldToNewCoordSys( data_global_coord.baseline{1,i}, ...
        cart_coord_dmp_baseline_params.T_global_to_local_H );
    
    plot3(demo_local(:,1),demo_local(:,2),demo_local(:,3),'b');
end
for i = 1:n_demos
    [demo_local] = convertCTrajAtOldToNewCoordSys( obs_avoid_demos_global{i}, ...
        cart_coord_dmp_baseline_params.T_global_to_local_H );
    
    plot3(demo_local(:,1),demo_local(:,2),demo_local(:,3),'r');
end
for op=1:size(data_global_coord.obs_avoid{setting_no,1},1)
    [obs_point_local] = convertCTrajAtOldToNewCoordSys( data_global_coord.obs_avoid{setting_no,1}(op,:), ...
                                                        cart_coord_dmp_baseline_params.T_global_to_local_H );
    plot_sphere(vicon_marker_radius, obs_point_local(1), obs_point_local(2),obs_point_local(3))
end
axis equal
hold off

