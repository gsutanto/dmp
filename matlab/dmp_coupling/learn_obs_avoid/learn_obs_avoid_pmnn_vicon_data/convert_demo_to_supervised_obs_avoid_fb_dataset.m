% Author: Giovanni Sutanto
% Date  : October 2017
% Description:
%   Convert (segmented) demonstrations into 
%   supervised obstacle avoidance feedback dataset.

clear  	all;
close   all;
clc;

addpath('../utilities/');
addpath('../vicon/');
addpath('../../../cart_dmp/cart_coord_dmp/');

unrolling_param.is_comparing_with_cpp_implementation                = 0;
unrolling_param.is_unrolling_only_1st_demo_each_trained_settings    = 1;
unrolling_param.is_plot_unrolling                                   = 1;
unrolling_param.verify_NN_inference                                 = 0;

learning_param.max_cond_number              = 5e3;
learning_param.feature_variance_threshold   = 1e-4;
learning_param.max_abs_ard_weight_threshold = 7.5e3;
learning_param.N_iter_ard                   = 200;
learning_param.learning_constraint_mode     = '_NONE_';

loa_feat_methods_to_be_evaluated= [7];

feat_constraint_mode            = '_CONSTRAINED_';

loa_feat_methods                = 7;    % Neural Network

max_num_trajs_per_setting       = 500;

D     	= 3;

n_rfs  	= 25;   % Number of basis functions used to represent the forcing term of DMP
c_order = 1;    % DMP is using 2nd order canonical system

%% Demo Dataset Preparation

data_global_coord   = prepareDemoDatasetLOAVicon;

save('data_multi_demo_vicon_static_global_coord.mat', 'data_global_coord');

% end of Demo Dataset Preparation

%% Baseline Primitive Learning

disp('Processing Local Coordinate System for Demonstrated Baseline Trajectories ...');
[ cart_coord_dmp_baseline_params, ...
  unrolling_param.cart_coord_dmp_baseline_unroll_global_traj ] = learnCartPrimitiveMultiOnLocalCoord(data_global_coord.baseline, data_global_coord.dt, n_rfs, c_order);

% end of Baseline Primitive Learning

%% Obstacle Avoidance Features Grid Setting

loa_feat_param = initializeAllInvolvedLOAparams( loa_feat_methods, ...
                                                 cart_coord_dmp_baseline_params.c_order, ...
                                                 learning_param, ...
                                                 unrolling_param, ...
                                                 feat_constraint_mode );

% end of Obstacle Avoidance Features Grid Setting

%% Conversion of Demonstration Dataset into Supervised Obstacle Avoidance Feedback Model Dataset

N_settings  = size(data_global_coord.obs_avoid, 1);

dataset_Ct_obs_avoid.sub_X          = cell(N_settings,1);
dataset_Ct_obs_avoid.sub_Ct_target  = cell(N_settings,1);
dataset_Ct_obs_avoid.sub_phase_PSI  = cell(N_settings,1);
dataset_Ct_obs_avoid.sub_phase_V    = cell(N_settings,1);
dataset_Ct_obs_avoid.sub_phase_X    = cell(N_settings,1);

for ns=1:N_settings
    N_demos = size(data_global_coord.obs_avoid{ns,2}, 2);
    
    dataset_Ct_obs_avoid.sub_X{ns,1}        = cell(N_demos,1);
    dataset_Ct_obs_avoid.sub_Ct_target{ns,1}= cell(N_demos,1);
    dataset_Ct_obs_avoid.sub_phase_PSI{ns,1}= cell(N_demos,1);
    dataset_Ct_obs_avoid.sub_phase_V{ns,1}  = cell(N_demos,1);
    dataset_Ct_obs_avoid.sub_phase_X{ns,1}  = cell(N_demos,1);
    for nd=1:N_demos
        disp(['Setting #', num2str(ns), '/', num2str(N_settings), ...
              ', Demo #',  num2str(nd), '/', num2str(N_demos)]);
        [ dataset_Ct_obs_avoid.sub_X{ns,1}{nd,1}, ...
          dataset_Ct_obs_avoid.sub_Ct_target{ns,1}{nd,1}, ...
          ~, ~, ...
          dataset_Ct_obs_avoid.sub_phase_PSI{ns,1}{nd,1}, ...
          dataset_Ct_obs_avoid.sub_phase_V{ns,1}{nd,1}, ...
          dataset_Ct_obs_avoid.sub_phase_X{ns,1}{nd,1} ...
          ]	= computeSubFeatMatAndSubTargetCt( data_global_coord.obs_avoid{ns,2}(:,nd), ...
                                               data_global_coord.obs_avoid{ns,1}, ...
                                               data_global_coord.dt, ...
                                               cart_coord_dmp_baseline_params, ...
                                               loa_feat_methods, ...
                                               loa_feat_param );
    end
end

save(['dataset_Ct_obs_avoid.mat'],'dataset_Ct_obs_avoid');
                                                   
% end of Conversion of Demonstration Dataset into Supervised Obstacle Avoidance Feedback Model Dataset