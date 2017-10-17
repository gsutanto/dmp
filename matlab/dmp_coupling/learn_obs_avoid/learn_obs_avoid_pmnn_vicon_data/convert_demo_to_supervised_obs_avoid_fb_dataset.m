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

dmp_baseline_params.cart_coord{1,1} = cart_coord_dmp_baseline_params;
save(['dmp_baseline_params_obs_avoid.mat'],'dmp_baseline_params');

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

dataset_Ct_obs_avoid.sub_X              = cell(1,N_settings);
dataset_Ct_obs_avoid.sub_Ct_target      = cell(1,N_settings);
dataset_Ct_obs_avoid.sub_phase_PSI      = cell(1,N_settings);
dataset_Ct_obs_avoid.sub_phase_V        = cell(1,N_settings);
dataset_Ct_obs_avoid.sub_phase_X        = cell(1,N_settings);
dataset_Ct_obs_avoid.sub_normalized_phase_PSI_mult_phase_V  = cell(1,N_settings);
dataset_Ct_obs_avoid.sub_data_point_priority                = cell(1,N_settings);
dataset_Ct_obs_avoid.trial_idx_ranked_by_outlier_metric_w_exclusion  = cell(1,N_settings);

for ns=1:N_settings
    N_demos = size(data_global_coord.obs_avoid{ns,2}, 2);
    
    dataset_Ct_obs_avoid.sub_X{1,ns}            = cell(N_demos,1);
    dataset_Ct_obs_avoid.sub_Ct_target{1,ns}    = cell(N_demos,1);
    dataset_Ct_obs_avoid.sub_phase_PSI{1,ns}    = cell(N_demos,1);
    dataset_Ct_obs_avoid.sub_phase_V{1,ns}      = cell(N_demos,1);
    dataset_Ct_obs_avoid.sub_phase_X{1,ns}      = cell(N_demos,1);
    dataset_Ct_obs_avoid.sub_normalized_phase_PSI_mult_phase_V{1,ns}            = cell(N_demos,1);
    dataset_Ct_obs_avoid.sub_data_point_priority{1,ns}                          = cell(N_demos,1);
    dataset_Ct_obs_avoid.trial_idx_ranked_by_outlier_metric_w_exclusion{1,ns}   = [];
    for nd=1:N_demos
        disp(['Setting #', num2str(ns), '/', num2str(N_settings), ...
              ', Demo #',  num2str(nd), '/', num2str(N_demos)]);
        [ dataset_Ct_obs_avoid.sub_X{1,ns}{nd,1}, ...
          dataset_Ct_obs_avoid.sub_Ct_target{1,ns}{nd,1}, ...
          ~, ~, ...
          dataset_Ct_obs_avoid.sub_phase_PSI{1,ns}{nd,1}, ...
          dataset_Ct_obs_avoid.sub_phase_V{1,ns}{nd,1}, ...
          dataset_Ct_obs_avoid.sub_phase_X{1,ns}{nd,1}, ...
          is_good_demo ]    = computeSubFeatMatAndSubTargetCt( data_global_coord.obs_avoid{ns,2}(:,nd), ...
                                                               data_global_coord.obs_avoid{ns,1}, ...
                                                               data_global_coord.dt, ...
                                                               cart_coord_dmp_baseline_params, ...
                                                               loa_feat_methods, ...
                                                               loa_feat_param );
        
        phase_V             = dataset_Ct_obs_avoid.sub_phase_V{1,ns}{nd,1};
        phase_PSI           = dataset_Ct_obs_avoid.sub_phase_PSI{1,ns}{nd,1};
        normalized_phase_PSI_mult_phase_V   = phase_PSI .* repmat((phase_V ./ sum((phase_PSI+1.e-10),2)),1,n_rfs);
        dataset_Ct_obs_avoid.sub_normalized_phase_PSI_mult_phase_V{1,ns}{nd,1}  = normalized_phase_PSI_mult_phase_V;
        
        traj_length 	= size(dataset_Ct_obs_avoid.sub_X{1,ns}{nd,1},1);
        dataset_Ct_obs_avoid.sub_data_point_priority{1,ns}{nd,1}    = [traj_length:-1:1].';
        
      	if (is_good_demo == 1)
            dataset_Ct_obs_avoid.trial_idx_ranked_by_outlier_metric_w_exclusion{1,ns}	= [dataset_Ct_obs_avoid.trial_idx_ranked_by_outlier_metric_w_exclusion{1,ns}, nd];
        end
    end
    
    if (ns == 1)
        min_num_considered_demo     = length(dataset_Ct_obs_avoid.trial_idx_ranked_by_outlier_metric_w_exclusion{1,ns});
    else
        if (min_num_considered_demo > length(dataset_Ct_obs_avoid.trial_idx_ranked_by_outlier_metric_w_exclusion{1,ns}))
            min_num_considered_demo = length(dataset_Ct_obs_avoid.trial_idx_ranked_by_outlier_metric_w_exclusion{1,ns});
        end
    end
end

fprintf(['Minimum # of Considered Demonstrations = ',num2str(min_num_considered_demo),'\n']);

save(['dataset_Ct_obs_avoid.mat'],'dataset_Ct_obs_avoid');
                                                   
% end of Conversion of Demonstration Dataset into Supervised Obstacle Avoidance Feedback Model Dataset