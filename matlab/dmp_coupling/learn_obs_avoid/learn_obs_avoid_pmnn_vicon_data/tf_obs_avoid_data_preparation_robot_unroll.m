% Author        : Giovanni Sutanto
% Date          : October 2017
% Description   :
%   Prepare (stack) feature matrix X and 
%   target regression variable Ct_target, that will be used as input to
%   Google's TensorFlow regression engine, for unrolling on robot.

clear all;
close all;
clc;

addpath('../../../utilities/');

task_type                   = 'obs_avoid';
load(['dataset_Ct_',task_type,'.mat']);

out_data_dir                = ['../../../../python/dmp_coupling/learn_',task_type,'/tf/input_data/'];
createDirIfNotExist(out_data_dir);

is_evaluating_pca           = 0;
is_evaluating_autoencoder   = 0;

subset_settings_indices     = [1:222];

considered_subset_outlier_ranked_demo_indices       = [1:3];
generalization_subset_outlier_ranked_demo_indices   = [4];
post_filename_stacked_data                          = '';

prepareData(task_type, dataset_Ct_obs_avoid, out_data_dir, ...
            is_evaluating_autoencoder, subset_settings_indices, ...
            considered_subset_outlier_ranked_demo_indices, ...
            generalization_subset_outlier_ranked_demo_indices, ...
            post_filename_stacked_data, is_evaluating_pca);