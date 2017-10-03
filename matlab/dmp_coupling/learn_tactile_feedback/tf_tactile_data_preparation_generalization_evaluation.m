% Author        : Zhe Su and Giovanni Sutanto
% Date          : September 2017
% Description   :
%   Prepare (stack) feature matrix X and 
%   target regression variable Ct_target, that will be used as input to
%   Google's TensorFlow regression engine, for generalization evaluation.

clear all;
close all;
clc;

addpath('../../utilities/');

task_type                   = 'scraping';
load(['dataset_Ct_tactile_asm_',task_type,'_augmented.mat']);

out_data_dir                = ['../../../python/dmp_coupling/learn_tactile_feedback/',task_type,'/generalization_test/'];
createDirIfNotExist(out_data_dir);
	
is_evaluating_autoencoder   = 0;

subset_settings_indices     = [1:4,9];

N_Trials_Per_Setting        = 15;

considered_subset_outlier_ranked_demo_indices       = [1:N_Trials_Per_Setting];

for n=1:N_Trials_Per_Setting
    fprintf(['Preparing generalization test on trial ',num2str(n),'.\n']);
    generalization_subset_outlier_ranked_demo_indices   = [n];
    post_filename_stacked_data                          = ['_', num2str(n)];

    prepareData(task_type, dataset_Ct_tactile_asm, out_data_dir, ...
                is_evaluating_autoencoder, subset_settings_indices, ...
                considered_subset_outlier_ranked_demo_indices, ...
                generalization_subset_outlier_ranked_demo_indices, ...
                post_filename_stacked_data);
end