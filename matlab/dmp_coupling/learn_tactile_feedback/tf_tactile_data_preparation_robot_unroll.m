% Author        : Zhe Su and Giovanni Sutanto
% Date          : February 2017
% Description   :
%   Prepare (stack) feature matrix X and 
%   target regression variable Ct_target, that will be used as input to
%   Google's TensorFlow regression engine, for unrolling on robot.

clear all;
close all;
clc;

addpath('../../utilities/');

task_type                   = 'scraping';
load(['dataset_Ct_tactile_asm_',task_type,'_augmented.mat']);

out_data_dir                = ['../../../python/dmp_coupling/learn_tactile_feedback/',task_type,'/'];
createDirIfNotExist(out_data_dir);
	
is_evaluating_autoencoder   = 0;

% subset_settings_indices     = [1,2,3,4,5,6,7,8,10,14,16,17,18,19,20,21,22,23,24,25,26,28,29,30];
% subset_settings_indices     = [2,3,4,6,7,8,10,14,16,17,18,19,20,21,22,23,24,25,26,28,29,30];
% subset_settings_indices     = [22];
% subset_settings_indices     = [1,2,3,4,5,6,7,8,9,10,11,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31];
% subset_settings_indices     = [1,2,3,4,5,6,7,8]; % roll-variation-only of the tiltboard
% subset_settings_indices     = [1,2,3,4,5,6,7,8,9]; % roll-variation-only of the tiltboard plus baseline (9) setting
% subset_settings_indices     = [5];    % setting 5 for "setting generalization" test
% subset_settings_indices     = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]; % roll-variation-only of the tiltboard plus baseline (17) setting
% subset_settings_indices     = [1,2,3,4,5,8,9,10,11,12,17];  % roll-variation-only of the tiltboard, with equal # of settings between positive-roll-angles and negative-roll-angles, plus baseline (17) setting
% subset_settings_indices     = [1,2,3,4,5,6,7,17];   % positive-roll-angle-variation-only of the tiltboard, plus baseline (17) setting
% subset_settings_indices     = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];   % roll-variation-only of the tiltboard, with equal # of settings between positive-roll-angles and negative-roll-angles
% subset_settings_indices     = [1:9];   % new dataset (correctable baseline unrolling on robot): roll-variation-only of the tiltboard, with equal # of settings between positive-roll-angles and negative-roll-angles
% subset_settings_indices     = [1:4,9];
% subset_settings_indices     = [1:4];
subset_settings_indices     = [4:8]; % for IJRR'19 Paper

% considered_subset_outlier_ranked_demo_indices       = [1:15]; % for ICRA'18 Paper
considered_subset_outlier_ranked_demo_indices       = [3:10]; % for IJRR'19 Paper
generalization_subset_outlier_ranked_demo_indices   = [1:2];
post_filename_stacked_data                          = '';

prepareData(task_type, dataset_Ct_tactile_asm, out_data_dir, ...
            is_evaluating_autoencoder, subset_settings_indices, ...
            considered_subset_outlier_ranked_demo_indices, ...
            generalization_subset_outlier_ranked_demo_indices, ...
            post_filename_stacked_data);