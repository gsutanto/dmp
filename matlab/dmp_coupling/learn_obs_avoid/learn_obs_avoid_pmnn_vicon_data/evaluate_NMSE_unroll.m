clear all;
close all;
clc;

addpath('../utilities/');

load('dataset_Ct_obs_avoid.mat');
load('unroll_dataset_learned_Ct_obs_avoid.mat');
load('no_dynamics_unroll_dataset_learned_Ct_obs_avoid.mat');

subset_settings_indices     = [1:222];
subset_demos_indices        = [1:3];
mode_stack_dataset          = 2;
feature_type                = 'raw';
N_primitive                 = size(dataset_Ct_obs_avoid.sub_Ct_target, 1);

for np=1:N_primitive
    [ ~, Ct_target ]	= stackDataset( dataset_Ct_obs_avoid, subset_settings_indices, ...
                                        mode_stack_dataset, subset_demos_indices, feature_type, np );
    [ ~, Ct_unroll ]	= stackDataset( unroll_dataset_learned_Ct_obs_avoid, subset_settings_indices, ...
                                        mode_stack_dataset, subset_demos_indices, feature_type, np );
    [ ~, Ct_unroll_no_dyn ] = stackDataset( no_dynamics_unroll_dataset_learned_Ct_obs_avoid, subset_settings_indices, ...
                                            mode_stack_dataset, subset_demos_indices, feature_type, np );
    [ mse_unroll, nmse_unroll ]                 = computeNMSE( Ct_unroll, Ct_target );
    [ mse_unroll_no_dyn, nmse_unroll_no_dyn ]   = computeNMSE( Ct_unroll_no_dyn, Ct_target );
    disp(['nmse_unroll        = ', num2str(nmse_unroll)]);
    disp(['nmse_unroll_no_dyn = ', num2str(nmse_unroll_no_dyn)]);
end