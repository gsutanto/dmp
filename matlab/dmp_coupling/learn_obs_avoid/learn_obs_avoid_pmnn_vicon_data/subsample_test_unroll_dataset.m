clear all;
close all;
clc;

rng(1234);

subsample_ratio     = 150;

task_type           = 'obs_avoid';

data_dir            = ['../../../../python/dmp_coupling/learn_',task_type,'/tf/input_data/'];

load([data_dir, 'test_unroll_prim_1_X_raw_',task_type,'.mat']);
load([data_dir, 'test_unroll_prim_1_normalized_phase_PSI_mult_phase_V_',task_type,'.mat']);
load([data_dir, 'test_unroll_prim_1_Ct_target_',task_type,'.mat']);

N_data_points    	= size(X, 1);

randpermed_data_idx = randperm(N_data_points);

subsampled_data_idx = randpermed_data_idx(1:round(N_data_points/subsample_ratio));

% subsampled data:
X                                   = X(subsampled_data_idx, :);
normalized_phase_PSI_mult_phase_V  	= normalized_phase_PSI_mult_phase_V(subsampled_data_idx, :);
Ct_target                         	= Ct_target(subsampled_data_idx, :);

save([data_dir,'subsampled_test_unroll_prim_1_X_raw_',task_type,'.mat'],'X');
save([data_dir,'subsampled_test_unroll_prim_1_normalized_phase_PSI_mult_phase_V_',task_type,'.mat'],'normalized_phase_PSI_mult_phase_V');
save([data_dir,'subsampled_test_unroll_prim_1_Ct_target_',task_type,'.mat'],'Ct_target');