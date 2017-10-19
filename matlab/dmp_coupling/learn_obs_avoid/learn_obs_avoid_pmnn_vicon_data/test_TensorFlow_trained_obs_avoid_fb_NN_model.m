clear all;
close all;
clc;

task_type                       = 'obs_avoid';
learn_fb_task                   = 'learn_obs_avoid';
amd_clmc_dmp_root_dir_path      = '../../../../';
data_learn_fb_task_subdir_path  = [learn_fb_task, '/static_obs/'];
python_learn_fb_task_TF_models_prefix_subdir_path   = 'tf/';
start_prim_num                  = 1;
end_prim_num                    = 1;
PMNN_name                       = 'my_PMNN_obs_avoid_fb';

testTensorFlowTrainedPMNNFeedbackModel( task_type, learn_fb_task, ...
                                        amd_clmc_dmp_root_dir_path, ...
                                        data_learn_fb_task_subdir_path, ...
                                        python_learn_fb_task_TF_models_prefix_subdir_path, ...
                                        start_prim_num, end_prim_num, PMNN_name );