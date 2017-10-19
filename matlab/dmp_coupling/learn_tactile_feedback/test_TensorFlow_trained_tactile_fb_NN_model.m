clear all;
close all;
clc;

task_type                       = 'scraping';
learn_fb_task                   = 'learn_tactile_feedback';
amd_clmc_dmp_root_dir_path      = '../../../';
data_learn_fb_task_subdir_path  = [learn_fb_task, '/scraping/'];
python_learn_fb_task_TF_models_prefix_subdir_path   = '';
start_prim_num                  = 1;
end_prim_num                    = 1;
PMNN_name                       = 'my_ffNNphaseLWR';

testTensorFlowTrainedPMNNFeedbackModel( task_type, learn_fb_task, ...
                                        amd_clmc_dmp_root_dir_path, ...
                                        data_learn_fb_task_subdir_path, ...
                                        python_learn_fb_task_TF_models_prefix_subdir_path, ...
                                        start_prim_num, end_prim_num, PMNN_name );