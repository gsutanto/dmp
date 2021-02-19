clear all;
close all;
clc;

generic_task_type           = 'scraping';
specific_task_type          = 'scraping_w_tool';

dmp_root_dir_path  = '../../../';

python_root_dir_path        = [dmp_root_dir_path, 'python/'];

python_learn_tactile_fb_dir_path            = [python_root_dir_path, 'dmp_coupling/learn_tactile_feedback/'];
python_learn_tactile_fb_models_dir_path     = [python_learn_tactile_fb_dir_path, 'models/'];
python_learn_tactile_fb_models_generalization_test_dir_path     = [python_learn_tactile_fb_models_dir_path, 'generalization_test/'];

% date                    = '20170908_1';
date                    = '20170908_z';
additional_description 	= '_after_pruning_inconsistent_demos_positive_side';
result_log_path         = ['~/Desktop/archives_learn_tactile_fb/',date,'_',specific_task_type,'_correctable',additional_description,'/generalization_test/'];

reinit_selection_idx    = dlmread([python_learn_tactile_fb_models_dir_path, 'reinit_selection_idx.txt']);
TF_max_train_iters      = dlmread([python_learn_tactile_fb_models_dir_path, 'TF_max_train_iters.txt']);
N_prims                 = size(reinit_selection_idx, 2);

precision_string     	= '%.20f';

generalization_test_comparison_dimension    = 5;
    
% input_selector = 1; % X_raw input, Phase-Modulated Neural Network                      (PMNN) with  1 regular hidden layer only of 100 nodes and 25 nodes in the phase-modulated final hidden layer (regular execution)
% input_selector = 2; % X_dim_reduced_pca input,                                          PMNN  with NO regular hidden layer                   and 25 nodes in the phase-modulated final hidden layer (comparison: with [Chebotar & Kroemer]'s model)
% input_selector = 3; % X_raw input, Phase-Modulated Neural Network                      (PMNN) with  1 regular hidden layer only of   6 nodes and 25 nodes in the phase-modulated final hidden layer (comparison: between different number of nodes in the regular hidden layer)
% input_selector = 4; % X_dim_reduced_autoencoder input,                                  PMNN  with NO regular hidden layer                   and 25 nodes in the phase-modulated final hidden layer (comparison: with [Chebotar & Kroemer]'s model)
% input_selector = 5; % X_raw input, regular Feed-Forward Neural Network                 (FFNN) with 100 and 25 nodes in the (regular) hidden layers,           NO phase modulation                   (comparison between different neural network structures)
% input_selector = 6; % X_raw input,                                                      PMNN  with NO regular hidden layer                   and 25 nodes in the phase-modulated final hidden layer (comparison between different neural network structures)
% input_selector = 7; % X_raw_phase_X_phase_V input, regular Feed-Forward Neural Network (FFNN) with 100 and 25 nodes in the (regular) hidden layers,           NO phase modulation                   (comparison between different neural network structures)

for input_selector = 1:7
    if (input_selector == 1)
        model_descriptor_sub_path   = '';
    elseif (input_selector == 2)
        model_descriptor_sub_path   = 'comparison_vs_separated_feature_learning/input_X_dim_reduced_pca_no_reg_hidden_layer/';
    elseif (input_selector == 3)
        model_descriptor_sub_path   = 'comparison_vs_separated_feature_learning/input_X_raw_reg_hidden_layer_6/';
    elseif (input_selector == 4)
        model_descriptor_sub_path   = 'comparison_vs_separated_feature_learning/input_X_dim_reduced_autoencoder_no_reg_hidden_layer/';
    elseif (input_selector == 5)
        model_descriptor_sub_path   = 'comparison_vs_different_neural_net_structure/input_X_raw_ffnn_hidden_layer_100_25/';
    elseif (input_selector == 6)
        model_descriptor_sub_path   = 'comparison_vs_different_neural_net_structure/input_X_raw_no_reg_hidden_layer/';
    elseif (input_selector == 7)
        model_descriptor_sub_path   = 'comparison_vs_different_neural_net_structure/input_X_raw_phase_X_phase_V_ffnn_hidden_layer_100_25/';
    end
    
    python_learn_tactile_fb_models_gen_test_specific_model_dir_path = [python_learn_tactile_fb_models_generalization_test_dir_path, model_descriptor_sub_path];

    nmse_log    = cell(N_prims, 1);
    mean_nmse   = cell(N_prims, 1);
    std_nmse    = cell(N_prims, 1);

    for np=2:3
        generalization_test_id                	= 1;
        matfilename     = [python_learn_tactile_fb_models_gen_test_specific_model_dir_path, 'prim_',num2str(np),'_best_nmse_trial_',num2str(generalization_test_id),'.mat'];
        while (exist(matfilename, 'file') == 2)
            load(matfilename);
            nmse_log{np}.train(generalization_test_id)  = best_nmse_train(generalization_test_comparison_dimension);
            nmse_log{np}.valid(generalization_test_id)  = best_nmse_valid(generalization_test_comparison_dimension);
            nmse_log{np}.test(generalization_test_id)   = best_nmse_test(generalization_test_comparison_dimension);
            nmse_log{np}.generalization_test(generalization_test_id)    = best_nmse_generalization_test(generalization_test_comparison_dimension);
            generalization_test_id 	= generalization_test_id + 1;
            matfilename             = [python_learn_tactile_fb_models_gen_test_specific_model_dir_path, 'prim_',num2str(np),'_best_nmse_trial_',num2str(generalization_test_id),'.mat'];
        end
        mean_nmse{np}.train                  = mean(nmse_log{np}.train);
        mean_nmse{np}.valid                  = mean(nmse_log{np}.valid);
        mean_nmse{np}.test                   = mean(nmse_log{np}.test);
        mean_nmse{np}.generalization_test    = mean(nmse_log{np}.generalization_test);
        
        std_nmse{np}.train                   = std(nmse_log{np}.train);
        std_nmse{np}.valid                   = std(nmse_log{np}.valid);
        std_nmse{np}.test                    = std(nmse_log{np}.test);
        std_nmse{np}.generalization_test     = std(nmse_log{np}.generalization_test);
    end

    save([python_learn_tactile_fb_models_gen_test_specific_model_dir_path,'ave_best_generalization_nmse.mat'], 'mean_nmse', 'std_nmse', 'nmse_log');
end

copyfile(python_learn_tactile_fb_models_generalization_test_dir_path, result_log_path);