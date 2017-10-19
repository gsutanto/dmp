% Author: Giovanni Sutanto
% Date  : July 2017
% Description   :
%   Visualize the predicted vs target coupling term (Ct),
%   especially on trials it has never seen before,
%   for evaluating the fitting/prediction quality,
%   as well as generalization to unseen data.

clear all;
close all;
clc;

rel_dir_path        = './';

addpath([rel_dir_path, '../../utilities/']);
addpath([rel_dir_path, '../../cart_dmp/cart_coord_dmp/']);
addpath([rel_dir_path, '../../cart_dmp/quat_dmp/']);
addpath([rel_dir_path, '../../dmp_multi_dim/']);
addpath([rel_dir_path, '../../neural_nets/feedforward/pmnn/']);

task_type           = 'scraping';
load(['dataset_Ct_tactile_asm_',task_type,'_augmented.mat']);

is_plotting_position_ct_target      = 0;
is_plotting_orientation_ct_target   = 1;

python_learn_tactile_fb_models_dir_path = [rel_dir_path, '../../../python/dmp_coupling/learn_tactile_feedback/models/'];

reinit_selection_idx= dlmread([python_learn_tactile_fb_models_dir_path, 'reinit_selection_idx.txt']);
TF_max_train_iters 	= dlmread([python_learn_tactile_fb_models_dir_path, 'TF_max_train_iters.txt']);

D                   = 3;
N_prims             = size(dataset_Ct_tactile_asm.sub_Ct_target, 1);

% subset_settings_indices = [1,2,3,4,6,7,8,9];   % new dataset (correctable baseline unrolling on robot): roll-variation-only of the tiltboard, with equal # of settings between positive-roll-angles and negative-roll-angles
subset_settings_indices = 1:size(dataset_Ct_tactile_asm.sub_Ct_target, 2);	% new dataset (correctable baseline unrolling on robot): roll-variation-only of the tiltboard, with equal # of settings between positive-roll-angles and negative-roll-angles

generalization_test_demo_trial_rank_no  = 3;

model_path  = [rel_dir_path, '../../../data/dmp_coupling/learn_tactile_feedback/scraping/neural_nets/pmnn/python_models/'];

% for np=1:N_prims
for np=2
    D_input             = size(dataset_Ct_tactile_asm.sub_X{np,1}{1,1}, 2);
    regular_NN_hidden_layer_topology = dlmread([python_learn_tactile_fb_models_dir_path, 'regular_NN_hidden_layer_topology.txt']);
    N_phaseLWR_kernels  = size(dataset_Ct_tactile_asm.sub_normalized_phase_PSI_mult_phase_V{np,1}{1,1}, 2);
    D_output            = 6;
    
    regular_NN_hidden_layer_activation_func_list = readStringsToCell([python_learn_tactile_fb_models_dir_path, 'regular_NN_hidden_layer_activation_func_list.txt']);
    
    NN_info.name        = 'my_ffNNphaseLWR';
    NN_info.topology    = [D_input, regular_NN_hidden_layer_topology, N_phaseLWR_kernels, D_output];
    NN_info.activation_func_list= {'identity', regular_NN_hidden_layer_activation_func_list{:}, 'identity', 'identity'};
    NN_info.filepath    = [model_path, 'prim_', num2str(np), '_params_reinit_', num2str(reinit_selection_idx(1, np)), '_step_',num2str(TF_max_train_iters,'%07d'),'.mat'];

    for ns_idx=1:size(subset_settings_indices, 2)
        ns              = subset_settings_indices(1, ns_idx);
        nd              = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,ns}(generalization_test_demo_trial_rank_no,1);
        
        Ct_target           = dataset_Ct_tactile_asm.sub_Ct_target{np,ns}{nd,1};
        
        % Perform PMNN Coupling Term Prediction
        [ Ct_prediction, ~ ]= performNeuralNetworkPrediction( NN_info, dataset_Ct_tactile_asm.sub_X{np,ns}{nd,1}, dataset_Ct_tactile_asm.sub_normalized_phase_PSI_mult_phase_V{np,ns}{nd,1} );
        
        %% Plotting
        
        if (is_plotting_position_ct_target)
            figure;
            act_type_string     = 'position';
            for d=1:D
                subplot(D,1,d);
                hold on;
                    if (d == 1)
                        dim_string  = 'x';
                    elseif (d == 2)
                        dim_string  = 'y';
                    elseif (d == 3)
                        dim_string  = 'z';
                    end
                    
                    plot(Ct_target(:,d), 'b');
                    plot(Ct_prediction(:,d), 'g');
                    
                    if (d == 1)
                        title([act_type_string, ' coupling term, prim #', num2str(np), ', setting #', num2str(ns), ', demo #', num2str(nd)]);
                        legend('target', 'prediction');
                    end
                hold off;
            end
        end

        if (is_plotting_orientation_ct_target)
            figure;
            act_type_string     = 'orientation';
            for d=1:D
                subplot(D,1,d);
                hold on;
                    if (d == 1)
                        dim_string  = 'alpha';
                    elseif (d == 2)
                        dim_string  = 'beta';
                    elseif (d == 3)
                        dim_string  = 'gamma';
                    end
                    
                    plot(Ct_target(:,D+d), 'b');
                    plot(Ct_prediction(:,D+d), 'g');
                    
                    if (d == 1)
                        title([act_type_string, ' coupling term, prim #', num2str(np), ', setting #', num2str(ns), ', demo #', num2str(nd)]);
                        legend('target', 'prediction');
                    end
                hold off;
            end
        end
    end
end