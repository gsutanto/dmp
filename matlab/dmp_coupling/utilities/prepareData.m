function [] = prepareData(task_type, dataset_Ct, out_data_dir, ...
                          is_evaluating_autoencoder, subset_settings_indices, ...
                          considered_subset_outlier_ranked_demo_indices, ...
                          generalization_subset_outlier_ranked_demo_indices, ...
                          post_filename_stacked_data, varargin)
    if (nargin > 8)
        is_evaluating_pca   = varargin{1};
    else
        is_evaluating_pca   = 1;
    end
    
    assert(exist(out_data_dir, 'dir')==7, ['out_data_dir=',out_data_dir,' does NOT exist!']);
    
    feature_type                = 'raw';
    mode_stack_dataset          = 1;

    N_primitive                 = size(dataset_Ct.sub_Ct_target, 1);

    training_subset_outlier_ranked_demo_indices         = setdiff(considered_subset_outlier_ranked_demo_indices, generalization_subset_outlier_ranked_demo_indices);
    if (isempty(generalization_subset_outlier_ranked_demo_indices))
        generalization_subset_outlier_ranked_demo_indices   = [1];  % CANNOT really be empty (for further Python processing)
    end
    subset_outlier_ranked_demo_indices_cell = {training_subset_outlier_ranked_demo_indices, generalization_subset_outlier_ranked_demo_indices};
    pre_filename_stacked_data_cell          = {'', 'test_unroll_'};

    for type=1:2
        for np=1:N_primitive
            if (is_evaluating_pca)
                if (is_evaluating_autoencoder)
                    [ X, Ct_target, ...
                      normalized_phase_PSI_mult_phase_V, phase_X, phase_V, ...
                      X_dim_reduced, data_point_priority, ...
                      X_dim_reduced_autoencoder ]   = stackDataset( dataset_Ct, subset_settings_indices, ...
                                                                    mode_stack_dataset, subset_outlier_ranked_demo_indices_cell{type}, feature_type, np );
                else
                    [ X, Ct_target, ...
                      normalized_phase_PSI_mult_phase_V, phase_X, phase_V, ...
                      X_dim_reduced, ...
                      data_point_priority ] = stackDataset( dataset_Ct, subset_settings_indices, ...
                                                            mode_stack_dataset, subset_outlier_ranked_demo_indices_cell{type}, feature_type, np );
                end
            else
                [ X, Ct_target, ...
                  normalized_phase_PSI_mult_phase_V, ~, ~, ~, ...
                  data_point_priority ] = stackDataset( dataset_Ct, subset_settings_indices, ...
                                                        mode_stack_dataset, subset_outlier_ranked_demo_indices_cell{type}, feature_type, np );
            end
            
            assert(size(X, 1) == size(Ct_target, 1), 'Ct_target size mis-match with X!');
            assert(size(X, 1) == size(normalized_phase_PSI_mult_phase_V, 1), 'normalized_phase_PSI_mult_phase_V size mis-match with X!');
            assert(size(X, 1) == size(phase_X, 1), 'phase_X size mis-match with X!');
            assert(size(X, 1) == size(phase_V, 1), 'phase_V size mis-match with X!');
            assert(size(X, 1) == size(data_point_priority, 1), 'data_point_priority size mis-match with X!');

            X_phase_X_phase_V = [X, phase_X, phase_V];
            
            save([out_data_dir, pre_filename_stacked_data_cell{type}, 'prim_',num2str(np),'_X_',feature_type,'_',task_type,post_filename_stacked_data,'.mat'],'X');
            save([out_data_dir, pre_filename_stacked_data_cell{type}, 'prim_',num2str(np),'_Ct_target_',task_type,post_filename_stacked_data,'.mat'],'Ct_target');
            save([out_data_dir, pre_filename_stacked_data_cell{type}, 'prim_',num2str(np),'_normalized_phase_PSI_mult_phase_V_',task_type,post_filename_stacked_data,'.mat'],'normalized_phase_PSI_mult_phase_V');
            save([out_data_dir, pre_filename_stacked_data_cell{type}, 'prim_',num2str(np),'_data_point_priority_',task_type,post_filename_stacked_data,'.mat'],'data_point_priority');
            if (is_evaluating_pca)
                assert(size(X, 1) == size(X_dim_reduced, 1), 'X_dim_reduced size mis-match with X!');
                
                save([out_data_dir, pre_filename_stacked_data_cell{type}, 'prim_',num2str(np),'_X_dim_reduced_',task_type,post_filename_stacked_data,'.mat'],'X_dim_reduced');  % for comparison with [Chebotar & Kroemer]'s method
            end
            if (is_evaluating_autoencoder)
                assert(size(X, 1) == size(X_dim_reduced_autoencoder, 1), 'X_dim_reduced_autoencoder size mis-match with X!');
                
                save([out_data_dir, pre_filename_stacked_data_cell{type}, 'prim_',num2str(np),'_X_dim_reduced_autoencoder_',task_type,post_filename_stacked_data,'.mat'],'X_dim_reduced_autoencoder');  % for comparison with [Chebotar & Kroemer]'s method
            end
            save([out_data_dir, pre_filename_stacked_data_cell{type}, 'prim_',num2str(np),'_X_',feature_type,'_phase_X_phase_V_',task_type,post_filename_stacked_data,'.mat'],'X_phase_X_phase_V');  % for comparison with naive structure involving phase_X and phase_V information but not via phase radial basis functions
            
            if (type == 1)
                fprintf(['Total # of Data Points for Training            Primitive ',num2str(np),': ',num2str(size(X, 1)),'\n'])
            elseif (type == 2)
                fprintf(['Total # of Data Points for Generalization Test Primitive ',num2str(np),': ',num2str(size(X, 1)),'\n'])
            end
        end
    end
end