function [ X, Ct_target, varargout ] = stackDataset( dataset, subset_settings_indices, mode, mode_arg, feature_type, primitive_no )
    % Author        : Zhe Su and Giovanni Sutanto
    % Date          : February 2017
    % Description   :
    %   Given X and Ct_target pairs per trajectory, 
    %   given in MATLAB's cell format,
    %   stack the overall X and Ct_target, that will be used as input to
    %   Google's TensorFlow regression engine.
    
    assert(length(mode) == 1, 'mode specification must be scalar, not vector/matrix');
    assert(((mode > 0) && (mode < 3)), 'supported mode is 1 or 2');
    
    if (mode == 1)
        % after ranking the trials based on the outlier metric
        % (ranked by dataset.outlier_metric{primitive_no,setting_no} field,
        % i.e. rank 1==most likely is NOT an outlier; 
        % rank <end>==most likely is an outlier),
        % pick a subset of it, specified in subset_outlier_ranked_demo_indices, 
        % e.g. if subset_outlier_ranked_demo_indices=[1,3,4,5],
        % then this function will stack dataset of 
        % trials rank 1, 3, 4, and 5 (RECOMMENDED).
        
        subset_outlier_ranked_demo_indices  = mode_arg;
    elseif (mode == 2)
        % pick trials with indices specified in subset_demos_index.
        
        subset_demos_indices= mode_arg;
    end
    
    N_settings_to_extract   = length(subset_settings_indices);
    X_setting_cell          = cell(N_settings_to_extract, 1);
    Ct_target_setting_cell  = cell(N_settings_to_extract, 1);
    if (nargout > 2)
        normalized_phase_PSI_mult_phase_V_setting_cell   = cell(N_settings_to_extract, 1);
    end
    if (nargout > 3)
        phase_X_setting_cell        = cell(N_settings_to_extract, 1);
    end
    if (nargout > 4)
        phase_V_setting_cell        = cell(N_settings_to_extract, 1);
    end
    if (nargout > 5)
        X_dim_reduced_setting_cell  = cell(N_settings_to_extract, 1);
    end
    if (nargout > 6)
        data_point_priority_setting_cell    = cell(N_settings_to_extract, 1);
    end
    if (nargout > 7)
        X_dim_reduced_autoencoder_setting_cell  = cell(N_settings_to_extract, 1);
    end
    for ns_idx = 1:N_settings_to_extract
        setting_no              = subset_settings_indices(ns_idx);
        if (mode == 1)
            existed_subset_outlier_ranked_demo_indices  = intersect([1:length(dataset.trial_idx_ranked_by_outlier_metric_w_exclusion{primitive_no,setting_no})], subset_outlier_ranked_demo_indices);
            subset_demos_indices= dataset.trial_idx_ranked_by_outlier_metric_w_exclusion{primitive_no,setting_no}(existed_subset_outlier_ranked_demo_indices);
        end
        
        if (strcmp(feature_type, 'raw'))
            X_setting_cell{ns_idx,1}= cell2mat(dataset.sub_X{primitive_no,setting_no}(subset_demos_indices,1));
        elseif (strcmp(feature_type, 'gauss_basis_func'))
            X_setting_cell{ns_idx,1}= cell2mat(dataset.sub_X_gauss_basis_func{primitive_no,setting_no}(subset_demos_indices,1));
        elseif (strcmp(feature_type, 'phase_var_mult_gauss_basis_func'))
            X_setting_cell{ns_idx,1}= cell2mat(dataset.sub_X_phase_var_mult_gauss_basis_func{primitive_no,setting_no}(subset_demos_indices,1));
        end
        
        Ct_target_setting_cell{ns_idx,1}= cell2mat(dataset.sub_Ct_target{primitive_no,setting_no}(subset_demos_indices,1));
        
        if (nargout > 2)
            normalized_phase_PSI_mult_phase_V_setting_cell{ns_idx,1} = cell2mat(dataset.sub_normalized_phase_PSI_mult_phase_V{primitive_no,setting_no}(subset_demos_indices,1));
        end
        if (nargout > 3)
            phase_X_setting_cell{ns_idx,1}  = cell2mat(dataset.sub_phase_X{primitive_no,setting_no}(subset_demos_indices,1));
        end
        if (nargout > 4)
            phase_V_setting_cell{ns_idx,1}  = cell2mat(dataset.sub_phase_V{primitive_no,setting_no}(subset_demos_indices,1));
        end
        if (nargout > 5)
            X_dim_reduced_setting_cell{ns_idx,1}    = cell2mat(dataset.sub_X_dim_reduced{primitive_no,setting_no}(subset_demos_indices,1));
        end
        if (nargout > 6)
            data_point_priority_setting_cell{ns_idx,1}	= cell2mat(dataset.sub_data_point_priority{primitive_no,setting_no}(subset_demos_indices,1));
        end
        if (nargout > 7)
            X_dim_reduced_autoencoder_setting_cell{ns_idx,1}	= cell2mat(dataset.sub_X_dim_reduced_autoencoder{primitive_no,setting_no}(subset_demos_indices,1));
        end
    end
    
    X           = cell2mat(X_setting_cell);
    Ct_target   = cell2mat(Ct_target_setting_cell);
    
    if (nargout > 2)
        normalized_phase_PSI_mult_phase_V   = cell2mat(normalized_phase_PSI_mult_phase_V_setting_cell);
        varargout(1)        = {normalized_phase_PSI_mult_phase_V};
    end
    if (nargout > 3)
        phase_X             = cell2mat(phase_X_setting_cell);
        varargout(2)        = {phase_X};
    end
    if (nargout > 4)
        phase_V             = cell2mat(phase_V_setting_cell);
        varargout(3)        = {phase_V};
    end
    if (nargout > 5)
        X_dim_reduced       = cell2mat(X_dim_reduced_setting_cell);
        varargout(4)        = {X_dim_reduced};
    end
    if (nargout > 6)
        data_point_priority = cell2mat(data_point_priority_setting_cell);
        varargout(5)        = {data_point_priority};
    end
    if (nargout > 7)
        X_dim_reduced_autoencoder 	= cell2mat(X_dim_reduced_autoencoder_setting_cell);
        varargout(6)                = {X_dim_reduced_autoencoder};
    end
end