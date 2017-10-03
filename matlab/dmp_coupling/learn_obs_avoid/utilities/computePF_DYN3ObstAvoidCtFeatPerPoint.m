function [ varargout ] = computePF_DYN3ObstAvoidCtFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau, px, pv )
    % Potential Field 3rd Dynamic Obst Avoid features (have some sense of KGF too...)
    
    x3          = endeff_state{1,1};
    v3          = endeff_state{2,1};
    
    o3          = obs_state{1,1};
    od3         = obs_state{2,1};
    
    ox3         = o3 - x3;
    rel_v3      = v3 - od3;
    
    num_thresh  = 1e-4;   % numerical threshold
    
    if ((norm(rel_v3) >= num_thresh) && (norm(ox3) >= num_thresh))
        cos_theta 	= (ox3.'*rel_v3)/(norm(ox3)*norm(rel_v3));
    
        loa_feat_matrix_per_point = -tau * norm(rel_v3) * ...
                                    (((ox3 * loa_feat_param.PF_DYN3_k_rowcoldepth_vector.') - ...
                                      ((((cos_theta/((norm(ox3))^2))*ox3)-((1.0/(norm(ox3)*norm(rel_v3)))*rel_v3)) * (((cos_theta*ones(size(loa_feat_param.PF_DYN3_beta_rowcoldepth_vector.'))) - loa_feat_param.PF_DYN3_beta_rowcoldepth_vector.') .* loa_feat_param.PF_DYN3_beta_rowcoldepth_D_vector.'))) .* ...
                                     repmat((exp(-0.5 * (((cos_theta*ones(size(loa_feat_param.PF_DYN3_beta_rowcoldepth_vector.'))) - loa_feat_param.PF_DYN3_beta_rowcoldepth_vector.').^2) .* loa_feat_param.PF_DYN3_beta_rowcoldepth_D_vector.') .* ...
                                             exp(-0.5 * loa_feat_param.PF_DYN3_k_rowcoldepth_vector.' * (ox3.'*ox3)) .* ...
                                             exp(-0.5 * (((px*ones(size(loa_feat_param.PF_DYN3_s_rowcoldepth_vector.'))) - loa_feat_param.PF_DYN3_s_rowcoldepth_vector.').^2) .* loa_feat_param.PF_DYN3_s_rowcoldepth_D_vector.')), 3, 1));

        s_kernel_normalizer         = sum(exp(-0.5 * (((px*ones(size(loa_feat_param.PF_DYN3_s_depth_grid.'))) - loa_feat_param.PF_DYN3_s_depth_grid.').^2) .* loa_feat_param.PF_DYN3_s_depth_D_grid.'));

        if (loa_feat_param.c_order == 1)
            s_multiplier            = pv;
        else
            s_multiplier            = px;
        end

        loa_feat_matrix_per_point   = loa_feat_matrix_per_point * s_multiplier/s_kernel_normalizer;
    else
        loa_feat_matrix_per_point   = zeros(3, size(loa_feat_param.PF_DYN3_beta_rowcoldepth_vector, 1));
    end
    
    varargout(1)                    = {loa_feat_matrix_per_point};
    
end