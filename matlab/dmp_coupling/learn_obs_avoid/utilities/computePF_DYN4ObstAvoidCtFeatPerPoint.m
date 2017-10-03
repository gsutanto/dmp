function [ varargout ] = computePF_DYN4ObstAvoidCtFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau, px, pv )
    % Potential Field 4th Dynamic Obst Avoid features (have some sense of KGF too...)
    
    x3          = endeff_state{1,1};
    v3          = endeff_state{2,1};
    
    o3          = obs_state{1,1};
    od3         = obs_state{2,1};
    
    ox3         = o3 - x3;
    rel_v3      = v3 - od3;
    
    loa_feat_matrix_per_point = -tau * norm(rel_v3) * ...
                                ((ox3 * loa_feat_param.PF_DYN4_k_rowcol_vector.') .* ...
                                 repmat((exp(-0.5 * loa_feat_param.PF_DYN4_k_rowcol_vector.' * (ox3.'*ox3)) .* ...
                                         exp(-0.5 * (((px*ones(size(loa_feat_param.PF_DYN4_s_rowcol_vector.'))) - loa_feat_param.PF_DYN4_s_rowcol_vector.').^2) .* loa_feat_param.PF_DYN4_s_rowcol_D_vector.')), 3, 1));

    s_kernel_normalizer         = sum(exp(-0.5 * (((px*ones(size(loa_feat_param.PF_DYN4_s_col_grid.'))) - loa_feat_param.PF_DYN4_s_col_grid.').^2) .* loa_feat_param.PF_DYN4_s_col_D_grid.'));

    if (loa_feat_param.c_order == 1)
        s_multiplier            = pv;
    else
        s_multiplier            = px;
    end

    loa_feat_matrix_per_point   = loa_feat_matrix_per_point * s_multiplier/s_kernel_normalizer;
    
    varargout(1)                = {loa_feat_matrix_per_point};
    
end