function [ varargout ] = computeFM_KGFv01ObstAvoidCtFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau, px, pv )
    % Franzi's Kernelized General Features (KGF) version 01 
    
    x3          = endeff_state{1,1};
    v3          = endeff_state{2,1};
    
    o3          = obs_state{1,1};
    od3         = obs_state{2,1};
    
    ox3         = o3 - x3;
    
    loa_feat_matrix_per_point = (exp(-loa_feat_param.FM_KGFv01_k_rowcol_vector.' * (ox3.'*ox3)) .* ...
                                 exp(-0.5 * (((px*ones(size(loa_feat_param.FM_KGFv01_s_rowcol_vector.'))) - loa_feat_param.FM_KGFv01_s_rowcol_vector.').^2) .* loa_feat_param.FM_KGFv01_s_rowcol_D_vector.'));
                             
%     s_kernel_normalizer       = sum(exp(-0.5 * (((px*ones(size(loa_feat_param.FM_KGFv01_s_col_grid.'))) - loa_feat_param.FM_KGFv01_s_col_grid.').^2) .* loa_feat_param.FM_KGFv01_s_col_D_grid.'));
    
    if (loa_feat_param.c_order == 1)
        s_multiplier          = pv;
    else
        s_multiplier          = px;
    end
    
%     loa_feat_matrix_per_point = loa_feat_matrix_per_point * s_multiplier/s_kernel_normalizer;
    loa_feat_matrix_per_point = loa_feat_matrix_per_point * s_multiplier;
    
    varargout(1)    = {loa_feat_matrix_per_point};
    
end