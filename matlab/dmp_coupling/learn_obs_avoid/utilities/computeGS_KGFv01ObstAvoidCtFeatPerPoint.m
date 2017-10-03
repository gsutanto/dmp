function [ varargout ] = computeGS_KGFv01ObstAvoidCtFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau, px, pv )
    % gsutanto's Kernelized General Features (KGF) version 01 
    
    x3          = endeff_state{1,1};
    v3          = endeff_state{2,1};
    
    o3          = obs_state{1,1};
    od3         = obs_state{2,1};
    
    ox3         = o3 - x3;
    rel_v3      = v3 - od3;
    
    is_normalizing_theta_kernel         = 1;
    is_using_phase_variable_multiplier  = 1;
    
    num_thresh  = 1e-4;   % numerical threshold
    
    if ((norm(rel_v3) >= num_thresh) && (norm(ox3) >= num_thresh))
        theta       = abs(acos((ox3.'*rel_v3)/(norm(ox3)*norm(rel_v3))));
    else
        theta       = 0.0;
    end
    
    rot_vec         = cross(ox3,rel_v3);
    rot_vec_norm    = norm(rot_vec);
    if (rot_vec_norm >= num_thresh)
        normed_rot_vec  = rot_vec/rot_vec_norm;
        R               = vrrotvec2mat([normed_rot_vec.', pi/2]);
    else
        R               = zeros(3,3);
    end
    
    loa_feat_matrix_per_point = tau * R * rel_v3 * ...
                                (exp(-0.5 * (((theta*ones(size(loa_feat_param.GS_KGFv01_beta_rowcoldepth_vector.'))) - loa_feat_param.GS_KGFv01_beta_rowcoldepth_vector.').^2) .* loa_feat_param.GS_KGFv01_beta_rowcoldepth_D_vector.') .* ...
                                 exp(-loa_feat_param.GS_KGFv01_k_rowcoldepth_vector.' * (ox3.'*ox3)) .* ...
                                 exp(-0.5 * (((px*ones(size(loa_feat_param.GS_KGFv01_s_rowcoldepth_vector.'))) - loa_feat_param.GS_KGFv01_s_rowcoldepth_vector.').^2) .* loa_feat_param.GS_KGFv01_s_rowcoldepth_D_vector.'));
                             
    if (is_normalizing_theta_kernel)
        theta_kernel_normalizer     = sum(exp(-0.5 * (((theta*ones(size(loa_feat_param.GS_KGFv01_beta_col_grid.'))) - loa_feat_param.GS_KGFv01_beta_col_grid.').^2) .* loa_feat_param.GS_KGFv01_beta_col_D_grid.'));
    else
        theta_kernel_normalizer     = 1.0;
    end
    
    s_kernel_normalizer = sum(exp(-0.5 * (((px*ones(size(loa_feat_param.GS_KGFv01_s_depth_grid.'))) - loa_feat_param.GS_KGFv01_s_depth_grid.').^2) .* loa_feat_param.GS_KGFv01_s_depth_D_grid.'));
    if (is_using_phase_variable_multiplier)
        if (loa_feat_param.c_order == 1)
            s_multiplier   	= pv;
        else
            s_multiplier   	= px;
        end
    else
        s_multiplier        = 1.0;
    end
    
    loa_feat_matrix_per_point = loa_feat_matrix_per_point * s_multiplier/(theta_kernel_normalizer * s_kernel_normalizer);
    
    varargout(1)    = {loa_feat_matrix_per_point};
    
end