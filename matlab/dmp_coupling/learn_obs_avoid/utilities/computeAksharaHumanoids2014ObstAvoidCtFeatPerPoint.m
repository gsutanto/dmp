function [ varargout ] = computeAksharaHumanoids2014ObstAvoidCtFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau )
    % also might be sometimes referred as Humanoids'14 ObstAvoid Ct
    
    x3          = endeff_state{1,1};
    v3          = endeff_state{2,1};
    
    o3          = obs_state{1,1};
    od3         = obs_state{2,1};
    
    ox3         = o3 - x3;
    
    num_thresh  = 1e-4;   % numerical threshold
    
    if ((norm(v3) >= num_thresh) && (norm(ox3) >= num_thresh))
        theta       = abs(acos((ox3.'*v3)/(norm(ox3)*norm(v3))));
    else
        theta       = 0.0;
    end
    
    rot_vec         = cross(ox3,v3);
    rot_vec_norm    = norm(rot_vec);
    if (rot_vec_norm >= num_thresh)
        normed_rot_vec  = rot_vec/rot_vec_norm;
        R               = vrrotvec2mat([normed_rot_vec.', pi/2]);
    else
        R               = zeros(3,3);
    end
    
    loa_feat_phi1_or_phi2_matrix_per_point = tau * R * (v3-od3) * (pi-theta) * theta * ...
                                             (exp(-loa_feat_param.AF_H14_beta_phi1_phi2_vector.' * theta) .* ...
                                              exp(-loa_feat_param.AF_H14_k_phi1_phi2_vector.' * (ox3.'*ox3)));
    
    if (loa_feat_param.AF_H14_N_k_phi3_grid > 0)
        loa_feat_phi3_matrix_per_point = tau * R * (v3-od3) * exp(-loa_feat_param.AF_H14_k_phi3_vector.' * (ox3.'*ox3));
    end
    
    if (nargout == 1)
        if (loa_feat_param.AF_H14_N_k_phi3_grid == 0)
            varargout(1)= {loa_feat_phi1_or_phi2_matrix_per_point};
        elseif (loa_feat_param.AF_H14_N_k_phi3_grid > 0)
            varargout(1)= {[loa_feat_phi1_or_phi2_matrix_per_point, loa_feat_phi3_matrix_per_point]};
        end
    elseif (nargout == 2)
        varargout(1)    = {loa_feat_phi1_or_phi2_matrix_per_point};
        if (loa_feat_param.AF_H14_N_k_phi3_grid > 0)
            varargout(2)= {loa_feat_phi3_matrix_per_point};
        end
    end
    
end