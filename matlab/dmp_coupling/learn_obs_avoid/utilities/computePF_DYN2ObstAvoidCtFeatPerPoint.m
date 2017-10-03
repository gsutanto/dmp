function [ loa_feat_matrix_per_point ] = computePF_DYN2ObstAvoidCtFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau )
    
    x3              = endeff_state{1,1};
    v3              = endeff_state{2,1};
    
    o3              = obs_state{1,1};
    
    ox3             = o3 - x3;
    
    num_thresh      = 1e-4; % numerical threshold
    
    if ((norm(v3) >= num_thresh) && (norm(ox3) >= num_thresh))
        loa_feat_matrix_per_point  = -tau * norm(v3) * [(((v3.'*ox3)/(norm(v3)*((norm(ox3))^3))) * ox3 * loa_feat_param.PF_DYN2_beta_vector.') - ...
                                                        ((1.0/(norm(v3)*norm(ox3))) * v3 * loa_feat_param.PF_DYN2_beta_vector.') + ...
                                                        (2.0 * ox3 * loa_feat_param.PF_DYN2_k_vector.')] .* ...
                                     repmat(exp((loa_feat_param.PF_DYN2_beta_vector.' * (v3.'*ox3/(norm(v3)*norm(ox3)))) - ...
                                                (loa_feat_param.PF_DYN2_k_vector.' * (ox3.'*ox3))), 3, 1);
    else
        loa_feat_matrix_per_point  = zeros(3, size(loa_feat_param.PF_DYN2_beta_vector, 1));
    end
end