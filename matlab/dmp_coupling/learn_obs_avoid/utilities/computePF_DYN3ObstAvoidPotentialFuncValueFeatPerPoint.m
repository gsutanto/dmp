function [ varargout ] = computePF_DYN3ObstAvoidPotentialFuncValueFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau )
    % Potential Function Value Computation for 3rd Dynamic Obst Avoid features (have some sense of KGF too...)
    
    x3          = endeff_state{1,1};
    v3          = endeff_state{2,1};
    
    o3          = obs_state{1,1};
    od3         = obs_state{2,1};
    
    ox3         = o3 - x3;
    rel_v3      = v3 - od3;
    
    num_thresh  = 1e-4;   % numerical threshold
    
    if ((norm(rel_v3) >= num_thresh) && (norm(ox3) >= num_thresh))
        cos_theta 	= (ox3.'*rel_v3)/(norm(ox3)*norm(rel_v3));
    
        loa_feat_vector_per_point   = tau * norm(rel_v3) * ...
                                      (exp(-0.5 * (((cos_theta*ones(size(loa_feat_param.PF_DYN3_beta_rowcoldepth_vector.'))) - loa_feat_param.PF_DYN3_beta_rowcoldepth_vector.').^2) .* loa_feat_param.PF_DYN3_beta_rowcoldepth_D_vector.') .* ...
                                       exp(-0.5 * loa_feat_param.PF_DYN3_k_rowcoldepth_vector.' * (ox3.'*ox3)));
    else
        loa_feat_vector_per_point   = zeros(1, size(loa_feat_param.PF_DYN3_beta_rowcoldepth_vector, 1));
    end
    
    varargout(1)                    = {loa_feat_vector_per_point};
    
end