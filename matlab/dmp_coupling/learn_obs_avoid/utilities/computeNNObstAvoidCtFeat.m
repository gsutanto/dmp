function [ varargout ] = computeNNObstAvoidCtFeat( loa_feat_param, endeff_state, point_obstacles_cart_position_local, tau, px, pv, goal_position_local )
    x3          = endeff_state{1,1};
    v3          = endeff_state{2,1};
    tau       	= tau/0.5; % tau is relative to 0.5 seconds (similar to tau computed in dcp_franzi.m)
    
    o3          = point_obstacles_cart_position_local;
    
    if (loa_feat_param.is_tau_invariant == 0)
        tau_v3 	= v3;
    else
        tau_v3  = tau * v3;
    end
    
    o3_center   = mean(o3);
    o3_centerx  = o3_center - x3';
    
    dists = pdist2(o3, x3');
    % sort by distance (descending order)
    [~,idx] = sort(dists);

    min_ds = o3(idx(1:3), :);

    min_rel = bsxfun(@minus, min_ds, x3');
    
    o3s = reshape(min_rel, 1, size(min_ds,1)*size(min_ds,2));
    
    
    ox3         = o3(idx(1),:) - x3';
    
    num_thresh  = 0;   % numerical threshold

    if ((norm(v3) > num_thresh) && (norm(ox3) > num_thresh))
        cos_theta               = ((ox3*v3)/(norm(ox3)*norm(v3)));
        clamped_cos_theta       = cos_theta;
        if (clamped_cos_theta < 0.0)
            clamped_cos_theta   = 0.0;
        end
        theta                   = abs(acos(cos_theta));
    else
        cos_theta               = -1.0;
        clamped_cos_theta       = 0.0;
        theta                   = pi;
    end
    
    if isnan(theta)
        keyboard;
    end
    %
    norm_dis = norm(o3_center - x3');
    
    dist_to_goal = goal_position_local - x3;
    % Five closest points     velocity    relative distance    end-effector
    % position     angle    cos(angle)    phase

%     phi = [o3_centerx o3s tau_v3' norm_dis cos_theta];
    phi = [o3_centerx o3s tau_v3' norm_dis cos_theta];

%     phi = [tau_v3' ox3 o3_centerx]; 
    loa_feat_matrix_per_point = phi; %(exp(-1.' * (ox3*ox3')) .* ...
                                        %phi);%
     
    if (loa_feat_param.c_order == 1)
        s_multiplier          = pv;
    else
        s_multiplier          = px;
    end
    
%     loa_feat_matrix_per_point = loa_feat_matrix_per_point * s_multiplier;

    
    varargout(1)    = {loa_feat_matrix_per_point};
    varargout{2} = s_multiplier;
   
end