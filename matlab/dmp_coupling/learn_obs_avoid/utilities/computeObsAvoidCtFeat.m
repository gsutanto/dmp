function [ varargout ] = computeObsAvoidCtFeat( varargin )
    % Author     : Giovanni Sutanto
    % Date       : September 07, 2016
    
    point_obstacles_cart_position_local = varargin{1};
    loa_feat_param                      = varargin{2};
    endeff_state                        = varargin{3};
    tau                                 = varargin{4};
    N_loa_feat_rows_per_point           = varargin{5};
    N_loa_feat_cols_per_point           = varargin{6};
    loa_feat_methods                    = varargin{7};
    px                                  = varargin{8};
    pv                                  = varargin{9};
    cart_coord_dmp_baseline_params      = varargin{10};
    if (nargout > 1)
        debugging_X_and_Ct_mode       	= 2;
        sub_X_per_obs_point_cell      	= varargin{11};
        traj_point_idx                  = varargin{12};
    else
        debugging_X_and_Ct_mode        	= 0;
    end
    
    % total number of obstacle points:
    nP                  = size(point_obstacles_cart_position_local, 1);
    
    % an obstacle point (coordinate):
    o3                  = zeros(3,1);
    
    obs_state           = cell(3,1);
    
    sub_X_per_traj_point= zeros(N_loa_feat_rows_per_point, ...
                                N_loa_feat_cols_per_point);
                            
    % gsutanto note: currently did NOT support multiple loa_feat_methods (iscell(loa_feat_methods) == 1)
    if (loa_feat_methods == 7)  % using Neural Network
        goal_position_local     = cart_coord_dmp_baseline_params.mean_goal_local;
        sub_X_per_traj_point    = computeNNObstAvoidCtFeat( loa_feat_param, endeff_state, point_obstacles_cart_position_local, tau, px, pv, goal_position_local );
    else
        for p = 1:nP
            o3(:,1)         = point_obstacles_cart_position_local(p,:)';                  % sphere obstacle center

            obs_state{1,1} 	= o3;           % obstacle position
            obs_state{2,1}	= zeros(3,1);	% obstacle velocity
            obs_state{3,1} 	= zeros(3,1);   % obstacle acceleration

            loa_feat_matrix_per_point   = (1.0/nP) * computeObsAvoidCtFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau, loa_feat_methods, px, pv );
            if (strcmp(loa_feat_param.feat_constraint_mode, '_UNCONSTRAINED_') == 1)
                % when '_UNCONSTRAINED_', the feature set is
                % actually a row vector:
                loa_feat_matrix_per_point   = reshape(loa_feat_matrix_per_point.', 1, (size(loa_feat_matrix_per_point,1)*size(loa_feat_matrix_per_point,2)));
            end

            sub_X_per_traj_point  = sub_X_per_traj_point + loa_feat_matrix_per_point;
            if (debugging_X_and_Ct_mode == 2)
                sub_X_per_obs_point_cell{p,1}(traj_point_idx,:,:)= loa_feat_matrix_per_point;
            end
        end
    end
    
    varargout(1)   	= {sub_X_per_traj_point};
    if (debugging_X_and_Ct_mode == 2)
        varargout(2)= {sub_X_per_obs_point_cell};
    end
end

