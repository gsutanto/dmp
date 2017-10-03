function [ X, T_ox3, T_v3 ] = constructObsAvoidSphereFeatMat3D( varargin )
    Yo              = varargin{1};
    Yod             = varargin{2};
    obs             = varargin{3};
    dt              = varargin{4};
    obs_radius      = varargin{5};
    loa_feat_method = varargin{6};
    loa_feat_param  = varargin{7};

    is_y_yd_shifted = 1;

    traj_length     = size(Yo,1);
    D               = size(Yo,2);
    tau             = (traj_length-1)*dt;
    
    % number of obstacle points considered
    if (strcmp(loa_feat_param.point_feat_mode, '_OBS_POINTS_AS_SEPARATE_FEATURES_') == 1)
        nP          = 2;
    else % for '_SUM_OBS_POINTS_FEATURE_CONTRIBUTION_' or '_USE_ONLY_CLOSEST_SURFACE_OBS_POINT_'
        nP          = 1;
    end

    c3              = zeros(3,1);
    x3              = zeros(3,1);
    v3              = zeros(3,1);
    a3              = zeros(3,1);
    
    endeff_state    = cell(3,1);
    obs_state       = cell(3,1);

    % Observed variable logging:
    T_ox3           = zeros(traj_length,nP*D);
    T_v3            = zeros(traj_length,nP*D);
    
    Yo_shifted      = Yo;
    Yod_shifted     = Yod;
    if (is_y_yd_shifted)
        Yo_shifted(2:end,:) = Yo_shifted(1:end-1,:);
        Yod_shifted(2:end,:)= Yod_shifted(1:end-1,:);
    end

    N_loa_feat_vect_per_point   = getLOA_FeatureDimensionPerPoint(loa_feat_method, D, loa_feat_param);
    
    X               = zeros(traj_length, nP*N_loa_feat_vect_per_point);

    % First unroll the obstacle avoidance trajectory:
    for i=1:traj_length
        c3(:,1)     = obs;                  % sphere obstacle center
        x3(:,1)     = Yo_shifted(i,:)';
        v3(:,1)     = Yod_shifted(i,:)';
        a3(:,1)     = zeros(3,1);
        
        OPs         = getPointsFromSphereObs( c3, obs_radius, x3, 1e-5 );
        x           = zeros(1, nP*N_loa_feat_vect_per_point);

        for pn = 1:nP
            if (strcmp(loa_feat_param.point_feat_mode, '_USE_ONLY_CLOSEST_SURFACE_OBS_POINT_') == 1)
                ox3 = OPs(:,2)-x3;  % uses only point on sphere obstacle surface closest to end-effector
                obs_state{1,1}  = OPs(:,2);     % obstacle position
            else % for '_OBS_POINTS_AS_SEPARATE_FEATURES_' or '_SUM_OBS_POINTS_FEATURE_CONTRIBUTION_'
                ox3 = OPs(:,pn)-x3;
                obs_state{1,1}  = OPs(:,pn);    % obstacle position
            end
            obs_state{2,1}      = zeros(3,1);   % obstacle velocity
            obs_state{3,1}      = zeros(3,1);   % obstacle acceleration

            endeff_state{1,1}   = x3;           % end-effector position
            endeff_state{2,1}   = v3;           % end-effector velocity
            endeff_state{3,1}   = a3;           % end-effector acceleration

            T_ox3(i,(pn-1)*D+1:(pn-1)*D+D)  = ox3(:,1)';
            T_v3(i,(pn-1)*D+1:(pn-1)*D+D)   = v3(:,1)';
            
            loa_feat_matrix_per_point   = computeObsAvoidCtFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau, loa_feat_method );
            if (strcmp(loa_feat_param.feat_constraint_mode, '_UNCONSTRAINED_') == 1)
                % when '_UNCONSTRAINED_', the feature set is
                % actually a row vector:
                loa_feat_vector_per_point   = reshape(loa_feat_matrix_per_point.', 1, (size(loa_feat_matrix_per_point,1)*size(loa_feat_matrix_per_point,2)));
            end
            
            if (strcmp(loa_feat_param.point_feat_mode, '_SUM_OBS_POINTS_FEATURE_CONTRIBUTION_') == 1)
                x(1,:)              = x(1,:) + loa_feat_vector_per_point;
            else % for '_OBS_POINTS_AS_SEPARATE_FEATURES_' or '_USE_ONLY_CLOSEST_SURFACE_OBS_POINT_'
                x(1,((pn-1)*N_loa_feat_vect_per_point+1):(pn*N_loa_feat_vect_per_point)) = loa_feat_vector_per_point;
            end
        end
        X(i,:)      = x;
    end
end
