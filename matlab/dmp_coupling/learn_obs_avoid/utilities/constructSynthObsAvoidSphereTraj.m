function [ X_gT, Yo, Ydo, Yddo, L_ox3, L_v3, Ct ] = constructSynthObsAvoidSphereTraj( varargin )
    w_DMP           = varargin{1};
    w_Ct_SYNTH      = varargin{2};
    start           = varargin{3};
    goal            = varargin{4};
    obs             = varargin{5};
    traj_length     = varargin{6};
    dt              = varargin{7};
    c_order         = varargin{8};
    obs_radius      = varargin{9};
    loa_feat_method = varargin{10};
    loa_feat_param  = varargin{11};
    
    global          dcps;
    n_rfs           = size(w_DMP,1);
    D               = size(w_DMP,2);
    tau             = (traj_length-1)*dt;
    
    % number of obstacle points considered
    if (strcmp(loa_feat_param.point_feat_mode, '_OBS_POINTS_AS_SEPARATE_FEATURES_') == 1)
        nP          = 2;
    else % for '_SUM_OBS_POINTS_FEATURE_CONTRIBUTION_' or '_USE_ONLY_CLOSEST_SURFACE_OBS_POINT_'
        nP          = 1;
    end

    Yo              = zeros(traj_length,D);
    Ydo             = zeros(traj_length,D);
    Yddo            = zeros(traj_length,D);
    Ct              = zeros(traj_length,D);

    c3              = zeros(3,1);
    x3              = zeros(3,1);
    v3              = zeros(3,1);
    a3              = zeros(3,1);
    
    endeff_state    = cell(3,1);
    obs_state       = cell(3,1);

    N_loa_feat_vect_per_point   = getLOA_FeatureDimensionPerPoint(loa_feat_method, D, loa_feat_param);
    
    X_gT            = zeros(traj_length, nP*N_loa_feat_vect_per_point);

    % latent variable logging:
    L_ox3           = zeros(traj_length,nP*D);
    L_v3            = zeros(traj_length,nP*D);

    % initialize x3 and v3:
    c3(:,1)         = obs;          % sphere obstacle center
    x3(:,1)         = start(:,1);
    v3(:,1)         = zeros(3,1);
    a3(:,1)         = zeros(3,1);

    for d=1:D
        dcp_franzi('init',d,n_rfs,num2str(d), c_order);
        dcp_franzi('reset_state',d, start(d,1));
        dcp_franzi('set_goal',d,goal(d,1),1);

        dcps(d).w   = w_DMP(:,d);
    end

    for i=1:traj_length
        OPs         = getPointsFromSphereObs( c3, obs_radius, x3, 1e-5 );
        x_gT        = zeros(1, nP*N_loa_feat_vect_per_point);

        % ox3 and v3 computed here is the "ground truth".
        % compute model-based coupling term:
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

            L_ox3(i,(pn-1)*D+1:(pn-1)*D+D)  = ox3';
            L_v3(i,(pn-1)*D+1:(pn-1)*D+D)   = v3';
            
            [loa_feat_matrix_per_point] = computeObsAvoidCtFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau, loa_feat_method );
            if (strcmp(loa_feat_param.feat_constraint_mode, '_UNCONSTRAINED_') == 1)
                % when '_UNCONSTRAINED_', the feature set is
                % actually a row vector:
                loa_feat_vector_per_point   = reshape(loa_feat_matrix_per_point.', 1, (size(loa_feat_matrix_per_point,1)*size(loa_feat_matrix_per_point,2)));
            end
            
            if (strcmp(loa_feat_param.point_feat_mode, '_SUM_OBS_POINTS_FEATURE_CONTRIBUTION_') == 1)
                x_gT(1,:)         	= x_gT(1,:) + loa_feat_vector_per_point;
            else % for '_OBS_POINTS_AS_SEPARATE_FEATURES_' or '_USE_ONLY_CLOSEST_SURFACE_OBS_POINT_'
                x_gT(1,((pn-1)*N_loa_feat_vect_per_point+1):(pn*N_loa_feat_vect_per_point)) = loa_feat_vector_per_point;
            end
        end
        X_gT(i,:)       = x_gT;
        ct              = X_gT(i,:) * w_Ct_SYNTH;
        Ct(i,:)         = ct;
        for d=1:D
            [y,yd,ydd,f] = dcp_franzi('run',d,tau,dt,ct(1,d));

            Yo(i,d)     = y;
            Ydo(i,d)    = yd;
            Yddo(i,d)   = ydd;
        end

        x3(:,1)         = Yo(i,:)';
        v3(:,1)         = Ydo(i,:)';
        a3(:,1)         = Yddo(i,:)';
    end
end
