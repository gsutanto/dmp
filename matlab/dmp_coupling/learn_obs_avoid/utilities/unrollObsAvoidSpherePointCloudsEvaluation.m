function [ varargout ] = unrollObsAvoidSpherePointCloudsEvaluation( varargin )
    % Author     : Giovanni Sutanto
    % Date       : August 12, 2016
    % Description:
    %    Given an obstacle setting, baseline primitive,
    %    obstacle avoidance coupling term parameters & weights, and 
    %    unrolling parameter (tau, etc.),
    %    unroll the trajectory.
    
    sphere_params_global          	= varargin{1};
    dt                            	= varargin{2};
    cart_coord_dmp_baseline_params  = varargin{3};
    loa_feat_methods               	= varargin{4};
    loa_feat_param                	= varargin{5};
    learning_param               	= varargin{6};
    
    if(loa_feat_methods == 7)
        net             = learning_param.net;
    end
    
    global              dcps;
    
    tau_unroll          = cart_coord_dmp_baseline_params.mean_tau;
    traj_length_unroll  = round(tau_unroll/dt) + 1;
    D                   = size(cart_coord_dmp_baseline_params.w, 2);

    point_obstacles_cart_position_global    = generatePointCloudsfromSphereParams( sphere_params_global );
    
    point_obstacles_cart_position_local     = convertCTrajAtOldToNewCoordSys( point_obstacles_cart_position_global, ...
                                                                              cart_coord_dmp_baseline_params.T_global_to_local_H );

    Y_unroll_local      = zeros(traj_length_unroll,D);
    Yd_unroll_local     = zeros(traj_length_unroll,D);
    Ydd_unroll_local    = zeros(traj_length_unroll,D);
    
    Ct_unroll       = zeros(traj_length_unroll,D);

    x3              = zeros(3,1);   % end-effector position
    v3              = zeros(3,1);   % end-effector velocity
    a3              = zeros(3,1);   % end-effector acceleration
    
    endeff_state    = cell(3,1);

    [N_loa_feat_rows_per_point, N_loa_feat_cols_per_point, loa_feat_param] = getLOA_FeatureDimensionPerPoint(loa_feat_methods, D, loa_feat_param);

    for d=1:D
        dcp_franzi('init', d, cart_coord_dmp_baseline_params.n_rfs, ...
                   num2str(d), cart_coord_dmp_baseline_params.c_order);
        dcp_franzi('reset_state', d, cart_coord_dmp_baseline_params.mean_start_local(d,1));
        dcp_franzi('set_goal', d, cart_coord_dmp_baseline_params.mean_goal_local(d,1), 1);

        dcps(d).w   = cart_coord_dmp_baseline_params.w(:,d);
    end
    
    % initialize x3 and v3:
    x3(:,1)         = cart_coord_dmp_baseline_params.mean_start_local(:,1);
    v3(:,1)         = zeros(3,1);
    a3(:,1)         = zeros(3,1);
    
    % initialize phase variables:
    px              = 1.0;  % phase variable's x
    pv              = 0.0;  % phase variable's v

    for i=1:traj_length_unroll
        endeff_state{1,1}       = x3;           % end-effector position
        endeff_state{2,1}       = v3;           % end-effector velocity
        endeff_state{3,1}       = a3;           % end-effector acceleration
    
        % compute coupling term:
        [ x ]   = computeObsAvoidCtFeat( point_obstacles_cart_position_local, ...
                                         loa_feat_param, ...
                                         endeff_state, ...
                                         tau_unroll, ...
                                         N_loa_feat_rows_per_point, ...
                                         N_loa_feat_cols_per_point, ...
                                         loa_feat_methods, ...
                                         px, pv, ...
                                         cart_coord_dmp_baseline_params );
        
        if (strcmp(learning_param.learning_constraint_mode, '_PER_AXIS_') == 1)
            ct       	= sum((x .* loa_feat_param.w.'), 2);
        elseif(loa_feat_methods == 7) % using Neural Network
            goal_position_local     = cart_coord_dmp_baseline_params.mean_goal_local;
            ct          = computeNNObsAvoidCt( point_obstacles_cart_position_local, ...
                                               endeff_state, x, learning_param, px, pv, ...
                                               goal_position_local );
        else
            ct       	= x * loa_feat_param.w;
        end
        if (strcmp(loa_feat_param.feat_constraint_mode, '_UNCONSTRAINED_') == 1)
            ct          = ct.';
        end
        Ct_unroll(i,:)  = ct.';
        for d=1:D
            [y,yd,ydd,f,px,pv]      = dcp_franzi('run',d,tau_unroll,dt,ct(d,1));

            Y_unroll_local(i,d)     = y;
            Yd_unroll_local(i,d)    = yd;
            Ydd_unroll_local(i,d)   = ydd;
        end

        x3(:,1)         = Y_unroll_local(i,:)';
        v3(:,1)         = Yd_unroll_local(i,:)';
        a3(:,1)         = Ydd_unroll_local(i,:)';
    end
    
    unroll_traj_local{1,1}  = Y_unroll_local;
    unroll_traj_local{2,1}  = Yd_unroll_local;
    unroll_traj_local{3,1}  = Ydd_unroll_local;
    
    [ unroll_traj_global ] = convertCTrajAtOldToNewCoordSys( unroll_traj_local, ...
                                                             cart_coord_dmp_baseline_params.T_local_to_global_H );
    
    [normalized_closest_distance_to_obs_traj, final_distance_to_goal] = measure2nd3rdPerformanceMetricUnseenSetting(unroll_traj_global, sphere_params_global, cart_coord_dmp_baseline_params);
                                                         
    varargout(1)    = {Ct_unroll};
    varargout(2)    = {unroll_traj_global};
    varargout(3)    = {unroll_traj_local};
    varargout(4)    = {normalized_closest_distance_to_obs_traj};
    varargout(5)    = {final_distance_to_goal};
    varargout(6)    = {point_obstacles_cart_position_global};
end