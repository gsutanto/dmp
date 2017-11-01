function [ varargout ] = unrollObsAvoidViconTraj( varargin )
    % Author     : Giovanni Sutanto
    % Date       : August 12, 2016
    % Description:
    %    Given an obstacle setting, baseline primitive,
    %    obstacle avoidance coupling term parameters & weights, and 
    %    unrolling parameter (tau, etc.),
    %    unroll the trajectory.
    
    baseline                                = varargin{1};
    demo_obs_avoid_traj_global              = varargin{2};
    point_obstacles_cart_position_global    = varargin{3};
    dt                                      = varargin{4};
    cart_coord_dmp_baseline_params          = varargin{5};
    loa_feat_methods                        = varargin{6};
    loa_feat_param                          = varargin{7};
    learning_param                          = varargin{8};
    unrolling_param                         = varargin{9};
    if (nargin > 9)
        is_measuring_demo_performance_metric= varargin{10};
    else
        is_measuring_demo_performance_metric= 0;
    end
    if (nargin > 10)
        is_measuring_all_performance_metric = varargin{11};
    else
        is_measuring_all_performance_metric = 1;
    end
    if (nargin > 11)
        is_unrolling_ct_with_dynamics      	= varargin{12};
    else
        is_unrolling_ct_with_dynamics      	= 1;    % unrolling DMP with Ct computed online (feature's value dynamics is affected by previous values of Ct predictions)
    end
    if (is_unrolling_ct_with_dynamics == 0) % unrolling DMP with Ct computed from pre-extracted dataset (which some part of it is used in training the Ct model)
        sub_X_extracted_dataset             = varargin{13};
    end
    
    if (isfield(learning_param, 'pmnn') == 1)
        is_using_pmnn                       = 1;
    else
        is_using_pmnn                       = 0;
    end
    
    if (nargout > 9)
        debugging_X_and_Ct_mode             = 2;
    else
        debugging_X_and_Ct_mode             = 0;
    end
    
    if (isfield(unrolling_param, 'tau_multiplier') == 0)
        unrolling_param.tau_multiplier      = 1;    % default tau multiplier is 1 (unless otherwise specified)
    end
    
    global              dcps;
    
    traj_length_unroll  = size(demo_obs_avoid_traj_global{1,1}, 1);
    tau_unroll          = unrolling_param.tau_multiplier * (traj_length_unroll - 1) * dt;
    D                   = size(cart_coord_dmp_baseline_params.w, 2);
    N_unroll            = (unrolling_param.tau_multiplier * traj_length_unroll);
    
    [ point_obstacles_cart_position_local ]	= convertCTrajAtOldToNewCoordSys( point_obstacles_cart_position_global, ...
                                                                              cart_coord_dmp_baseline_params.T_global_to_local_H );

    Y_unroll_local      = zeros(traj_length_unroll,D);
    Yd_unroll_local     = zeros(traj_length_unroll,D);
    Ydd_unroll_local    = zeros(traj_length_unroll,D);
    
    Ct_unroll         	= zeros(traj_length_unroll,D);
    
    if (unrolling_param.verify_NN_inference == 1)
        NN_net_inference_diff   = zeros(traj_length_unroll,1);
    end
    
    % total number of obstacle points:
    nP              = size(point_obstacles_cart_position_local, 1);

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
        
        if (d == 1)
            psi     = dcps(d).psi.';
            pv      = dcps(d).v;
            normalized_phase_PSI_mult_phase_V   = psi .* repmat((pv ./ sum((psi+1.e-10),2)),1,cart_coord_dmp_baseline_params.n_rfs);
        end
    end
    
    % initialize x3 and v3:
    x3(:,1)         = cart_coord_dmp_baseline_params.mean_start_local(:,1);
    v3(:,1)         = zeros(3,1);
    a3(:,1)         = zeros(3,1);
    
    if (debugging_X_and_Ct_mode == 2)
        if (loa_feat_methods ~= 7)
            sub_X_unroll_per_obs_point_cell         = cell(nP, 1);
            for p = 1:nP
                sub_X_unroll_per_obs_point_cell{p,1}= zeros( traj_length_unroll, ...
                                                             N_loa_feat_rows_per_point, ...
                                                             N_loa_feat_cols_per_point );
            end
        else
            sub_X_unroll_cell                       = cell(N_unroll, 1);
        end
    end
    
    % initialize phase variables:
    px              = 1.0;  % phase variable's x
    pv              = 0.0;  % phase variable's v

    for i=1:N_unroll
        endeff_state{1,1}       = x3;           % end-effector position
        endeff_state{2,1}       = v3;           % end-effector velocity
        endeff_state{3,1}       = a3;           % end-effector acceleration
        
        if (is_unrolling_ct_with_dynamics ~= 0)
            % compute coupling term:
            if ((debugging_X_and_Ct_mode == 2) && (loa_feat_methods ~= 7))
                [ x, ...
                  sub_X_unroll_per_obs_point_cell ] = computeObsAvoidCtFeat( point_obstacles_cart_position_local, ...
                                                                             loa_feat_param, ...
                                                                             endeff_state, ...
                                                                             tau_unroll, ...
                                                                             N_loa_feat_rows_per_point, ...
                                                                             N_loa_feat_cols_per_point, ...
                                                                             loa_feat_methods, ...
                                                                             px, pv, ...
                                                                             cart_coord_dmp_baseline_params, ...
                                                                             sub_X_unroll_per_obs_point_cell, i );
            else
                [ x ]   = computeObsAvoidCtFeat( point_obstacles_cart_position_local, ...
                                                 loa_feat_param, ...
                                                 endeff_state, ...
                                                 tau_unroll, ...
                                                 N_loa_feat_rows_per_point, ...
                                                 N_loa_feat_cols_per_point, ...
                                                 loa_feat_methods, ...
                                                 px, pv, ...
                                                 cart_coord_dmp_baseline_params );
            end
        else    % if (is_unrolling_ct_with_dynamics == 0)
            x   = sub_X_extracted_dataset(i,:);
        end
        if ((debugging_X_and_Ct_mode == 2) && (loa_feat_methods == 7))
            sub_X_unroll_cell{i,1}  = x;
        end
        
        if (strcmp(learning_param.learning_constraint_mode, '_PER_AXIS_') == 1)
            ct       	= sum((x .* loa_feat_param.w.'), 2);
        elseif(loa_feat_methods == 7) % using Neural Network
            if (is_using_pmnn == 1)
                [ ct, ~, ...
                  learning_param.pmnn ] = performPMNNPrediction( learning_param.pmnn, x, normalized_phase_PSI_mult_phase_V );
            else
                goal_position_local     = cart_coord_dmp_baseline_params.mean_goal_local;
                if (unrolling_param.verify_NN_inference == 1)
                    [ct, NN_net_inference_diff(i,1)]    = computeNNObsAvoidCt( point_obstacles_cart_position_local, ...
                                                                               endeff_state, x, learning_param, px, pv, ...
                                                                               goal_position_local );
                else
                    ct    	= computeNNObsAvoidCt( point_obstacles_cart_position_local, ...
                                                   endeff_state, x, learning_param, px, pv, ...
                                                   goal_position_local );
                end
            end
        else
            ct       	= x * loa_feat_param.w;
        end
        if (strcmp(loa_feat_param.feat_constraint_mode, '_UNCONSTRAINED_') == 1)
            ct          = ct.';
        end
        Ct_unroll(i,:)  = ct.';
        for d=1:D
            [y,yd,ydd,f,px,pv,psi]  = dcp_franzi('run',d,tau_unroll,dt,ct(d,1));

            Y_unroll_local(i,d)     = y;
            Yd_unroll_local(i,d)    = yd;
            Ydd_unroll_local(i,d)   = ydd;
            
            if (d == 1)
                normalized_phase_PSI_mult_phase_V   = psi .* repmat((pv ./ sum((psi+1.e-10),2)),1,cart_coord_dmp_baseline_params.n_rfs);
            end
        end

        x3(:,1)         = Y_unroll_local(i,:)';
        v3(:,1)         = Yd_unroll_local(i,:)';
        a3(:,1)         = Ydd_unroll_local(i,:)';
    end
    
    if ((debugging_X_and_Ct_mode == 2) && (loa_feat_methods == 7))
        sub_X_unroll    = cell2mat(sub_X_unroll_cell);
    end
    
    if (unrolling_param.verify_NN_inference == 1)
        disp(['max(NN_net_inference_diff) = ', num2str(max(NN_net_inference_diff))]);
    end
    
    unroll_traj_local{1,1}  = Y_unroll_local;
    unroll_traj_local{2,1}  = Yd_unroll_local;
    unroll_traj_local{3,1}  = Ydd_unroll_local;
    
    [ unroll_traj_global ]  = convertCTrajAtOldToNewCoordSys( unroll_traj_local, ...
                                                              cart_coord_dmp_baseline_params.T_local_to_global_H );
    
    if (is_measuring_all_performance_metric == 1)
        if (is_measuring_demo_performance_metric == 1)
            % measure 2nd and 3rd performance metric of human baseline demonstration:
            [ baseline_normalized_closest_distance_to_obs_traj, baseline_final_distance_to_goal ]   = measure2nd3rdPerformanceMetricTrainedSetting( baseline , point_obstacles_cart_position_global, cart_coord_dmp_baseline_params );
            
            % measure 2nd and 3rd performance metric of human obstacle avoidance demonstration:
            [ demo_normalized_closest_distance_to_obs_traj, demo_final_distance_to_goal ]   = measure2nd3rdPerformanceMetricTrainedSetting( demo_obs_avoid_traj_global, point_obstacles_cart_position_global, cart_coord_dmp_baseline_params );
        else
            baseline_normalized_closest_distance_to_obs_traj= [];
            baseline_final_distance_to_goal                 = NaN;
            
            demo_normalized_closest_distance_to_obs_traj    = [];
            demo_final_distance_to_goal                     = NaN;
        end

        % measure 2nd and 3rd performance metric of unrolled trajectory:
        [ unroll_normalized_closest_distance_to_obs_traj, unroll_final_distance_to_goal ] = measure2nd3rdPerformanceMetricTrainedSetting( unroll_traj_global, point_obstacles_cart_position_global, cart_coord_dmp_baseline_params );
    else
        baseline_normalized_closest_distance_to_obs_traj    = [];
        baseline_final_distance_to_goal                     = NaN;
        
        demo_normalized_closest_distance_to_obs_traj        = [];
        demo_final_distance_to_goal                         = NaN;
        
        unroll_normalized_closest_distance_to_obs_traj      = [];
        unroll_final_distance_to_goal                       = NaN;
    end
    
    varargout(1)        = {Ct_unroll};
    varargout(2)        = {unroll_traj_global};
    varargout(3)        = {unroll_traj_local};
    varargout(4)        = {demo_normalized_closest_distance_to_obs_traj};
    varargout(5)        = {demo_final_distance_to_goal};
    varargout(6)        = {baseline_normalized_closest_distance_to_obs_traj};
    varargout(7)        = {baseline_final_distance_to_goal};
    varargout(8)        = {unroll_normalized_closest_distance_to_obs_traj};
    varargout(9)        = {unroll_final_distance_to_goal};
    if (debugging_X_and_Ct_mode == 2)
        if (loa_feat_methods == 7)
            varargout(10)   = {sub_X_unroll};
        else
            varargout(10)   = {sub_X_unroll_per_obs_point_cell};
        end
    else
        varargout(10)       = {[]};
    end
    if (nargout > 10)
        varargout(11)   = {learning_param.pmnn};
    end
end