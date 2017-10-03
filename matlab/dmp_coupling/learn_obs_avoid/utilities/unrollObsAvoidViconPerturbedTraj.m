function [ varargout ] = unrollObsAvoidViconPerturbedTraj( varargin )
    % Author     : Giovanni Sutanto
    % Date       : September 13, 2016
    
    dt                                      = varargin{1};
    cart_coord_dmp_baseline_params          = varargin{2};
    loa_feat_methods                        = varargin{3};
    loa_feat_param                          = varargin{4};
    learning_param                          = varargin{5};
    unrolling_param                         = varargin{6};
    cart_coord_dmp_perturbed_params       	= varargin{7};
    
    global              dcps;
    
    traj_length_unroll  = size(cart_coord_dmp_perturbed_params.perturbed_demo_obs_avoid_traj_global{1,1}, 1);
    tau_unroll          = (traj_length_unroll - 1) * dt;
    D                   = size(cart_coord_dmp_baseline_params.w, 2);
    
    [ point_obstacles_cart_position_local ]	= convertCTrajAtOldToNewCoordSys( cart_coord_dmp_perturbed_params.perturbed_point_obstacles_cart_position_global, ...
                                                                              cart_coord_dmp_perturbed_params.T_global_to_perturbed_H );

    Y_unroll_local      = zeros(traj_length_unroll,D);
    Yd_unroll_local     = zeros(traj_length_unroll,D);
    Ydd_unroll_local    = zeros(traj_length_unroll,D);
    
    Ct_unroll         	= zeros(traj_length_unroll,D);
    
    if (unrolling_param.verify_NN_inference == 1)
        NN_net_inference_diff   = zeros(traj_length_unroll,1);
    end
    
    x3              = zeros(3,1);   % end-effector position
    v3              = zeros(3,1);   % end-effector velocity
    a3              = zeros(3,1);   % end-effector acceleration
    
    endeff_state    = cell(3,1);

    [N_loa_feat_rows_per_point, N_loa_feat_cols_per_point, loa_feat_param] = getLOA_FeatureDimensionPerPoint(loa_feat_methods, D, loa_feat_param);

    for d=1:D
        dcp_franzi('init', d, cart_coord_dmp_baseline_params.n_rfs, ...
                   num2str(d), cart_coord_dmp_baseline_params.c_order);
        dcp_franzi('reset_state', d, cart_coord_dmp_perturbed_params.mean_start_perturbed(d,1));
        dcp_franzi('set_goal', d, cart_coord_dmp_perturbed_params.mean_goal_perturbed(d,1), 1);

        dcps(d).w   = cart_coord_dmp_baseline_params.w(:,d);
    end
    
    % initialize x3 and v3:
    x3(:,1)         = cart_coord_dmp_perturbed_params.mean_start_perturbed(:,1);
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
            goal_position_local     = cart_coord_dmp_perturbed_params.mean_goal_perturbed;
            if (unrolling_param.verify_NN_inference == 1)
                [ct, NN_net_inference_diff(i,1)]    = computeNNObsAvoidCt( point_obstacles_cart_position_local, ...
                                                                           endeff_state, x, learning_param, px, pv, ...
                                                                           goal_position_local );
            else
                ct    	= computeNNObsAvoidCt( point_obstacles_cart_position_local, ...
                                               endeff_state, x, learning_param, px, pv, ...
                                               goal_position_local );
            end
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
    
    if (unrolling_param.verify_NN_inference == 1)
        disp(['max(NN_net_inference_diff) = ', num2str(max(NN_net_inference_diff))]);
    end
    
    unroll_traj_local{1,1}  = Y_unroll_local;
    unroll_traj_local{2,1}  = Yd_unroll_local;
    unroll_traj_local{3,1}  = Ydd_unroll_local;
    
    [ unroll_traj_global ] = convertCTrajAtOldToNewCoordSys( unroll_traj_local, ...
                                                             cart_coord_dmp_perturbed_params.T_perturbed_to_global_H );
    
    varargout(1)        = {Ct_unroll};
    varargout(2)        = {unroll_traj_global};
    varargout(3)        = {unroll_traj_local};  
end