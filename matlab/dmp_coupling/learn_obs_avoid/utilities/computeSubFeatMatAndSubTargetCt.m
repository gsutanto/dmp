function [ varargout ] = computeSubFeatMatAndSubTargetCt( varargin )
    % Author     : Giovanni Sutanto
    % Date       : August 08, 2016
    % Description:
    %    Compute the sub-feature-matrix and the sub-target ct
    %    (as a sub-block of a matrix) from a single demonstrated 
    %    obstacle avoidance trajectory.
    %    This will later be stacked together with those 
    %    from other demonstrations, to become a huge feature matrix 
    %    (usually called X matrix) and a huge target ct 
    %    (target regression variable, usually a vector).
    
    demo_obs_avoid_traj_global              = varargin{1};
    point_obstacles_cart_position_global    = varargin{2};
    dt                                    	= varargin{3};
    cart_coord_dmp_baseline_params        	= varargin{4};
    loa_feat_methods                        = varargin{5};
    loa_feat_param                          = varargin{6};
    
    if (nargout > 2)
        if (nargout == 3)
            debugging_X_and_Ct_mode       	= 1;
        elseif (nargout == 4)
            debugging_X_and_Ct_mode         = 2;
        end
    else
        debugging_X_and_Ct_mode             = 0;
    end
    
    max_critical_point_distance_baseline_vs_oa_demo = 0.1;  % in meter
    
    Y_obs_global                            = demo_obs_avoid_traj_global{1,1};
    start_position_global_obs_avoid_demo    = Y_obs_global(1,:).';
    goal_position_global_obs_avoid_demo     = Y_obs_global(end,:).';
    
    % some error checking on the demonstration:
    % (distance between start position of baseline DMP and obstacle avoidance demonstration,
    %  as well as distance between goal position of baseline DMP and obstacle avoidance demonstration,
    %  both should be lower than max_tolerable_distance, otherwise the demonstrated obstacle avoidance
    %  trajectory is flawed):
    if ((norm(start_position_global_obs_avoid_demo-cart_coord_dmp_baseline_params.mean_start_global) > max_critical_point_distance_baseline_vs_oa_demo) || ...
        (norm(goal_position_global_obs_avoid_demo-cart_coord_dmp_baseline_params.mean_goal_global) > max_critical_point_distance_baseline_vs_oa_demo))
        disp('ERROR: Critical position distance between baseline and obstacle avoidance demonstration is beyond tolerable threshold!!!');
    end
    
    [ demo_obs_avoid_traj_local ] = convertCTrajAtOldToNewCoordSys( demo_obs_avoid_traj_global, ...
                                                                    cart_coord_dmp_baseline_params.T_global_to_local_H );
    
    [ point_obstacles_cart_position_local ]	= convertCTrajAtOldToNewCoordSys( point_obstacles_cart_position_global, ...
                                                                              cart_coord_dmp_baseline_params.T_global_to_local_H );
    
    if (debugging_X_and_Ct_mode == 2)
        [sub_X, sub_X_per_obs_point_cell]   = constructObsAvoidViconFeatMat( demo_obs_avoid_traj_local, ...
                                                                             point_obstacles_cart_position_local, ...
                                                                             dt, ...
                                                                             cart_coord_dmp_baseline_params, ...
                                                                             loa_feat_methods, ...
                                                                             loa_feat_param );
    else
        [sub_X]	= constructObsAvoidViconFeatMat( demo_obs_avoid_traj_local, ...
                                                 point_obstacles_cart_position_local, ...
                                                 dt, ...
                                                 cart_coord_dmp_baseline_params, ...
                                                 loa_feat_methods, ...
                                                 loa_feat_param );
    end
    
    Y_obs_local     = demo_obs_avoid_traj_local{1,1};
    Yd_obs_local    = demo_obs_avoid_traj_local{2,1};
    Ydd_obs_local   = demo_obs_avoid_traj_local{3,1};
    [sub_target_ct] = computeDMPCtTarget(   Y_obs_local, ...
                                            Yd_obs_local, ...
                                            Ydd_obs_local, ...
                                            cart_coord_dmp_baseline_params.w, ...
                                            cart_coord_dmp_baseline_params.n_rfs, ...
                                            cart_coord_dmp_baseline_params.mean_start_local, ...
                                            cart_coord_dmp_baseline_params.mean_goal_local, ...
                                            dt, ...
                                            cart_coord_dmp_baseline_params.c_order );
    if (debugging_X_and_Ct_mode > 0)
        sub_target_ct_3D= sub_target_ct;
    end
    
    if (strcmp(loa_feat_param.feat_constraint_mode, '_CONSTRAINED_') == 1)
        sub_target_ct   = reshape(sub_target_ct.', [(size(sub_target_ct,1)*size(sub_target_ct,2)),1]);
    end
                                   
    varargout(1)        = {sub_X};
    varargout(2)        = {sub_target_ct};
    if (debugging_X_and_Ct_mode > 0)
        varargout(3)    = {sub_target_ct_3D};
        if (debugging_X_and_Ct_mode == 2)
            varargout(4)= {sub_X_per_obs_point_cell};
        end
    end
end