function [ varargout ] = constructObsAvoidViconFeatMat( varargin )
    % Author     : Giovanni Sutanto
    % Date       : August 11, 2016
    % Description:
    %    Compute the sub-feature-matrix 
    %    (as a sub-block of a matrix) from a single demonstrated 
    %    obstacle avoidance trajectory.
    %    This will later be stacked together with those 
    %    from other demonstrations, to become a huge feature matrix 
    %    (usually called X matrix).
    
    demo_obs_avoid_traj_local        	= varargin{1};
    point_obstacles_cart_position_local = varargin{2};
    dt                                	= varargin{3};
    cart_coord_dmp_baseline_params     	= varargin{4};
    loa_feat_methods                 	= varargin{5};
    loa_feat_param                  	= varargin{6};
    
    if (nargout > 1)
        debugging_X_and_Ct_mode       	= 2;
    else
        debugging_X_and_Ct_mode        	= 0;
    end
    
    Y_obs_local     = demo_obs_avoid_traj_local{1,1};
    Yd_obs_local    = demo_obs_avoid_traj_local{2,1};
    Ydd_obs_local   = demo_obs_avoid_traj_local{3,1};

    % based on previous test on learning on synthetic dataset
    % (see for example in ../learn_obs_avoid_fixed_learning_algo/main.m ), 
    % the ground-truth feature matrix is attained if
    % the trajectory is delayed 1 time step:
    is_demo_traj_shifted    = 1;

    traj_length     = size(Y_obs_local, 1);
    D               = size(Y_obs_local, 2); % dimensionality of the problem
    
    tau             = (traj_length-1)*dt;
    
    % total number of obstacle points:
    nP              = size(point_obstacles_cart_position_local, 1);

    endeff_state    = cell(3,1);
    
    Y_obs_local_shifted     = Y_obs_local;
    Yd_obs_local_shifted    = Yd_obs_local;
    Ydd_obs_local_shifted   = Ydd_obs_local;
    if (is_demo_traj_shifted)
        Y_obs_local_shifted(2:end,:)    = Y_obs_local_shifted(1:end-1,:);
        Yd_obs_local_shifted(2:end,:)   = Yd_obs_local_shifted(1:end-1,:);
        Ydd_obs_local_shifted(2:end,:)  = Ydd_obs_local_shifted(1:end-1,:);
    end
    
    [N_loa_feat_rows_per_point, N_loa_feat_cols_per_point, loa_feat_param] = getLOA_FeatureDimensionPerPoint(loa_feat_methods, D, loa_feat_param);

    sub_X   = zeros(traj_length*N_loa_feat_rows_per_point, N_loa_feat_cols_per_point);
    if (debugging_X_and_Ct_mode == 2)
        sub_X_per_obs_point_cell            = cell(nP, 1);
        for p = 1:nP
            sub_X_per_obs_point_cell{p,1}	= zeros(traj_length, N_loa_feat_rows_per_point, N_loa_feat_cols_per_point);
        end
    end
    
    % initialize phase variables:
    px    	= 1.0;  % phase variable's x
    pv     	= 0.0;  % phase variable's v
    
    % First unroll the obstacle avoidance trajectory:
    for i=1:traj_length
        endeff_state{1,1}   = Y_obs_local_shifted(i,:)';    % end-effector position
        endeff_state{2,1}   = Yd_obs_local_shifted(i,:)';   % end-effector velocity
        endeff_state{3,1}   = Ydd_obs_local_shifted(i,:)';  % end-effector acceleration
        
        start_row_idx       = ((i-1)*N_loa_feat_rows_per_point) + 1;
        end_row_idx         = i*N_loa_feat_rows_per_point;
        
        if (debugging_X_and_Ct_mode == 2)
            [ sub_X_per_traj_point, ...
              sub_X_per_obs_point_cell ]= computeObsAvoidCtFeat( point_obstacles_cart_position_local, ...
                                                                 loa_feat_param, ...
                                                                 endeff_state, ...
                                                                 tau, ...
                                                                 N_loa_feat_rows_per_point, ...
                                                                 N_loa_feat_cols_per_point, ...
                                                                 loa_feat_methods, ...
                                                                 px, pv, ...
                                                                 cart_coord_dmp_baseline_params, ...
                                                                 sub_X_per_obs_point_cell, i );
        else
            [ sub_X_per_traj_point ]	= computeObsAvoidCtFeat( point_obstacles_cart_position_local, ...
                                                                 loa_feat_param, ...
                                                                 endeff_state, ...
                                                                 tau, ...
                                                                 N_loa_feat_rows_per_point, ...
                                                                 N_loa_feat_cols_per_point, ...
                                                                 loa_feat_methods, ...
                                                                 px, pv, ...
                                                                 cart_coord_dmp_baseline_params );
        end
        sub_X(start_row_idx:end_row_idx,:)  = sub_X_per_traj_point;
        
        if (i > 1)
            % update phase variable/canonical state
            if (loa_feat_param.c_order == 1)
                pvd = (loa_feat_param.alpha_v*(loa_feat_param.beta_v*(0-px)-pv))*0.5/tau;
                pxd = pv*0.5/tau;
            else
                pvd = 0;
                pxd = (loa_feat_param.alpha_x*(0-px))*0.5/tau;
            end
            px  = pxd*dt+px;
            if (px < 0)
                display('WARNING: x- computation')
            end
            pv  = pvd*dt+pv;
        end
    end
    
    varargout(1)        = {sub_X};
    if (debugging_X_and_Ct_mode == 2)
        varargout(2)    = {sub_X_per_obs_point_cell};
    end
end
