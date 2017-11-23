function [ cart_coord_dmp_params_complete ] = completeCartCoordDMPParams( cart_coord_dmp_params_basic,...
                                                                          unroll_cart_coord_params_basic )
    % Author: Giovanni Sutanto
    % Date  : February 2017
    % Description:
    %   Given   global  coordinate system information, 
    %   compute local   coordinate system information for DMP.
    
    cart_coord_dmp_critical_states_learn_global             = zeros(2,3);
    cart_coord_dmp_critical_states_learn_global(1,:)        = unroll_cart_coord_params_basic.mean_start_global.';
    cart_coord_dmp_critical_states_learn_global(2,:)        = unroll_cart_coord_params_basic.mean_goal_global.';
    cart_coord_dmp_critical_states_learn_global_traj        = cell(3,1);
    cart_coord_dmp_critical_states_learn_global_traj{1,1}   = cart_coord_dmp_critical_states_learn_global;
    cart_coord_dmp_critical_states_learn_global_traj{2,1}   = zeros(2,3);   % velocity
    if (isfield(unroll_cart_coord_params_basic, 'yd0_global'))
        cart_coord_dmp_critical_states_learn_global_traj{2,1}(1,:)  = unroll_cart_coord_params_basic.yd0_global.';
    end
    cart_coord_dmp_critical_states_learn_global_traj{3,1}   = zeros(2,3);   % acceleration
    if (isfield(unroll_cart_coord_params_basic, 'ydd0_global'))
        cart_coord_dmp_critical_states_learn_global_traj{3,1}(1,:)  = unroll_cart_coord_params_basic.ydd0_global.';
    end
    
    [cart_coord_dmp_critical_states_learn_local_traj, ...
     T_mean_local_to_global_H, ...
     T_mean_global_to_local_H] = computeCartLocalTraj(cart_coord_dmp_critical_states_learn_global_traj, 0, ...
                                                      unroll_cart_coord_params_basic.ctraj_local_coordinate_frame_selection);
    
    mean_start_local    = cart_coord_dmp_critical_states_learn_local_traj{1,1}(1,:).';
    mean_goal_local     = cart_coord_dmp_critical_states_learn_local_traj{1,1}(end,:).';
    yd0_local           = cart_coord_dmp_critical_states_learn_local_traj{2,1}(1,:).';
    ydend_local        	= cart_coord_dmp_critical_states_learn_local_traj{2,1}(end,:).';
    ydd0_local          = cart_coord_dmp_critical_states_learn_local_traj{3,1}(1,:).';
    yddend_local      	= cart_coord_dmp_critical_states_learn_local_traj{3,1}(end,:).';
    
    cart_coord_dmp_params_complete.dt                   = cart_coord_dmp_params_basic.dt;
    cart_coord_dmp_params_complete.n_rfs                = cart_coord_dmp_params_basic.n_rfs;
    cart_coord_dmp_params_complete.c_order              = cart_coord_dmp_params_basic.c_order;
    cart_coord_dmp_params_complete.w                    = cart_coord_dmp_params_basic.w;
    cart_coord_dmp_params_complete.dG                   = cart_coord_dmp_params_basic.dG;
    cart_coord_dmp_params_complete.mean_tau             = unroll_cart_coord_params_basic.mean_tau;
    cart_coord_dmp_params_complete.mean_start_global    = unroll_cart_coord_params_basic.mean_start_global;
    cart_coord_dmp_params_complete.mean_goal_global     = unroll_cart_coord_params_basic.mean_goal_global;
    cart_coord_dmp_params_complete.mean_start_local     = mean_start_local;
    cart_coord_dmp_params_complete.mean_goal_local      = mean_goal_local;
    if (isfield(unroll_cart_coord_params_basic, 'yd0_global'))
        cart_coord_dmp_params_complete.yd0_global       = unroll_cart_coord_params_basic.yd0_global;
        cart_coord_dmp_params_complete.yd0_local        = yd0_local;
    end
    if (isfield(unroll_cart_coord_params_basic, 'ydd0_global'))
        cart_coord_dmp_params_complete.ydd0_global      = unroll_cart_coord_params_basic.ydd0_global;
        cart_coord_dmp_params_complete.ydd0_local       = ydd0_local;
    end
    cart_coord_dmp_params_complete.ctraj_local_coordinate_frame_selection	= unroll_cart_coord_params_basic.ctraj_local_coordinate_frame_selection;
    cart_coord_dmp_params_complete.T_local_to_global_H 	= T_mean_local_to_global_H;
    cart_coord_dmp_params_complete.T_global_to_local_H  = T_mean_global_to_local_H;
end