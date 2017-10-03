function [ cart_coord_Ct_target ]  = computeCartCoordDMPCtTargetAtNewPositionRetainOrientation( cart_coord_demo_coupled_traj_global,...
                                                                                                cart_coord_dmp_baseline_params )
    new_T_local_seen_from_global_H          = cart_coord_dmp_baseline_params.T_local_to_global_H;
    new_T_local_seen_from_global_H(1:3,4)   = cart_coord_demo_coupled_traj_global{1,1}(1,:).';
    new_T_global_seen_from_local_H          = inv(new_T_local_seen_from_global_H);
    [ cart_coord_demo_coupled_traj_local ]  = convertCTrajAtOldToNewCoordSys( cart_coord_demo_coupled_traj_global, ...
                                                                              new_T_global_seen_from_local_H );
    
    Yc_local    = cart_coord_demo_coupled_traj_local{1,1};
    Ycd_local   = cart_coord_demo_coupled_traj_local{2,1};
    Ycdd_local  = cart_coord_demo_coupled_traj_local{3,1};
    [ cart_coord_Ct_target, ~ ] = computeDMPCtTarget(   Yc_local, ...
                                                        Ycd_local, ...
                                                        Ycdd_local, ...
                                                        cart_coord_dmp_baseline_params.w, ...
                                                        cart_coord_dmp_baseline_params.n_rfs, ...
                                                        cart_coord_dmp_baseline_params.mean_start_local, ...
                                                        cart_coord_dmp_baseline_params.mean_goal_local, ...
                                                        cart_coord_dmp_baseline_params.dt, ...
                                                        cart_coord_dmp_baseline_params.c_order );
end