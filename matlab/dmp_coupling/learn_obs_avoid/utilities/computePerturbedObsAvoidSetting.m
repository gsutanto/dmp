function [ varargout ] = computePerturbedObsAvoidSetting( varargin )
    anchor_T_global_to_local_H                      = varargin{1};
    anchor_T_local_to_global_H                      = varargin{2};
    anchor_mean_start_local                         = varargin{3};
    anchor_mean_goal_local                          = varargin{4};
    considered_baseline_traj_global                 = varargin{5};
    considered_demo_obs_avoid_traj_global           = varargin{6};
    considered_point_obstacles_cart_position_global = varargin{7};
    cart_coord_dmp_baseline_params                  = varargin{8};
    local_z_axis_rotation_angle_perturbation        = varargin{9};
    ctraj_local_coordinate_frame_selection          = varargin{10};
    if (nargin > 10)
        is_plotting_settings                        = varargin{11};
    else
        is_plotting_settings                        = 0;
    end
    
    vicon_marker_radius             = 15.0/2000.0;  % in meter
    critical_position_marker_radius = 30/1000.0;    % in meter
    
    gamma           = local_z_axis_rotation_angle_perturbation;
    cos_gamma       = cos(gamma);
    sin_gamma       = sin(gamma);
    
    T_local_to_perturbed_H      = zeros(4);
    T_local_to_perturbed_H(1,1) = cos_gamma;
    T_local_to_perturbed_H(1,2) = sin_gamma;
    T_local_to_perturbed_H(2,1) = -sin_gamma;
    T_local_to_perturbed_H(2,2) = cos_gamma;
    T_local_to_perturbed_H(3,3) = 1.0;
    T_local_to_perturbed_H(4,4) = 1.0;
    
    [ considered_baseline_traj_local ]  = convertCTrajAtOldToNewCoordSys( considered_baseline_traj_global, ...
                                                                          anchor_T_global_to_local_H );
    
    [ considered_demo_obs_avoid_traj_local ] = convertCTrajAtOldToNewCoordSys( considered_demo_obs_avoid_traj_global, ...
                                                                               anchor_T_global_to_local_H );
    
    [ considered_point_obstacles_cart_position_local ]  = convertCTrajAtOldToNewCoordSys( considered_point_obstacles_cart_position_global, ...
                                                                                          anchor_T_global_to_local_H );
    
    % now consider if the representation in local coordinate system is
    % actually in the perturbed coordinate system:
    perturbed_baseline_traj_local                   = considered_baseline_traj_local;
    perturbed_demo_obs_avoid_traj_local             = considered_demo_obs_avoid_traj_local;
    perturbed_point_obstacles_cart_position_local   = considered_point_obstacles_cart_position_local;
    
    T_perturbed_to_local_H  = T_local_to_perturbed_H.';
    T_perturbed_to_global_H = anchor_T_local_to_global_H * T_perturbed_to_local_H;
    T_global_to_perturbed_H = T_local_to_perturbed_H * anchor_T_global_to_local_H;
    
    [ perturbed_baseline_traj_global ]  = convertCTrajAtOldToNewCoordSys( perturbed_baseline_traj_local, ...
                                                                          T_perturbed_to_global_H );
    
    [ perturbed_demo_obs_avoid_traj_global ]= convertCTrajAtOldToNewCoordSys( perturbed_demo_obs_avoid_traj_local, ...
                                                                              T_perturbed_to_global_H );
                                                                          
    [ perturbed_point_obstacles_cart_position_global ]  = convertCTrajAtOldToNewCoordSys( perturbed_point_obstacles_cart_position_local, ...
                                                                                          T_perturbed_to_global_H );
    
    if (ctraj_local_coordinate_frame_selection == 0)
        cart_coord_dmp_perturbed_params.T_perturbed_to_global_H = eye(4);
        cart_coord_dmp_perturbed_params.T_global_to_perturbed_H = eye(4);
    else
        cart_coord_dmp_perturbed_params.T_perturbed_to_global_H = T_perturbed_to_global_H;
        cart_coord_dmp_perturbed_params.T_global_to_perturbed_H = T_global_to_perturbed_H;
    end
    [cart_coord_dmp_perturbed_params.mean_start_global]     = convertCTrajAtOldToNewCoordSys(anchor_mean_start_local.', ...
                                                                                             T_perturbed_to_global_H);
    [cart_coord_dmp_perturbed_params.mean_goal_global]      = convertCTrajAtOldToNewCoordSys(anchor_mean_goal_local.', ...
                                                                                             T_perturbed_to_global_H);
    cart_coord_dmp_perturbed_params.mean_start_global       = cart_coord_dmp_perturbed_params.mean_start_global.';
    cart_coord_dmp_perturbed_params.mean_goal_global        = cart_coord_dmp_perturbed_params.mean_goal_global.';

    [cart_coord_dmp_perturbed_params.mean_start_perturbed]  = convertCTrajAtOldToNewCoordSys(cart_coord_dmp_perturbed_params.mean_start_global.', ...
                                                                                             cart_coord_dmp_perturbed_params.T_global_to_perturbed_H);
    [cart_coord_dmp_perturbed_params.mean_goal_perturbed]   = convertCTrajAtOldToNewCoordSys(cart_coord_dmp_perturbed_params.mean_goal_global.', ...
                                                                                             cart_coord_dmp_perturbed_params.T_global_to_perturbed_H);
    cart_coord_dmp_perturbed_params.mean_start_perturbed    = cart_coord_dmp_perturbed_params.mean_start_perturbed.';
    cart_coord_dmp_perturbed_params.mean_goal_perturbed     = cart_coord_dmp_perturbed_params.mean_goal_perturbed.';
    
    cart_coord_dmp_perturbed_params.perturbed_demo_obs_avoid_traj_global            = perturbed_demo_obs_avoid_traj_global;
    cart_coord_dmp_perturbed_params.perturbed_point_obstacles_cart_position_global  = perturbed_point_obstacles_cart_position_global;
    cart_coord_dmp_perturbed_params.perturbed_baseline_traj_global                  = perturbed_baseline_traj_global;
    
    if (is_plotting_settings == 1)  % 3D Plot:
        h_3D    = figure;
        axis equal;
        hold on;
            pct  = plot3(considered_demo_obs_avoid_traj_global{1,1}(:,1),...
                         considered_demo_obs_avoid_traj_global{1,1}(:,2),...
                         considered_demo_obs_avoid_traj_global{1,1}(:,3),...
                         'b');
            plot_sphere(critical_position_marker_radius,...
                        considered_demo_obs_avoid_traj_global{1,1}(end,1),...
                        considered_demo_obs_avoid_traj_global{1,1}(end,2),...
                        considered_demo_obs_avoid_traj_global{1,1}(end,3));

            ppt  = plot3(perturbed_demo_obs_avoid_traj_global{1,1}(:,1),...
                         perturbed_demo_obs_avoid_traj_global{1,1}(:,2),...
                         perturbed_demo_obs_avoid_traj_global{1,1}(:,3),...
                         'g');
            plot_sphere(critical_position_marker_radius,...
                        perturbed_demo_obs_avoid_traj_global{1,1}(end,1),...
                        perturbed_demo_obs_avoid_traj_global{1,1}(end,2),...
                        perturbed_demo_obs_avoid_traj_global{1,1}(end,3));
                    
            plot_sphere(critical_position_marker_radius,...
                        cart_coord_dmp_perturbed_params.mean_start_global(1,1),...
                        cart_coord_dmp_perturbed_params.mean_start_global(2,1),...
                        cart_coord_dmp_perturbed_params.mean_start_global(3,1));
                    
            plot_sphere(critical_position_marker_radius,...
                        cart_coord_dmp_perturbed_params.mean_goal_global(1,1),...
                        cart_coord_dmp_perturbed_params.mean_goal_global(2,1),...
                        cart_coord_dmp_perturbed_params.mean_goal_global(3,1));

            for op=1:size(considered_point_obstacles_cart_position_global,1)
                plot_sphere(vicon_marker_radius,...
                            considered_point_obstacles_cart_position_global(op,1),...
                            considered_point_obstacles_cart_position_global(op,2),...
                            considered_point_obstacles_cart_position_global(op,3));
                plot_sphere(vicon_marker_radius,...
                            perturbed_point_obstacles_cart_position_global(op,1),...
                            perturbed_point_obstacles_cart_position_global(op,2),...
                            perturbed_point_obstacles_cart_position_global(op,3));
            end
            legend([pct, ppt], ...
                   'original', 'perturbed');
        hold off;
    end
                                                                                      
    varargout(1)    = {cart_coord_dmp_perturbed_params};
end

