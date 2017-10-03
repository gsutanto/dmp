function [ normalized_closest_distance_to_obs_traj, final_distance_to_goal ] = measure2nd3rdPerformanceMetricTrainedSetting( traj_global, point_obstacles_cart_position_global, cart_coord_dmp_baseline_params )
    % Author: Giovanni Sutanto
    % Date  : September 09, 2016
    % Description:
    % normalized_closest_distance_to_obs_traj measures the
    %    distance of the end-effector (each point on the trajectory) 
    %    from a closest point on the virtual sphere surface
    %    formed by the obstacle's center and radius being the average distance
    %    between obstacle's center and all points of the obstacle, 
    %    normalized (divided) by the above-defined radius 
    %    (value is positive if end-effector is outside of 
    %    this virtual sphere's geometry/volume, and negative otherwise).
    % final_distance_to_goal is a measure of convergence of the
    %    end-effector trajectory to the goal position.
    
    Y_global                = traj_global{1,1};
    traj_length             = size(Y_global, 1);
    virtual_sphere_center   = mean(point_obstacles_cart_position_global, 1);
    virtual_sphere_radius   = mean(sqrt(sum(((repmat(virtual_sphere_center, size(point_obstacles_cart_position_global, 1), 1) - point_obstacles_cart_position_global).^2), 2)));
    diff_vector             = Y_global - repmat(virtual_sphere_center, traj_length, 1);
    normalized_closest_distance_to_obs_traj = zeros(traj_length, 1);
    for i=1:traj_length
        normalized_closest_distance_to_obs_traj(i,1)    = (norm(diff_vector(i,:)) - virtual_sphere_radius)/virtual_sphere_radius;
    end
    final_distance_to_goal  = norm(Y_global(end,:) - cart_coord_dmp_baseline_params.mean_goal_global.');
end