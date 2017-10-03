function [ normalized_closest_distance_to_obs_traj, final_distance_to_goal ] = measure2nd3rdPerformanceMetricUnseenSetting( unroll_traj_global, sphere_params_global, cart_coord_dmp_baseline_params )
    % Author: Giovanni Sutanto
    % Date  : August 22, 2016
    % Description:
    % normalized_closest_distance_to_obs_traj measures the
    %    distance of the end-effector (each point on the trajectory) 
    %    from a closest point on the sphere obstacle's surface, 
    %    normalized (divided) by the radius of the sphere obstacle 
    %    (value is positive if end-effector is outside of 
    %    the sphere obstacle's geometry/volume, and negative otherwise).
    % final_distance_to_goal is a measure of convergence of the
    %    end-effector trajectory to the goal position.
    
    Y_unroll_global = unroll_traj_global{1,1};
    traj_length     = size(Y_unroll_global, 1);
    diff_vector     = Y_unroll_global - repmat(sphere_params_global.center.',traj_length,1);
    normalized_closest_distance_to_obs_traj = zeros(traj_length, 1);
    for i=1:traj_length
        normalized_closest_distance_to_obs_traj(i,1)    = (norm(diff_vector(i,:)) - sphere_params_global.radius)/sphere_params_global.radius;
    end
    final_distance_to_goal  = norm(Y_unroll_global(end,:) - cart_coord_dmp_baseline_params.mean_goal_global.');
end

