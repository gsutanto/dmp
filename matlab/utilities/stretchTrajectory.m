function [ stretched_trajectory ] = stretchTrajectory( input_trajectory, new_traj_length )
    D           = size(input_trajectory, 1);
    traj_length = size(input_trajectory, 2);
    
    stretched_trajectory            = zeros(D, new_traj_length);
    
    for d = 1:D
        xi  = 1:1:traj_length;
        vi  = input_trajectory(d,:);
        xq  = 1:(traj_length-1)/(new_traj_length-1):traj_length;
        vq  = interp1(xi,vi,xq,'spline');

        stretched_trajectory(d,:)   = vq;
    end
end

