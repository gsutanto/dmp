function [ points ] = getPointsFromSphereObs( obs_center, radius, end_effector, num_thresh )
    points      = zeros(3,2);
    
    points(:,1) = obs_center;
    
    obsctr2ee   = end_effector - obs_center;
    if (norm(obsctr2ee) > num_thresh)
        normalized_obsctr2ee= obsctr2ee/norm(obsctr2ee);
        
        % point on the sphere obstacle's surface, which is closest to the
        % end-effector:
        points(:,2)         = obs_center + (radius * normalized_obsctr2ee);
    end
end

