function [ quat_r ] = computeQuatProduct( quat_p, quat_q )
    % Author        : Giovanni Sutanto
    % Date          : October 16, 2016
    % Description   : Computing quat_r as follows: quat_r = quat_p * quat_q
    %                 where * is Quaternion Multiplication operator.
    assert(size(quat_p,2) == size(quat_q,2),'Trajectory length is NOT equal.');
    
    p0  = quat_p(1,:);
    p1  = quat_p(2,:);
    p2  = quat_p(3,:);
    p3  = quat_p(4,:);
    
    traj_length = size(quat_p,2);
    
    if (traj_length == 1)
        P   = [ p0, -p1, -p2, -p3; ...
                p1,  p0, -p3,  p2; ...
                p2,  p3,  p0, -p1; ...
                p3, -p2,  p1,  p0];

        quat_r  = P * quat_q;
    else
        P   = zeros(4, traj_length, 4);
        
        P(1,:,1)    = p0;
        P(2,:,1)    = p1;
        P(3,:,1)    = p2;
        P(4,:,1)    = p3;
        
        P(1,:,2)    = -p1;
        P(2,:,2)    = p0;
        P(3,:,2)    = p3;
        P(4,:,2)    = -p2;
        
        P(1,:,3)    = -p2;
        P(2,:,3)    = -p3;
        P(3,:,3)    = p0;
        P(4,:,3)    = p1;
        
        P(1,:,4)    = -p3;
        P(2,:,4)    = p2;
        P(3,:,4)    = -p1;
        P(4,:,4)    = p0;
        
        quat_r      = computeMatrixTrajectoryProduct(P, quat_q);
    end
end