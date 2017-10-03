function [ quat_err ] = computeQuatError( quat1, quat2 )
    % Author        : Giovanni Sutanto
    % Date          : October 16, 2016
    assert(size(quat1,2) == size(quat2,2),'Trajectory length is NOT equal.');
    
    etha1   = quat1(1,:);
    q1      = quat1(2:4,:);
    etha2   = quat2(1,:);
    q2      = quat2(2:4,:);
    
    traj_length = size(quat1,2);
    
    q1_cross_prod_mat   = computeCrossProductMatrix(q1);
    if (traj_length == 1)
        quat_err    = (etha1 * q2) - (etha2 * q1) - (q1_cross_prod_mat * q2);
    else
        quat_err    = (repmat(etha1,3,1) .* q2) - (repmat(etha2,3,1) .* q1) ...
                      - computeMatrixTrajectoryProduct(q1_cross_prod_mat, q2);
    end
end