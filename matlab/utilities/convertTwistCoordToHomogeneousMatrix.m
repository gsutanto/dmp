function [ hmg_mat ] = convertTwistCoordToHomogeneousMatrix( twist )
    v               = twist(1:3,1);
    omega           = twist(4:6,1);
    omega_skew_mat  = convertVector3ToSkewSymmMatrix(omega);
    
    hmg_mat         = [omega_skew_mat, v;
                       zeros(1,4)];
end

