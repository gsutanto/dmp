function [ twist ] = convertHomogeneousMatrixToTwistCoord( hmg_mat )
    assert(size(hmg_mat,1) == 4)
    assert(size(hmg_mat,2) == 4)
    assert(norm(hmg_mat(4,:)) == 0);
    
    v               = hmg_mat(1:3,4);
    omega_skew_mat  = hmg_mat(1:3,1:3);
    omega           = convertSkewSymmMatrixToVector3(omega_skew_mat);
    
    twist           = [v;
                       omega];
end

