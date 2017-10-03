function [ v3 ] = convertSkewSymmMatrixToVector3( skew_symm_mat )
    zero_thresh     = 1e-10;
    assert(abs(skew_symm_mat(1,1)) < zero_thresh);
    assert(abs(skew_symm_mat(2,2)) < zero_thresh);
    assert(abs(skew_symm_mat(3,3)) < zero_thresh);
    assert(abs(skew_symm_mat(1,2) + skew_symm_mat(2,1)) < zero_thresh);
    assert(abs(skew_symm_mat(1,3) + skew_symm_mat(3,1)) < zero_thresh);
    assert(abs(skew_symm_mat(2,3) + skew_symm_mat(3,2)) < zero_thresh);
    
    v3      = [skew_symm_mat(3,2);
               skew_symm_mat(1,3);
               skew_symm_mat(2,1)];
end

