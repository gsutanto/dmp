function [ skew_symm_mat ] = convertVector3ToSkewSymmMatrix( v3 )
    skew_symm_mat   = computeCrossProductMatrix(v3);
end

