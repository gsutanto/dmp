function [ Ad ] = computeAdjointTransformation( T )
    R   = T(1:3,1:3);
    t   = T(1:3,4);
    
    t_hat   = computeCrossProductMatrix(t);
    Ad      = zeros(6,6);
    
    Ad(1:3,1:3) = R;
    Ad(1:3,4:6) = t_hat * R;
    Ad(4:6,4:6) = R;
end

