function [ output ] = computeCrossProductMatrix( v )
    traj_length = size(v,2);
    
    v1          = v(1,:);
    v2          = v(2,:);
    v3          = v(3,:);
    
    if (traj_length == 1)
        cross_product_matrix        = [0,       -v3,        v2; ...
                                       v3,      0,          -v1; ...
                                       -v2,     v1,         0];
        output                      = cross_product_matrix;
    else
        cross_product_tensor        = zeros(3, traj_length, 3);
        cross_product_tensor(2,:,1) = v3;
        cross_product_tensor(3,:,1) = -v2;
        cross_product_tensor(1,:,2) = -v3;
        cross_product_tensor(3,:,2) = v1;
        cross_product_tensor(1,:,3) = v2;
        cross_product_tensor(2,:,3) = -v1;
        output                      = cross_product_tensor;
    end
end