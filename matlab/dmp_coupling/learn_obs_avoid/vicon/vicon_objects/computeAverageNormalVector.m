function [ ave_normal_vector ] = computeAverageNormalVector( points_on_top_surface )
    count_normal_vectors            = 0;
    sum_normal_vector               = zeros(1,3);
    
    while (size(points_on_top_surface,1) >= 3)
        % anchor point
        pa  = points_on_top_surface(1,:);
        
        % children points
        pC  = points_on_top_surface(2:end,:);
        
        % for next iteration
        points_on_top_surface    = pC;
        
        while (size(pC,1) >= 2)
            pc1                     = pC(1,:);
            pC2                     = pC(2:end,:);
            pC                      = pC2;
            for idx_c2 = 1:size(pC2,1)
                pc2                 = pC2(idx_c2,:);
                normal_vector       = cross((pc1-pa),(pc2-pa));
                normal_vector       = normal_vector/norm(normal_vector);
                sum_normal_vector   = sum_normal_vector + normal_vector;
                count_normal_vectors= count_normal_vectors + 1;
            end
        end
    end
    
    ave_normal_vector               = sum_normal_vector/count_normal_vectors;
end