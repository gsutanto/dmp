function [ xc, yc, zc ] = computeSphereCenterUsingLeastSquares( points_on_sphere_surface )
    N_row_data      = sum(1:(size(points_on_sphere_surface,1)-1));
    X               = zeros(N_row_data, 3);
    t               = zeros(N_row_data, 1);
    idx_row_data    = 1;
    
    while (idx_row_data <= N_row_data)
        % anchor point
        pa  = points_on_sphere_surface(1,:);
        pO  = points_on_sphere_surface(2:end,:);
        for j=1:size(pO,1)
            po                      = pO(j,:);
            X_i                     = 2*(po-pa);
            t_i                     = sum(po.^2 - pa.^2);
            X(idx_row_data,:)       = X_i;
            t(idx_row_data,:)       = t_i;
            idx_row_data            = idx_row_data + 1;
        end
        
        points_on_sphere_surface    = pO;
    end
    XX  = X.'*X;
    xt  = X.'*t;
    
    reg = 1e-9;
    A   = reg*eye(size(XX,2));
    
    pc  = (A + XX)\xt;
    
    xc  = pc(1,1);
    yc  = pc(2,1);
    zc  = pc(3,1);
end

