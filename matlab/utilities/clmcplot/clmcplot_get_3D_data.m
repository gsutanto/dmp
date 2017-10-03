function [x,y,z] = clmcplot_get_3D_data(fname,x_name,y_name,z_name)
    [D,vars,freq]   = clmcplot_convert(fname);
    [x,y,z]         = clmcplot_getvariables(D, vars, {x_name,y_name,z_name});
    
    % Searching for null-data (zeros) clipping index of [x,y,z]:
    clipping_idx    = size(x,1);
    while ((x(clipping_idx,1)==0) && (y(clipping_idx,1)==0) && (z(clipping_idx,1)==0))
        clipping_idx= clipping_idx - 1;
    end
    
    % Clipping null-data (zeros) from [x,y,z]:
    x               = x(1:clipping_idx,1);
    y               = y(1:clipping_idx,1);
    z               = z(1:clipping_idx,1);
end