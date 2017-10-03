function [var_matrix] = clmcplotGetNullClippedData(file_name, var_names)
    [D, vars, freq] = clmcplot_convert(file_name);
    var_matrix      = clmcplot_getvariables(D, vars, var_names);
    
    [ start_clipping_idx, end_clipping_idx ] = getNullClippingIndex( var_matrix );
    
    % Clipping null-data (zeros) from [x,y,z]:
    var_matrix      = var_matrix(start_clipping_idx:end_clipping_idx,:);
end