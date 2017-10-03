function [ local_minima_idx, local_minima_value ] = getLocalMinima( function_values, start_idx )
    % Author        : Giovanni Sutanto
    % Date          : February 2017
    % Description   : Line search to refine the local minima index (which
    %                 is initialized by start_idx). function_values is
    %                 1-dimensional.
    assert(length(size(function_values)) == 2, 'function_values (input tensor) must be 2-dimensional!');
    assert(min(size(function_values)) == 1,    'function_values (input matrix) must be 1-dimensional, i.e. a vector!');
    assert(((start_idx >= 1) && (start_idx <= length(function_values))), 'start_idx is out of range!');
    
    local_minima_idx    = start_idx;
    local_minima_value  = function_values(local_minima_idx);
    previous_value      = max(function_values(min(local_minima_idx+1,length(function_values))), ...
                              function_values(max(1,local_minima_idx-1)));
    while (previous_value > local_minima_value)
        previous_value  = local_minima_value;
        local_gradient  = (function_values(min(local_minima_idx+1,length(function_values))) - ...
                           function_values(max(1,local_minima_idx-1))) / ...
                          (min(local_minima_idx+1,length(function_values)) - ...
                           max(1,local_minima_idx-1));
        gradient_direction  = -round(local_gradient/norm(local_gradient));
        temp                = local_minima_idx + gradient_direction;
        if ((temp >= 1) && (temp <= length(function_values)))
            local_minima_idx    = temp;
            local_minima_value  = function_values(local_minima_idx);
        else
            break;
        end
    end
end