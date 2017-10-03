function [ local_maxima_idx, local_maxima_value ] = getLocalMaxima( function_values, start_idx )
    % Author        : Giovanni Sutanto
    % Date          : February 2017
    % Description   : Line search to refine the local maxima index (which
    %                 is initialized by start_idx). function_values is
    %                 1-dimensional.
    assert(length(size(function_values)) == 2, 'function_values (input tensor) must be 2-dimensional!');
    assert(min(size(function_values)) == 1,    'function_values (input matrix) must be 1-dimensional, i.e. a vector!');
    assert(((start_idx >= 1) && (start_idx <= length(function_values))), 'start_idx is out of range!');
    
    local_maxima_idx    = start_idx;
    local_maxima_value  = function_values(local_maxima_idx);
    previous_value      = min(function_values(min(local_maxima_idx+1,length(function_values))), ...
                              function_values(max(1,local_maxima_idx-1)));
    while (previous_value < local_maxima_value)
        previous_value  = local_maxima_value;
        local_gradient  = (function_values(min(local_maxima_idx+1,length(function_values))) - ...
                           function_values(max(1,local_maxima_idx-1))) / ...
                          (min(local_maxima_idx+1,length(function_values)) - ...
                           max(1,local_maxima_idx-1));
        gradient_direction  = round(local_gradient/norm(local_gradient));
        temp                = local_maxima_idx + gradient_direction;
        if ((temp >= 1) && (temp <= length(function_values)))
            local_maxima_idx    = temp;
            local_maxima_value  = function_values(local_maxima_idx);
        else
            break;
        end
    end
end