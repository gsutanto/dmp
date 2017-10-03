function [retval] = detectPathDiscontinuity(path, path_abs_diff_threshold)
    path_abs_diff   = abs(diff(path,1,1));
    if (max(max(path_abs_diff)) >= path_abs_diff_threshold)
        retval  = 1;
    else
        retval  = 0;
    end
end