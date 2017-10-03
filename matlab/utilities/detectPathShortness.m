function [retval] = detectPathShortness(path, path_shortness_threshold)
    path_abs_diff           = diff(path,1,1);
    euclidean_displacement  = sqrt(sum(path_abs_diff.^2,2));
    if (sum(euclidean_displacement) <= path_shortness_threshold)
        retval  = 1;
    else
        retval  = 0;
    end
end