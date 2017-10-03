function [ log_R_output ] = computeLogMapRotMat( R_input )
    % Author        : Giovanni Sutanto
    % Date          : July 2017
    thresh  = 1e-5;
    
    assert(size(R_input,1) == 3, 'Invalid 3-D rotation matrix R! # of rows    of R_input must be 3!');
    assert(size(R_input,2) == 3, 'Invalid 3-D rotation matrix R! # of columns of R_input must be 3!');
    assert(abs(det(R_input) - 1) < thresh, 'Invalid rotation matrix R_input! det(R_input) must be 1!');
    
    if (isdiag(R_input) && all(diag(R_input)==1))
        log_R_output    = zeros(3,1);
    else
        theta           = acos((trace(R_input) - 1)/2.0);
        n               = (1.0/(2.0 * sin(theta))) * [R_input(3,2) - R_input(2,3); R_input(1,3) - R_input(3,1); R_input(2,1) - R_input(1,2)];
        log_R_output    = theta * n;
    end
end