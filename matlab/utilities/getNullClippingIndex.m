function [ start_clipping_idx, end_clipping_idx ] = getNullClippingIndex( traj_matrix )
% Author: Giovanni Sutanto
% Date  : July 5, 2017
% Input :
%   traj_matrix = trajectory matrix, each row represent a state:
%                                    row 0 is 0-th state in the trajectory
%                                    t-th row is t-th state in the trajectory
    
    % searching for null-data (zeros) start clipping (row) index of traj_matrix:
    start_clipping_idx      = 1;
    while (all(traj_matrix(start_clipping_idx, :)==0))
        start_clipping_idx  = start_clipping_idx + 1;
        if (start_clipping_idx > size(traj_matrix, 1))
            keyboard;
        end
    end
    
    % searching for null-data (zeros) end clipping (row) index of traj_matrix:
    end_clipping_idx        = size(traj_matrix, 1);
    if (end_clipping_idx < 1)
        keyboard;
    end
    while (all(traj_matrix(end_clipping_idx, :)==0))
        end_clipping_idx    = end_clipping_idx - 1;
        if ((end_clipping_idx < 1) || (end_clipping_idx < start_clipping_idx))
            keyboard;
        end
    end
end