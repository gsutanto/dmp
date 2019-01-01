function [ varargout ] = extractSetQuaternionTrajectories( varargin )
    % Author: Giovanni Sutanto
    % Date  : Dec. 31, 2018
    
    input_path   	= varargin{1};
    if (nargin > 1)
        start_column_idx    = varargin{2};
    else
        start_column_idx    = 2;
    end
    if (nargin > 2)
        time_column_idx     = varargin{3};
    else
        time_column_idx     = 1;
    end
    
    data            = cell(0,0);
    if (exist(input_path, 'dir') == 7)  % input_path is a directory
        i               = 1;
        file_path       = [input_path, '/', num2str(i, '%02d'), '.txt'];
        while (exist(file_path, 'file') == 2)
            [ quattraj_data, quattraj_dt ]  = extractQuaternionTrajectory( file_path, start_column_idx, time_column_idx );
            data(:,i)   = quattraj_data(:,1);
            if (i == 1)
                dt      = quattraj_dt;
            else
                diff_dt = dt - quattraj_dt;
                rel_abs_diff_dt     = max(abs(diff_dt/dt), abs(diff_dt/quattraj_dt));
                assert(rel_abs_diff_dt <= 1e-3, ['dt deviation between Quaternion Trajectories=', num2str(rel_abs_diff_dt), ' is beyond threshold!']);
            end
            i           = i+1;
            file_path   = [input_path, num2str(i), '.txt'];
        end
    elseif (exist(input_path, 'file') == 2)  % input_path is a file
        [ data, dt ]    = extractQuaternionTrajectory( input_path, start_column_idx, time_column_idx );
    else
        error('Input path is neither a directory nor a file!');
    end
    
    varargout(1)    = {data};
    varargout(2)    = {dt};
end