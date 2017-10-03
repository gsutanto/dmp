function [ varargout ] = extractSetCartCoordTrajectories( varargin )
    % Author: Giovanni Sutanto
    % Date  : January 27, 2016
    
    input_path   	= varargin{1};
    if (nargin > 1)
        version     = varargin{2};
    else
        version     = 0;
    end
    
    data            = cell(0,0);
    if (exist(input_path, 'dir') == 7)  % input_path is a directory
        i               = 1;
        file_path       = [input_path, '/', num2str(i), '.txt'];
        while (exist(file_path, 'file') == 2)
            [ cctraj_data, cctraj_dt ]  = extractCartCoordTrajectory( file_path, version );
            data(:,i)   = cctraj_data(:,1);
            if (i == 1)
                dt      = cctraj_dt;
            else
                diff_dt = dt - cctraj_dt;
                rel_abs_diff_dt     = max(abs(diff_dt/dt), abs(diff_dt/cctraj_dt));
                assert(rel_abs_diff_dt <= 1e-3, ['dt deviation between Cartesian Coordinate Trajectories=', num2str(rel_abs_diff_dt), ' is beyond threshold!']);
            end
            i           = i+1;
            file_path   = [input_path, num2str(i), '.txt'];
        end
    elseif (exist(input_path, 'file') == 2)  % input_path is a file
        [ data, dt ]    = extractCartCoordTrajectory( input_path, version );
    else
        error('Input path is neither a directory nor a file!');
    end
    
    varargout(1)    = {data};
    varargout(2)    = {dt};
end