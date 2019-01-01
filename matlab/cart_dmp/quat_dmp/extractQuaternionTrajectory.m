function [ varargout ] = extractQuaternionTrajectory( varargin )
    % Author: Giovanni Sutanto
    % Date  : Dec. 31, 2018
    
    file_path               = varargin{1};
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
    
    assert(exist(file_path, 'file') == 2, 'File does NOT exist!');
    
    data            = cell(0,0);
    trajectory      = dlmread(file_path);
    time            = trajectory(:,time_column_idx)';
    data{1,1}       = trajectory(:,start_column_idx:(start_column_idx+3))';
    data{2,1}       = trajectory(:,(start_column_idx+4):(start_column_idx+6))';
    data{3,1}       = trajectory(:,(start_column_idx+7):(start_column_idx+9))';
    tau             = time(1,end) - time(1,1);
    traj_length     = size(time,2);
    dt              = tau/(traj_length-1);
    
    varargout(1)    = {data};
    varargout(2)    = {dt};
    varargout(3)    = {time};
end

