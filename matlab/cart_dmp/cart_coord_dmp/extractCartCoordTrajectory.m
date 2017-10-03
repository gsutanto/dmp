function [ varargout ] = extractCartCoordTrajectory( varargin )
    % Author: Giovanni Sutanto
    % Date  : June 08, 2017
    
    file_path       = varargin{1};
    if (nargin > 1)
        version     = varargin{2};
    else
        version     = 0;
    end
    
    assert(exist(file_path, 'file') == 2, 'File does NOT exist!');
    
    data            = cell(0,0);
    trajectory      = dlmread(file_path);
    time            = trajectory(:,1);
    if (version == 0)
        data{1,1}   = trajectory(:,2:4);    % only x-y-z (position) are recorded
    elseif (version == 1)
        data{1,1}   = trajectory(:,2:10);   % time-x-y-z-xd-yd-zd-xdd-ydd-zdd (entire trajectory information) are recorded
    elseif (version == 2)
        data{1,1}   = trajectory(:,2:4);
        data{2,1}   = trajectory(:,5:7);
        data{3,1}   = trajectory(:,8:10);
    end
    tau             = time(end,1) - time(1,1);
    traj_length     = size(time,1);
    dt              = tau/(traj_length-1);
    
    varargout(1)    = {data};
    varargout(2)    = {dt};
    varargout(3)    = {time};
end

