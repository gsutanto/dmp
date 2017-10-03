function [ smoothed_ND_traj ] = smoothStartEndNDTrajectoryBasedOnPosition( ND_traj, ...
                                                                           percentage_padding, ...
                                                                           percentage_smoothing_points, ...
                                                                           mode, dt, ...
                                                                           varargin )
    if (nargin > 4)
        fc  = varargin{1};
    else
        fc  = 40;           % 40 Hz cutoff frequency
    end
    
    fs      = 1/dt;                 % sampling frequency
    [b, a]  = butter(2, fc/(fs/2)); % 2nd order Butterworth filter
    
    ND_position_prof        = ND_traj{1,1};
    D                       = size(ND_position_prof, 2);
    smoothed_position_prof  = smoothStartEndNDPositionProfile( ND_position_prof, ...
                                                               percentage_padding, ...
                                                               percentage_smoothing_points, ...
                                                               mode, b, a );
    
    smoothed_velocity_prof      = zeros(size(smoothed_position_prof));
    smoothed_acceleration_prof  = zeros(size(smoothed_position_prof));
    
    for d=1:D
        smoothed_velocity_prof(:,d)     = diffnc(smoothed_position_prof(:,d), dt);
        smoothed_acceleration_prof(:,d) = diffnc(smoothed_velocity_prof(:,d), dt);
    end
    
    smoothed_ND_traj        = cell(3,1);
    smoothed_ND_traj{1,1}   = smoothed_position_prof;
    smoothed_ND_traj{2,1}   = smoothed_velocity_prof;
    smoothed_ND_traj{3,1}   = smoothed_acceleration_prof;
end

