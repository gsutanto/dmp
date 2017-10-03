%**************************************************************************
% create_3D_min_jerk_sample_traj.m                                        *
% Written by Giovanni Sutanto (gsutanto@usc.edu)                          *
% USC ID 8587-8103-42                                                     *
% 19th July 2015                                                          *
%**************************************************************************
clear all;
clc;

% Specify 5th-Order Spline Constants:
c0              = 0;
c1              = 0;
c2              = 0;
c3              = 2.5;
c4              = -1.875;
c5              = 0.375;

% Initialize Trajectory Storage for Position (x), Velocity (xd), and
% Acceleration (xdd):
trajectory_x    = [];
trajectory_xd   = [];
trajectory_xdd  = [];

% Compute Position (x), Velocity (xd), and Acceleration (xdd) 
% for each Time instants 0 <= t <= 2, with time step of 0.01 seconds, 
% and store the result in Trajectory Storage:
fileId = fopen('sample_traj_3D_min_jerk.txt','w');
for t = 0:0.001:2
    x               = (c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5)/40.0;
    trajectory_x    = [trajectory_x x];

    xd              = (c1 + 2*c2*t + 3*c3*t^2 + 4*c4*t^3 + 5*c5*t^4)/40.0;
    trajectory_xd   = [trajectory_xd xd];

    xdd             = (2*c2 + 6*c3*t + 12*c4*t^2 + 20*c5*t^3)/40.0;
    trajectory_xdd  = [trajectory_xdd xdd];
    fprintf(fileId, '%f %f %f %f %f %f %f %f %f %f\n', t, x, x, x,...
            xd, xd, xd, xdd, xdd, xdd);
end
fclose(fileId);
