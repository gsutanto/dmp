clear all;
close all;
clc;

addpath('../../../../../utilities/clmcplot/');
addpath('../../../../../utilities/quaternion/');
addpath('../../../../../utilities/');
addpath('../../vicon_objects/');

vicon_marker_radius     = 15.0/2000.0;  % in meter
hand_marker_radius      = 50/1000.0;    % in meter
window_size             = 600;

obs_type        = 'cube';
obs_name        = [obs_type,'_object'];

[D,vars,freq]   = clmcplot_convert('d04258');
RIGHT_HAND_XYZ  = clmcplot_getvariables(D, vars, {'R_HAND_x','R_HAND_y','R_HAND_z'});
OBS_XYZ         = clmcplot_getvariables(D, vars, {[obs_name,'_x'],[obs_name,'_y'],[obs_name,'_z']});
OBS_QWXYZ       = clmcplot_getvariables(D, vars, {[obs_name,'_qw'],[obs_name,'_qx'],[obs_name,'_qy'],[obs_name,'_qz']});
OBS_POSE        = zeros(4,4,size(OBS_XYZ,1));

OBS_MARKERS_LOCAL_XYZ              = dlmread([obs_name,'.txt']);
OBS_MARKERS_LOCAL_XYZ_HOMOGENEOUS  = [(OBS_MARKERS_LOCAL_XYZ/1000.0).';ones(1,size(OBS_MARKERS_LOCAL_XYZ,1))];
OBS_MARKERS_GLOBAL_XYZ_TRAJ        = zeros(size(OBS_MARKERS_LOCAL_XYZ_HOMOGENEOUS,1),size(OBS_MARKERS_LOCAL_XYZ_HOMOGENEOUS,2),size(OBS_XYZ,1));

for t=1:size(OBS_XYZ,1)
    q                       = quaternion(OBS_QWXYZ(t,:));
    qn                      = q.normalize;
    R                       = qn.RotationMatrix;
    OBS_POSE(1:3,1:3,t)    = R;
    OBS_POSE(1:3,4,t)      = OBS_XYZ(t,:).';
    OBS_POSE(4,4,t)        = 1.0;
    OBS_MARKERS_GLOBAL_XYZ_TRAJ(:,:,t) = OBS_POSE(:,:,t) * OBS_MARKERS_LOCAL_XYZ_HOMOGENEOUS;
end

OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL    = cell(1,size(OBS_MARKERS_GLOBAL_XYZ_TRAJ,2));
for j=1:size(OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL,2)
    OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}   = reshape(OBS_MARKERS_GLOBAL_XYZ_TRAJ(1:3,j,:),3,size(OBS_XYZ,1));
end

figure;
axis equal;
for t=1:10:size(OBS_XYZ,1)
    hold on;
        clc;
        t
%         plot_sphere(hand_marker_radius, ...
%                     RIGHT_HAND_XYZ(t,1), ...
%                     RIGHT_HAND_XYZ(t,2), ...
%                     RIGHT_HAND_XYZ(t,3));
        for j=1:size(OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL,2)
            plot3(OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}(1,max(1,(t-window_size)):t),...
                  OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}(2,max(1,(t-window_size)):t),...
                  OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}(3,max(1,(t-window_size)):t));
            plot3(RIGHT_HAND_XYZ(max(1,(t-window_size)):t,1),...
                  RIGHT_HAND_XYZ(max(1,(t-window_size)):t,2),...
                  RIGHT_HAND_XYZ(max(1,(t-window_size)):t,3));
%             plot_sphere(vicon_marker_radius,...
%                         OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}(1,t),...
%                         OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}(2,t),...
%                         OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}(3,t))
        end
        drawnow;
    hold off;
end