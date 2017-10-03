clear all;
close all;
clc;

addpath('../../../../../utilities/clmcplot/');
addpath('../../../../../utilities/quaternion/');
addpath('../../../../../utilities/');
addpath('../../vicon_objects/');

vicon_marker_radius     = 15.0/2000.0;  % in meter
hand_marker_radius      = 50/1000.0;    % in meter

[D,vars,freq]   = clmcplot_convert('d00120');
POKE_XYZ        = clmcplot_getvariables(D, vars, {'poke_x','poke_y','poke_z'});
POKE_QWXYZ      = clmcplot_getvariables(D, vars, {'poke_qw','poke_qx','poke_qy','poke_qz'});
RIGHT_WRIST_XYZ = clmcplot_getvariables(D, vars, {'right_wrist_x','right_wrist_y','right_wrist_z'});
START_MARKER_XYZ= clmcplot_getvariables(D, vars, {'start_marker_x','start_marker_y','start_marker_z'});
GOAL_MARKER_XYZ = clmcplot_getvariables(D, vars, {'goal_marker_x','goal_marker_y','goal_marker_z'});
CYL_OBJ_XYZ     = clmcplot_getvariables(D, vars, {'cyl_object_x','cyl_object_y','cyl_object_z'});
CYL_OBJ_QWXYZ   = clmcplot_getvariables(D, vars, {'cyl_object_qw','cyl_object_qx','cyl_object_qy','cyl_object_qz'});
CYL_OBJ_POSE    = zeros(4,4,size(CYL_OBJ_XYZ,1));

CYL_OBJ_MARKERS_LOCAL_XYZ              = dlmread('cyl_object.txt');
CYL_OBJ_MARKERS_LOCAL_XYZ_HOMOGENEOUS  = [(CYL_OBJ_MARKERS_LOCAL_XYZ/1000.0).';ones(1,size(CYL_OBJ_MARKERS_LOCAL_XYZ,1))];
CYL_OBJ_MARKERS_GLOBAL_XYZ_TRAJ        = zeros(size(CYL_OBJ_MARKERS_LOCAL_XYZ_HOMOGENEOUS,1),size(CYL_OBJ_MARKERS_LOCAL_XYZ_HOMOGENEOUS,2),size(CYL_OBJ_XYZ,1));

for t=1:size(CYL_OBJ_XYZ,1)
    q                           = quaternion(CYL_OBJ_QWXYZ(t,:));
    qn                          = q.normalize;
    R                           = qn.RotationMatrix;
    CYL_OBJ_POSE(1:3,1:3,t)  = R;
    CYL_OBJ_POSE(1:3,4,t)    = CYL_OBJ_XYZ(t,:).';
    CYL_OBJ_POSE(4,4,t)      = 1.0;
    CYL_OBJ_MARKERS_GLOBAL_XYZ_TRAJ(:,:,t) = CYL_OBJ_POSE(:,:,t) * CYL_OBJ_MARKERS_LOCAL_XYZ_HOMOGENEOUS;
end

CYL_OBJ_MARKER_GLOBAL_XYZ_TRAJ_CELL    = cell(1,size(CYL_OBJ_MARKERS_GLOBAL_XYZ_TRAJ,2));
for j=1:size(CYL_OBJ_MARKER_GLOBAL_XYZ_TRAJ_CELL,2)
    CYL_OBJ_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}   = reshape(CYL_OBJ_MARKERS_GLOBAL_XYZ_TRAJ(1:3,j,:),3,size(CYL_OBJ_XYZ,1));
end

figure;
axis equal;
hold on;
    clc;
    plot_sphere(hand_marker_radius, ...
                START_MARKER_XYZ(1,1), ...
                START_MARKER_XYZ(1,2), ...
                START_MARKER_XYZ(1,3));
    plot_sphere(hand_marker_radius, ...
                GOAL_MARKER_XYZ(1,1), ...
                GOAL_MARKER_XYZ(1,2), ...
                GOAL_MARKER_XYZ(1,3));
%     plot_sphere(hand_marker_radius, ...
%                 CYL_OBJ_XYZ(end,1), ...
%                 CYL_OBJ_XYZ(end,2), ...
%                 CYL_OBJ_XYZ(end,3));
    for j=1:size(CYL_OBJ_MARKER_GLOBAL_XYZ_TRAJ_CELL,2)
        plot_sphere(vicon_marker_radius,...
                    CYL_OBJ_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}(1,end),...
                    CYL_OBJ_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}(2,end),...
                    CYL_OBJ_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}(3,end))
    end
    plot3(RIGHT_WRIST_XYZ(:,1),...
          RIGHT_WRIST_XYZ(:,2),...
          RIGHT_WRIST_XYZ(:,3),'g');
hold off;