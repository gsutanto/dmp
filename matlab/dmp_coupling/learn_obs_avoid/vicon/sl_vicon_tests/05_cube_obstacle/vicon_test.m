clear all;
close all;
clc;

addpath('../../../../../utilities/clmcplot/');
addpath('../../../../../utilities/quaternion/');
addpath('../../../../../utilities/');
addpath('../../vicon_objects/');

vicon_marker_radius     = 15.0/2000.0;  % in meter
hand_marker_radius      = 50/1000.0;    % in meter

[D,vars,freq]   = clmcplot_convert('d00102');
POKE_XYZ        = clmcplot_getvariables(D, vars, {'poke_x','poke_y','poke_z'});
POKE_QWXYZ      = clmcplot_getvariables(D, vars, {'poke_qw','poke_qx','poke_qy','poke_qz'});
RIGHT_WRIST_XYZ = clmcplot_getvariables(D, vars, {'right_wrist_x','right_wrist_y','right_wrist_z'});
START_MARKER_XYZ= clmcplot_getvariables(D, vars, {'start_marker_x','start_marker_y','start_marker_z'});
GOAL_MARKER_XYZ = clmcplot_getvariables(D, vars, {'goal_marker_x','goal_marker_y','goal_marker_z'});
CUBE_OBJ_XYZ    = clmcplot_getvariables(D, vars, {'cube_object_x','cube_object_y','cube_object_z'});
CUBE_OBJ_QWXYZ  = clmcplot_getvariables(D, vars, {'cube_object_qw','cube_object_qx','cube_object_qy','cube_object_qz'});
CUBE_OBJ_POSE   = zeros(4,4,size(CUBE_OBJ_XYZ,1));

CUBE_OBJ_MARKERS_LOCAL_XYZ              = dlmread('cube_object.txt');
CUBE_OBJ_MARKERS_LOCAL_XYZ_HOMOGENEOUS  = [(CUBE_OBJ_MARKERS_LOCAL_XYZ/1000.0).';ones(1,size(CUBE_OBJ_MARKERS_LOCAL_XYZ,1))];
CUBE_OBJ_MARKERS_GLOBAL_XYZ_TRAJ        = zeros(size(CUBE_OBJ_MARKERS_LOCAL_XYZ_HOMOGENEOUS,1),size(CUBE_OBJ_MARKERS_LOCAL_XYZ_HOMOGENEOUS,2),size(CUBE_OBJ_XYZ,1));

for t=1:size(CUBE_OBJ_XYZ,1)
    q                           = quaternion(CUBE_OBJ_QWXYZ(t,:));
    qn                          = q.normalize;
    R                           = qn.RotationMatrix;
    CUBE_OBJ_POSE(1:3,1:3,t)  = R;
    CUBE_OBJ_POSE(1:3,4,t)    = CUBE_OBJ_XYZ(t,:).';
    CUBE_OBJ_POSE(4,4,t)      = 1.0;
    CUBE_OBJ_MARKERS_GLOBAL_XYZ_TRAJ(:,:,t) = CUBE_OBJ_POSE(:,:,t) * CUBE_OBJ_MARKERS_LOCAL_XYZ_HOMOGENEOUS;
end

CUBE_OBJ_MARKER_GLOBAL_XYZ_TRAJ_CELL    = cell(1,size(CUBE_OBJ_MARKERS_GLOBAL_XYZ_TRAJ,2));
for j=1:size(CUBE_OBJ_MARKER_GLOBAL_XYZ_TRAJ_CELL,2)
    CUBE_OBJ_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}   = reshape(CUBE_OBJ_MARKERS_GLOBAL_XYZ_TRAJ(1:3,j,:),3,size(CUBE_OBJ_XYZ,1));
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
%                 CUBE_OBJ_XYZ(end,1), ...
%                 CUBE_OBJ_XYZ(end,2), ...
%                 CUBE_OBJ_XYZ(end,3));
    for j=1:size(CUBE_OBJ_MARKER_GLOBAL_XYZ_TRAJ_CELL,2)
        plot_sphere(vicon_marker_radius,...
                    CUBE_OBJ_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}(1,end),...
                    CUBE_OBJ_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}(2,end),...
                    CUBE_OBJ_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}(3,end))
    end
    plot3(RIGHT_WRIST_XYZ(:,1),...
          RIGHT_WRIST_XYZ(:,2),...
          RIGHT_WRIST_XYZ(:,3),'g');
hold off;