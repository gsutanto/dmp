clear all;
close all;
clc;

addpath('../../../../../utilities/clmcplot/');
addpath('../../../../../utilities/quaternion/');
addpath('../../../../../utilities/');
addpath('../../vicon_objects/');

vicon_marker_radius     = 15.0/2000.0;  % in meter
hand_marker_radius      = 50/1000.0;    % in meter

[D,vars,freq]   = clmcplot_convert('d00042');
POKE_XYZ        = clmcplot_getvariables(D, vars, {'poke_x','poke_y','poke_z'});
POKE_QWXYZ      = clmcplot_getvariables(D, vars, {'poke_qw','poke_qx','poke_qy','poke_qz'});
RIGHT_WRIST_XYZ = clmcplot_getvariables(D, vars, {'right_wrist_x','right_wrist_y','right_wrist_z'});
START_MARKER_XYZ= clmcplot_getvariables(D, vars, {'start_marker_x','start_marker_y','start_marker_z'});
GOAL_MARKER_XYZ = clmcplot_getvariables(D, vars, {'goal_marker_x','goal_marker_y','goal_marker_z'});

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
    plot3(POKE_XYZ(:,1),...
          POKE_XYZ(:,2),...
          POKE_XYZ(:,3),'b');
    plot3(RIGHT_WRIST_XYZ(:,1),...
          RIGHT_WRIST_XYZ(:,2),...
          RIGHT_WRIST_XYZ(:,3),'g');
hold off;