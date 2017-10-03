% Author: Giovanni Sutanto
% Date  : June 23, 2015
clear   all;
close   all;
clc;

addpath('../../../../utilities/clmcplot/');

[D,vars,freq]   = clmcplot_convert('d02209');
[traj_x, traj_y, traj_z] = clmcplot_getvariables(D, vars, {'R_HAND_x','R_HAND_y','R_HAND_z'});
traj            = [traj_x'; traj_y'; traj_z'];

start           = [traj_x(1,:); traj_y(1,:); traj_z(1,:)];
goal            = [traj_x(end,:); traj_y(end,:); traj_z(end,:)];
t               = start;  % translation vector
new_x           = goal - start;
new_x           = new_x/(norm(new_x));  % normalize new x axis
if (abs([0;0;1]' * new_x) < 0.97)  % check if new_x is "too parallel" with the z axis; if not:
    new_z       = [0;0;1];  % anchor axis is z axis
    new_y       = cross(new_z, new_x);
    new_y       = new_y/(norm(new_y));  % normalize new y axis
    new_z       = cross(new_x, new_y);  % normalized new z axis
else    % if it is "parallel enough":
    new_y       = [0;1;0];  % anchor axis is y axis
    new_z       = cross(new_x, new_y);
    new_z       = new_z/(norm(new_z));  % normalize new z axis
    new_y       = cross(new_z, new_x);  % normalized new y axis
end
rotM            = [new_x, new_y, new_z];
T               = [[rotM,t]; [0,0,0,1]];
T_inv           = [[rotM',-(rotM')*t]; [0,0,0,1]];
traj_H          = [traj; ones(1,size(traj,2))];
rot_traj_H      = T_inv * traj_H;

[traj_xd, traj_yd, traj_zd] = clmcplot_getvariables(D, vars, {'R_HAND_xd','R_HAND_yd','R_HAND_zd'});
trajd           = [traj_xd'; traj_yd'; traj_zd'];
rot_trajd       = rotM' * trajd;

[traj_xdd, traj_ydd, traj_zdd] = clmcplot_getvariables(D, vars, {'R_HAND_xdd','R_HAND_ydd','R_HAND_zdd'});
trajdd          = [traj_xdd'; traj_ydd'; traj_zdd'];
rot_trajdd      = rotM' * trajdd;

figure;
hold            on;
px1             = quiver3(0,0,0,1,0,0,'r');
py1             = quiver3(0,0,0,0,1,0,'g');
pz1             = quiver3(0,0,0,0,0,1,'b');
ptraj1          = plot3(rot_traj_H(1,:)', rot_traj_H(2,:)', rot_traj_H(3,:)', 'c', 'LineWidth', 3);
legend([px1, py1, pz1, ptraj1], 'Local x-axis', 'Local y-axis', 'Local z-axis', 'Trajectory Representation in Local Coordinate System');
hold            off;

rep_traj_H      = T * rot_traj_H;
figure;
hold            on;
px2             = quiver3(t(1,1),t(2,1),t(3,1),new_x(1,1),new_x(2,1),new_x(3,1),'r');
py2             = quiver3(t(1,1),t(2,1),t(3,1),new_y(1,1),new_y(2,1),new_y(3,1),'g');
pz2             = quiver3(t(1,1),t(2,1),t(3,1),new_z(1,1),new_z(2,1),new_z(3,1),'b');
ptraj2          = plot3(traj_x, traj_y, traj_z, 'cx', rep_traj_H(1,:)', rep_traj_H(2,:)', rep_traj_H(3,:)', 'm+');
legend([px2, py2, pz2], 'Local x-axis', 'Local y-axis', 'Local z-axis');
hold            off;

figure;
axis            equal;
hold            on;
px1             = quiver3(0,0,0,0.35,0,0,'r-.', 'LineWidth', 2);
py1             = quiver3(0,0,0,0,0.35,0,'g-.', 'LineWidth', 2);
pz1             = quiver3(0,0,0,0,0,0.35,'b-.', 'LineWidth', 2);
px2             = quiver3(t(1,1),t(2,1),t(3,1),0.75*new_x(1,1),0.75*new_x(2,1),0.75*new_x(3,1),'r', 'LineWidth', 2);
py2             = quiver3(t(1,1),t(2,1),t(3,1),0.75*new_y(1,1),0.75*new_y(2,1),0.75*new_y(3,1),'g', 'LineWidth', 2);
pz2             = quiver3(t(1,1),t(2,1),t(3,1),0.75*new_z(1,1),0.75*new_z(2,1),0.75*new_z(3,1),'b', 'LineWidth', 2);
ptraj2          = plot3(traj_x, traj_y, traj_z, 'k', 'LineWidth', 3);
legend([px1, py1, pz1, px2, py2, pz2, ptraj2], 'Global x-axis', 'Global y-axis', 'Global z-axis', 'Local x-axis', 'Local y-axis', 'Local z-axis',...
                                               'Trajectory');
title('Global and Local Coordinate System in DMP');
hold            off;