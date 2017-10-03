% Author: Giovanni Sutanto
% Date  : June 24, 2015

close all;
clear all;
clc;

addpath('../../../matlab/utilities/clmcplot/');

[D,vars,freq]   = clmcplot_convert('d02209');
time                        = clmcplot_getvariables(D, vars, {'time'}); % [0.0:(1/420.0):4.0-(1/420.0)];
[traj_x, traj_xd, traj_xdd] = clmcplot_getvariables(D, vars, {'R_HAND_x','R_HAND_xd','R_HAND_xdd'});
[traj_y, traj_yd, traj_ydd] = clmcplot_getvariables(D, vars, {'R_HAND_y','R_HAND_yd','R_HAND_ydd'});
[traj_z, traj_zd, traj_zdd] = clmcplot_getvariables(D, vars, {'R_HAND_z','R_HAND_zd','R_HAND_zdd'});
[traj_q0, traj_q0d, traj_q0dd]  = clmcplot_getvariables(D, vars, {'R_HAND_q0','R_HAND_q0d','R_HAND_q0dd'});
[traj_q1, traj_q1d, traj_q1dd]  = clmcplot_getvariables(D, vars, {'R_HAND_q1','R_HAND_q1d','R_HAND_q1dd'});
[traj_q2, traj_q2d, traj_q2dd]  = clmcplot_getvariables(D, vars, {'R_HAND_q2','R_HAND_q2d','R_HAND_q2dd'});
[traj_q3, traj_q3d, traj_q3dd]  = clmcplot_getvariables(D, vars, {'R_HAND_q3','R_HAND_q3d','R_HAND_q3dd'});
[traj_ad,  traj_bd,  traj_gd]   = clmcplot_getvariables(D, vars, {'R_HAND_ad','R_HAND_bd','R_HAND_gd'});
[traj_add, traj_bdd, traj_gdd]  = clmcplot_getvariables(D, vars, {'R_HAND_add','R_HAND_bdd','R_HAND_gdd'});
cart_coord_traj         = [time, traj_x, traj_y, traj_z, traj_xd, traj_yd, traj_zd, traj_xdd, traj_ydd, traj_zdd];
cart_quat_traj          = [time, traj_q0, traj_q1, traj_q2, traj_q3, traj_q0d, traj_q1d, traj_q2d, traj_q3d, traj_q0dd, traj_q1dd, traj_q2dd, traj_q3dd];
cart_quat_ABGomega_traj = [time, traj_q0, traj_q1, traj_q2, traj_q3, traj_ad,  traj_bd,  traj_gd, traj_add, traj_bdd, traj_gdd];

fileID = fopen('sample_traj_3D_recorded_demo_wiggling_orig.txt','w');
for i=1:size(cart_coord_traj,1)
    nbytes = fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f\n',cart_coord_traj(i,:));
end
fclose(fileID);

fileID = fopen('sample_traj_3D_recorded_demo_wiggling.txt','w');
for i=1:840
    nbytes = fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f\n',cart_coord_traj(i,:));
end
fclose(fileID);

fileID = fopen('sample_quat_traj_recorded_demo_wiggling.txt','w');
for i=1:840
%     norm_q = norm(cart_quat_traj(i,2:5));
%     disp(['norm_q = ', num2str(norm_q)]);
    nbytes = fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f %f\n',cart_quat_traj(i,:));
end
fclose(fileID);

fileID = fopen('sample_quat_ABGomega_traj_recorded_demo_wiggling.txt','w');
for i=1:840
    nbytes = fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f %f\n',cart_quat_ABGomega_traj(i,:));
end
fclose(fileID);