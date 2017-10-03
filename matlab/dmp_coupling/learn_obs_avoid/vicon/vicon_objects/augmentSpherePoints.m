clear all;
close all;
clc;

points_on_sphere_surface    = dlmread('sphere_object_unaugmented.txt');
[ xc, yc, zc ] = computeSphereCenterUsingLeastSquares( points_on_sphere_surface );

rot_axis_CELL       = {[1;0;0], [-1;0;0], [0;1;0], [0;-1;0], [1;0;0]};
rot_angle_CELL      = {pi/2, pi/2, pi/2, pi/2, pi};
points_sphere_augmented     = [xc, yc, zc; points_on_sphere_surface];

for i=1:length(rot_axis_CELL)
    rot_axis        = rot_axis_CELL{i};
    normed_rot_axis = rot_axis/norm(rot_axis);
    R               = vrrotvec2mat([normed_rot_axis.', rot_angle_CELL{i}]);
    T_inv_tr        =  [eye(3), -[ xc; yc; zc ]; 0, 0, 0, 1];
    T_rot           = [R, zeros(3,1); 0, 0, 0, 1];
    T_tr            = [eye(3), [ xc; yc; zc ]; 0, 0, 0, 1];
    points_on_sphere_surface_H      = [points_on_sphere_surface.'; ones(1, size(points_on_sphere_surface,1))];
    new_points_on_sphere_surface_H  = T_tr * T_rot * T_inv_tr * points_on_sphere_surface_H;
    points_sphere_augmented         = [ points_sphere_augmented; new_points_on_sphere_surface_H(1:3,:).' ];
end

dlmwrite('sphere_object.txt', points_sphere_augmented, 'delimiter', ' ');