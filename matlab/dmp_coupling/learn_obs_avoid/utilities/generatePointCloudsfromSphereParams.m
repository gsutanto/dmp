function [ point_clouds_cart_position ] = generatePointCloudsfromSphereParams( sphere_params )
    N_xz_points             = 4;
    N_xy_rotations          = 8;

    normalized_y_vector     = [0; 1; 0];
    normalized_z_vector     = [0; 0; 1];
    rot_angle_wrt_y_axis    = [0:pi/(N_xz_points+1):pi];
    rot_angle_wrt_y_axis    = rot_angle_wrt_y_axis(1,2:end-1);
    rot_angle_wrt_z_axis    = [0:2*pi/N_xy_rotations:2*pi];
    rot_angle_wrt_z_axis    = rot_angle_wrt_z_axis(1,1:end-1);
    
    points_to_be_z_rotated  = zeros(3, N_xz_points);
    for iy=1:N_xz_points
        Ry  = vrrotvec2mat([normalized_y_vector.',rot_angle_wrt_y_axis(1,iy)]);
        points_to_be_z_rotated(:,iy)    = Ry * normalized_z_vector;
    end
    
    N_points                    = N_xz_points*N_xy_rotations + 3;
    point_clouds_cart_position  = zeros(3, N_points);
    for iz=1:N_xy_rotations
        Rz  = vrrotvec2mat([normalized_z_vector.',rot_angle_wrt_z_axis(1,iz)]);
        point_clouds_cart_position(:,((iz-1)*N_xz_points)+1:(iz*N_xz_points))   = Rz * points_to_be_z_rotated;
    end
    
    point_clouds_cart_position(:,end)   = zeros(3,1);
    point_clouds_cart_position(:,end-1)	= normalized_z_vector;
    point_clouds_cart_position(:,end-2)	= -normalized_z_vector;
    
    point_clouds_cart_position          = sphere_params.radius * point_clouds_cart_position;
    point_clouds_cart_position          = point_clouds_cart_position + repmat(sphere_params.center, 1, N_points);
    
    % some plotting for troubleshooting:
%     figure;
%     axis equal;
%     hold on;
%         for i=1:N_points
%             plot_sphere(0.1*sphere_params.radius, ...
%                         point_clouds_cart_position(1,i), ...
%                         point_clouds_cart_position(2,i), ...
%                         point_clouds_cart_position(3,i));
%         end
%     hold off;

    point_clouds_cart_position          = point_clouds_cart_position.';
end