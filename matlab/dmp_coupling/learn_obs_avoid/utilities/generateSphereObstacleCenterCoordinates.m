function [ sphere_obs_center_coords ] = generateSphereObstacleCenterCoordinates( point1, point2 )
    % Author: Giovanni Sutanto
    % Date  : August 22, 2016
    % Description:
    % Generate (unseen) sphere obstacles' center coordinates around 2 points, 
    % point1 and point2 for evaluating performance metric II and III
    
    N_sph_obs_center_x_variations   = 7;
    N_sph_obs_center_y_variations   = 3;
    N_sph_obs_center_z_variations   = 3;
    diff_sph_obs_center_y_variations= 0.025; % in meter
    diff_sph_obs_center_z_variations= 0.025; % in meter
    
    sph_obs_center_x_low            = min(point1(1,1), point2(1,1));
    sph_obs_center_x_high           = max(point1(1,1), point2(1,1));
    sph_obs_center_x_variations     = linspace(sph_obs_center_x_low, sph_obs_center_x_high, N_sph_obs_center_x_variations);
    diff_sph_obs_center_x_variations    = sph_obs_center_x_variations(1,2) - sph_obs_center_x_variations(1,1);
    sph_obs_center_x_variations(1,1)    = sph_obs_center_x_variations(1,1) - diff_sph_obs_center_x_variations;
    sph_obs_center_x_variations(1,end)  = sph_obs_center_x_variations(1,end) + diff_sph_obs_center_x_variations;
    ctr_point                           = (point1 + point2)/2.0;
    sph_obs_center_y_low            = ctr_point(2,1) - (floor(N_sph_obs_center_y_variations/2) * diff_sph_obs_center_y_variations);
    sph_obs_center_y_high           = ctr_point(2,1) + (floor(N_sph_obs_center_y_variations/2) * diff_sph_obs_center_y_variations);
    sph_obs_center_z_low            = ctr_point(3,1) - (floor(N_sph_obs_center_z_variations/2) * diff_sph_obs_center_z_variations);
    sph_obs_center_z_high           = ctr_point(3,1) + (floor(N_sph_obs_center_z_variations/2) * diff_sph_obs_center_z_variations);
    sph_obs_center_y_variations     = linspace(sph_obs_center_y_low, sph_obs_center_y_high, N_sph_obs_center_y_variations);
    sph_obs_center_z_variations     = linspace(sph_obs_center_z_low, sph_obs_center_z_high, N_sph_obs_center_z_variations);
    [sph_obs_center_x_coords, sph_obs_center_y_coords, sph_obs_center_z_coords] = meshgrid(sph_obs_center_x_variations, ...
                                                                                           sph_obs_center_y_variations, ...
                                                                                           sph_obs_center_z_variations);
    sph_obs_center_x_coords     = reshape(sph_obs_center_x_coords, 1, size(sph_obs_center_x_coords,1)*size(sph_obs_center_x_coords,2)*size(sph_obs_center_x_coords,3));
    sph_obs_center_y_coords     = reshape(sph_obs_center_y_coords, 1, size(sph_obs_center_y_coords,1)*size(sph_obs_center_y_coords,2)*size(sph_obs_center_y_coords,3));
    sph_obs_center_z_coords     = reshape(sph_obs_center_z_coords, 1, size(sph_obs_center_z_coords,1)*size(sph_obs_center_z_coords,2)*size(sph_obs_center_z_coords,3));
    sphere_obs_center_coords    = [sph_obs_center_x_coords; ...
                                   sph_obs_center_y_coords; ...
                                   sph_obs_center_z_coords];
    
    % some plotting for troubleshooting:
%     figure;
%     axis equal;
%     hold on;
%         plot_sphere(0.05, ...
%                     point1(1,1), ...
%                     point1(2,1), ...
%                     point1(3,1));
%         plot_sphere(0.05, ...
%                     point2(1,1), ...
%                     point2(2,1), ...
%                     point2(3,1));
%         for j=1:size(sphere_obs_center_coords,2)
%             plot_sphere(0.01, ...
%                         sphere_obs_center_coords(1,j), ...
%                         sphere_obs_center_coords(2,j), ...
%                         sphere_obs_center_coords(3,j));
%         end
%     hold off;
end