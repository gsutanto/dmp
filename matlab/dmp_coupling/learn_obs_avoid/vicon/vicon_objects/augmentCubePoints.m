clear all;
close all;
clc;

cube_height                 = 100.0; % in milimeter
N_augment                   = 4;

points_on_cube_top_surface  = dlmread('cube_object_unaugmented.txt');
[ ave_normal_vector ]       = computeAverageNormalVector( points_on_cube_top_surface );

points_cube_augmented       = [points_on_cube_top_surface];

for i=1:N_augment
    points_cube_augment     = points_on_cube_top_surface - ((i/N_augment) * cube_height * repmat(ave_normal_vector,size(points_on_cube_top_surface,1),1));
    points_cube_augmented   = [points_cube_augmented; points_cube_augment];
end

dlmwrite('cube_object.txt', points_cube_augmented, 'delimiter', ' ');