clear all;
close all;
clc;

cyl_height                  = 185.0; % in milimeter
N_augment                   = 6;

points_on_cyl_top_surface  = dlmread('cyl_object_unaugmented.txt');
[ ave_normal_vector ]       = computeAverageNormalVector( points_on_cyl_top_surface );

points_cyl_augmented       = [points_on_cyl_top_surface];

for i=1:N_augment
    points_cyl_augment     = points_on_cyl_top_surface + ((i/N_augment) * cyl_height * repmat(ave_normal_vector,size(points_on_cyl_top_surface,1),1));
    points_cyl_augmented   = [points_cyl_augmented; points_cyl_augment];
end

dlmwrite('cyl_object.txt', points_cyl_augmented, 'delimiter', ' ');