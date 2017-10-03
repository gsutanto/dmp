clear all;
close all;
clc;

addpath('../../../../../utilities/clmcplot/');
addpath('../../../../../utilities/quaternion/');
addpath('../../../../../utilities/');
addpath('../../vicon_objects/');

is_plotting_trajs       = 1;

D                       = 3;

thresh_max_abs_z_accel  = 1000;
thresh_max_abs_z_vel    = 3;
vicon_marker_radius     = 15.0/2000.0;  % in meter
hand_marker_radius      = 50/1000.0;    % in meter

in_dir_path     = ['/home/gsutanto/Desktop/CLMC/Data/DMP_LOA_Ct_Vicon_Data_Collection_201607/learn_obs_avoid_gsutanto_vicon_data/baseline/'];

freq            = 300.0;

dt              = 1.0/freq;

files = dir([in_dir_path,'d*']);
RIGHT_WRIST_XYZ_CELL        = cell(3,1);
idx                         = 1;
for file = files'
    [data,vars,freq] = clmcplot_convert([in_dir_path, file.name]);
    is_there_data_jump      = 0;
    % position
    position                = clmcplot_getvariables(data, vars, {'right_wrist_x','right_wrist_y','right_wrist_z'});
    velocity                = zeros(size(position));
    acceleration            = zeros(size(position));
    for d=1:D
        velocity(:,d)       = diffnc(position(:,d),dt);
        acceleration(:,d)   = diffnc(velocity(:,d),dt);
    end
    if ((max(abs(acceleration(:,3))) > thresh_max_abs_z_accel) || ...
        (max(abs(velocity(:,3))) > thresh_max_abs_z_vel))
        is_there_data_jump              = 1;
    end
    if (is_there_data_jump == 0)
        RIGHT_WRIST_XYZ_CELL{1,idx}     = position;
        RIGHT_WRIST_XYZ_CELL{2,idx}     = velocity;
        RIGHT_WRIST_XYZ_CELL{3,idx}     = acceleration;
        if (idx==1)
            START_MARKER_XYZ= clmcplot_getvariables(data, vars, {'start_marker_x','start_marker_y','start_marker_z'});
            GOAL_MARKER_XYZ = clmcplot_getvariables(data, vars, {'goal_marker_x','goal_marker_y','goal_marker_z'});
        end
        idx                             = idx + 1;
    end
end

num_good_trajs  = size(RIGHT_WRIST_XYZ_CELL,2);

if (is_plotting_trajs)
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
        for k=1:num_good_trajs
            plot3(RIGHT_WRIST_XYZ_CELL{1,k}(:,1),...
                  RIGHT_WRIST_XYZ_CELL{1,k}(:,2),...
                  RIGHT_WRIST_XYZ_CELL{1,k}(:,3),'g');
        end
        campos([-3.4744, 6.6812, 4.3045]);
        xlabel('x');
        ylabel('y');
        zlabel('z');
    hold off;

    for order=1:3 % 1=position; 2=velocity; 3=acceleration
        figure;
        axis equal;
        for d=1:D
            subplot(D,1,d)
            if (d==1)
                title(['order=',num2str(order)]);
            end
            hold on;
                for k=1:num_good_trajs
                    plot(RIGHT_WRIST_XYZ_CELL{order,k}(:,d),'g');
                end
            hold off;
        end
    end
end