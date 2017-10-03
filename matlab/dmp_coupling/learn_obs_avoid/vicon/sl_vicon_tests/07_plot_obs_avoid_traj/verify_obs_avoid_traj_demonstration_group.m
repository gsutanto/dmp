function [ num_good_trajs ] = verify_obs_avoid_traj_demonstration_group( varargin )
    obs_type        = varargin{1};
    setting_number  = varargin{2};
    if (nargin > 2)
        is_plotting_trajs   = varargin{3};
    else
        is_plotting_trajs   = 1;
    end

    addpath('../../../../../utilities/clmcplot/');
    addpath('../../../../../utilities/quaternion/');
    addpath('../../../../../utilities/');
    addpath('../../vicon_objects/');

    D               = 3;

    thresh_max_abs_z_accel  = 1000;
    thresh_max_abs_z_vel    = 3;
    vicon_marker_radius     = 15.0/2000.0;  % in meter
    hand_marker_radius      = 50/1000.0;    % in meter

    obs_name        = [obs_type,'_object'];
    in_dir_path     = ['/home/gsutanto/Desktop/CLMC/Data/DMP_LOA_Ct_Vicon_Data_Collection_201607/learn_obs_avoid_gsutanto_vicon_data/',obs_type,'_static_obs/',num2str(setting_number),'/'];

    file_obs_pos    = dir([in_dir_path,'obs_position/d*']);
    [data,vars,freq]= clmcplot_convert([in_dir_path,'obs_position/',file_obs_pos.name]);
    dt              = 1.0/freq;
    START_MARKER_XYZ= clmcplot_getvariables(data, vars, {'start_marker_x','start_marker_y','start_marker_z'});
    GOAL_MARKER_XYZ = clmcplot_getvariables(data, vars, {'goal_marker_x','goal_marker_y','goal_marker_z'});
    OBS_XYZ         = clmcplot_getvariables(data, vars, {[obs_name,'_x'],[obs_name,'_y'],[obs_name,'_z']});
    OBS_QWXYZ       = clmcplot_getvariables(data, vars, {[obs_name,'_qw'],[obs_name,'_qx'],[obs_name,'_qy'],[obs_name,'_qz']});
    OBS_POSE        = zeros(4,4,size(OBS_XYZ,1));

    OBS_MARKERS_LOCAL_XYZ              = dlmread([obs_name,'.txt']);
    OBS_MARKERS_LOCAL_XYZ_HOMOGENEOUS  = [(OBS_MARKERS_LOCAL_XYZ/1000.0).';ones(1,size(OBS_MARKERS_LOCAL_XYZ,1))];
    OBS_MARKERS_GLOBAL_XYZ_TRAJ        = zeros(size(OBS_MARKERS_LOCAL_XYZ_HOMOGENEOUS,1),size(OBS_MARKERS_LOCAL_XYZ_HOMOGENEOUS,2),size(OBS_XYZ,1));

    for t=1:size(OBS_XYZ,1)
        q                           = quaternion(OBS_QWXYZ(t,:));
        qn                          = q.normalize;
        R                           = qn.RotationMatrix;
        OBS_POSE(1:3,1:3,t)  = R;
        OBS_POSE(1:3,4,t)    = OBS_XYZ(t,:).';
        OBS_POSE(4,4,t)      = 1.0;
        OBS_MARKERS_GLOBAL_XYZ_TRAJ(:,:,t) = OBS_POSE(:,:,t) * OBS_MARKERS_LOCAL_XYZ_HOMOGENEOUS;
    end

    OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL    = cell(1,size(OBS_MARKERS_GLOBAL_XYZ_TRAJ,2));
    for j=1:size(OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL,2)
        OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}   = reshape(OBS_MARKERS_GLOBAL_XYZ_TRAJ(1:3,j,:),3,size(OBS_XYZ,1));
    end

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
        %     plot_sphere(hand_marker_radius, ...
        %                 OBS_XYZ(end,1), ...
        %                 OBS_XYZ(end,2), ...
        %                 OBS_XYZ(end,3));
            for j=1:size(OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL,2)
                plot_sphere(vicon_marker_radius,...
                            OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}(1,end),...
                            OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}(2,end),...
                            OBS_MARKER_GLOBAL_XYZ_TRAJ_CELL{1,j}(3,end))
            end
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
end