function extract_obs_avoid_single_demo_static_vicon_data(varargin)
    % Author: Giovanni Sutanto
    % Date  : July 17, 2016
    path_to_be_added    = '../../../../utilities';
    current_dir_str     = pwd;
    cd(path_to_be_added);
    target_dir_str      = pwd;
    cd(current_dir_str);
    path_cell           = regexp(path, pathsep, 'split');
    is_on_path          = or(any(strcmpi(path_to_be_added, path_cell)), ...
                             any(strcmpi(target_dir_str, path_cell)));
    if (~is_on_path)
        addpath(path_to_be_added);
        fprintf('Path added.\n');
    end
    
    in_data_subdir_name         = varargin{1};
    out_data_dir_name           = varargin{2};
    overall_demo_setting_count  = varargin{3};
    
    is_plotting_trajs           = 0;

    D                           = 3;
    
    thresh_max_abs_z_accel      = 1000;
    thresh_max_abs_z_vel        = 3;

    Wn                          = 0.01;

    % Use the following if filtering is needed:
    [b,a]                       = butter(2, Wn);
    
    vicon_marker_radius         = 15.0/2000.0;  % in meter
    hand_marker_radius          = 50/1000.0;    % in meter
    
    fprintf(['Processing Obstacle Avoidance Demonstration Setting ', in_data_subdir_name, '...\n']);
    
    var_names   = {'time',...
                   'right_wrist_x','right_wrist_y','right_wrist_z',...
                   'start_marker_x','start_marker_y','start_marker_z',...
                   'goal_marker_x','goal_marker_y','goal_marker_z'};
    
    if (strcmp(in_data_subdir_name, 'baseline') == 1)
        out_data_subdir_name    = strcat(out_data_dir_name, '/', in_data_subdir_name);
    else % for obstacle avoidance demonstrations
        strs                    = strsplit(in_data_subdir_name, '_');
        obs_geometry            = char(strs(1));
        obs_name                = [obs_geometry,'_object'];
        obs_pose_var_names      = {[obs_name,'_x'],[obs_name,'_y'],[obs_name,'_z'],...
                                   [obs_name,'_qw'],[obs_name,'_qx'],[obs_name,'_qy'],[obs_name,'_qz']};
         
        obs_pose_subdir_name    = [in_data_subdir_name, 'obs_position/'];
        file_obs_position       = dir(strcat(obs_pose_subdir_name,'/','d*'));
        [obs_pose_var_data]     = clmcplotGetNullClippedData(strcat(obs_pose_subdir_name,'/',file_obs_position(1,1).name), obs_pose_var_names);
        obs_xyz                 = obs_pose_var_data(1,1:3);
        obs_qwxyz             	= obs_pose_var_data(1,4:7);
        obs_pose_homogeneous_T  = zeros(4,4);
        
        obs_markers_local_xyz                   = dlmread([obs_name,'.txt']);
        obs_markers_local_xyz_homogeneous       = [(obs_markers_local_xyz/1000.0).';ones(1,size(obs_markers_local_xyz,1))];
        
        q                                       = quaternion(obs_qwxyz);
        qn                                      = q.normalize;
        R                                       = qn.RotationMatrix;
        obs_pose_homogeneous_T(1:3,1:3)         = R;
        obs_pose_homogeneous_T(1:3,4)           = obs_xyz.';
        obs_pose_homogeneous_T(4,4)             = 1.0;
        obs_markers_global_xyz_homogeneous      = obs_pose_homogeneous_T * obs_markers_local_xyz_homogeneous;
        obs_markers_global_xyz                  = obs_markers_global_xyz_homogeneous(1:3,:);
        
        out_data_subdir_name    = strcat(out_data_dir_name, '/', num2str(overall_demo_setting_count));
    end
    if (exist(out_data_subdir_name, 'dir'))
        rmdir(out_data_subdir_name, 's');
    end
    mkdir(out_data_subdir_name);
    
    out_data_endeff_subdir_name = strcat(out_data_subdir_name, '/endeff_trajs/');
    if (exist(out_data_endeff_subdir_name, 'dir'))
        rmdir(out_data_endeff_subdir_name, 's');
    end
    mkdir(out_data_endeff_subdir_name);
                        
    file_count                  = 1;
    files = dir(strcat(in_data_subdir_name,'/','d*'));
    end_eff_xyz_cell            = cell(3,0);
    end_eff_xyz_unfiltered_cell = cell(1,0);
    filenames                   = cell(1,0);
    for file = files'
        if (file_count == 1)
            [data_unused,vars_unused,freq]  = clmcplot_convert(strcat(in_data_subdir_name,'/',file.name));
            dt                  = 1.0/freq;
        end
        [var_data]              = clmcplotGetNullClippedData(strcat(in_data_subdir_name,'/',file.name), var_names);
        time                    = var_data(:,1);
        end_eff_x_unfiltered    = var_data(:,2);
        end_eff_y_unfiltered    = var_data(:,3);
        end_eff_z_unfiltered    = var_data(:,4);
        start_x                 = var_data(:,5);
        start_y                 = var_data(:,6);
        start_z                 = var_data(:,7);
        goal_x                  = var_data(:,8);
        goal_y                  = var_data(:,9);
        goal_z                  = var_data(:,10);
        
        if (file_count == 1)
            start_xyz           = [start_x(1,1), start_y(1,1), start_z(1,1)];
            goal_xyz            = [goal_x(1,1), goal_y(1,1), goal_z(1,1)];
        end
        
        end_eff_xd_unfiltered   = diffnc(end_eff_x_unfiltered,dt);
        end_eff_yd_unfiltered   = diffnc(end_eff_y_unfiltered,dt);
        end_eff_zd_unfiltered   = diffnc(end_eff_z_unfiltered,dt);
        end_eff_xdd_unfiltered  = diffnc(end_eff_xd_unfiltered,dt);
        end_eff_ydd_unfiltered  = diffnc(end_eff_yd_unfiltered,dt);
        end_eff_zdd_unfiltered  = diffnc(end_eff_zd_unfiltered,dt);
        
        end_eff_xyz_unfiltered  = [end_eff_x_unfiltered, end_eff_y_unfiltered, end_eff_z_unfiltered];
        
        % Filter out abnormal trajectories:
        % (1) Discontinuous Trajectory
        if ((max(abs(end_eff_zdd_unfiltered)) > thresh_max_abs_z_accel) || ...
            (max(abs(end_eff_zd_unfiltered)) > thresh_max_abs_z_vel))
            fprintf(['Path Discontinuity occurs in ', in_data_subdir_name,...
                     ', file ', file.name, '!\n']);
            continue; % skip if there is a movement discontinuity
        end
%         path_abs_diff_threshold = 0.03; % path absolute diff threshold is 3 cm (for consecutive points in the path)
%         if (detectPathDiscontinuity(end_eff_xyz_unfiltered, path_abs_diff_threshold) == 1)
%             fprintf(['Path Discontinuity occurs in ', in_data_subdir_name,...
%                      ', file ', file.name, '!\n']);
%             continue;
%         end
%         % (2) Too-Short Trajectory
        path_shortness_threshold = 0.05; % path shortness threshold is 5 cm (overall path length)
        if (detectPathShortness(end_eff_xyz_unfiltered, path_shortness_threshold) == 1)
            fprintf(['Path Too-Short occurs in ', in_data_subdir_name,...
                     ', file ', file.name, '!\n']);
            continue;
        end
%         % (3) Clip Start and End Trajectory (Eliminates Initial Motion Delay and Termination of Motion Delay)
        motion_threshold    = 0.015; % clipping threshold is 1.5 cm, both from start and end points of trajectory
        [start_idx, end_idx] = getTrajectoryStartEndClippingIndex(end_eff_xyz_unfiltered, motion_threshold);
%         fprintf(['Path Original Start, Chosen Start, Chosen End, Original End Indices are: ',...
%                  '1, ', num2str(start_idx), ', ', num2str(end_idx), ', ', num2str(size(end_eff_xyz,1)), ' in ', in_data_subdir_name,...
%                      ', file ', file.name, '\n']);
        end_eff_x_unfiltered    = end_eff_x_unfiltered(start_idx:end_idx,:);
        end_eff_y_unfiltered    = end_eff_y_unfiltered(start_idx:end_idx,:);
        end_eff_z_unfiltered    = end_eff_z_unfiltered(start_idx:end_idx,:);
        end_eff_xyz_unfiltered  = [end_eff_x_unfiltered, end_eff_y_unfiltered, end_eff_z_unfiltered];

        % "Complete" trajectory, such that it starts and ends with 
        % zero velocity and zero acceleration:
        tautmp                          = (size(end_eff_xyz_unfiltered,1)-1)*dt;
        if (tautmp == 0)
            keyboard;
        end
        end_eff_xyz_unfilt_completed    = completeTrajectory(end_eff_xyz_unfiltered, dt, tautmp, 1.85);
        
        % Filter and differentiate to get the velocity and acceleration:
        end_eff_x               = filtfilt(b,a,end_eff_xyz_unfilt_completed(:,1));
        end_eff_y               = filtfilt(b,a,end_eff_xyz_unfilt_completed(:,2));
        end_eff_z               = filtfilt(b,a,end_eff_xyz_unfilt_completed(:,3));
        end_eff_xd              = diffnc(end_eff_x,dt);
        end_eff_yd              = diffnc(end_eff_y,dt);
        end_eff_zd              = diffnc(end_eff_z,dt);
        end_eff_xdd             = diffnc(end_eff_xd,dt);
        end_eff_ydd             = diffnc(end_eff_yd,dt);
        end_eff_zdd             = diffnc(end_eff_zd,dt);
        
        end_eff_xyz             = [end_eff_x, end_eff_y, end_eff_z];
        
        end_eff_xyz_unfiltered_cell{1,file_count}   = [end_eff_x_unfiltered, end_eff_y_unfiltered, end_eff_z_unfiltered];

        end_eff_xyz_cell{1,file_count}  = end_eff_xyz;
        filenames{1,file_count}         = file.name;
        
        end_eff_xdydzd                  = [end_eff_xd, end_eff_yd, end_eff_zd];
        end_eff_xyz_cell{2,file_count}  = end_eff_xdydzd;
        
        end_eff_xddyddzdd               = [end_eff_xdd, end_eff_ydd, end_eff_zdd];
        end_eff_xyz_cell{3,file_count}  = end_eff_xddyddzdd;
        
        if (strcmp(in_data_subdir_name, 'baseline'))
            fileID  = fopen(strcat(out_data_subdir_name, '/does_obs_exist.txt'),'w');
            nbytes  = fprintf(fileID, '%d\n', 0);       % obstacle does NOT exist
            fclose(fileID);
        else % for obstacle avoidance demonstrations
            fileID  = fopen(strcat(out_data_subdir_name, '/does_obs_exist.txt'),'w');
            nbytes  = fprintf(fileID, '%d\n', 1);       % obstacle exists
            fclose(fileID);
            
            dlmwrite(strcat(out_data_subdir_name, '/obs_markers_global_coord.txt'), obs_markers_global_xyz.', 'delimiter', ' ');
            
            fileID  = fopen(strcat(out_data_subdir_name, '/obs_geometry.txt'),'w');
            nbytes  = fprintf(fileID, '%s\n', obs_geometry);
            fclose(fileID);
        end
                    
        fileID = fopen(strcat(out_data_subdir_name, '/is_obs_static.txt'),'w');
        nbytes = fprintf(fileID, '%d\n', 1);        % obstacle is static (not moving)
        fclose(fileID);
        
        time        = [0.0:dt:dt*(length(end_eff_x)-1)].';
        
        traj        = [time, end_eff_x, end_eff_y, end_eff_z, end_eff_xd, end_eff_yd, end_eff_zd, end_eff_xdd, end_eff_ydd, end_eff_zdd];
        
        dlmwrite(strcat(out_data_endeff_subdir_name, num2str(file_count), '.txt'), traj, 'delimiter', ' ');
        
        file_count              = file_count + 1;
    end
    
    % The plotting below is for hand-picking trajectories, based on
    % color-coded plot with filenames in the legend:
    if (is_plotting_trajs)
%         if (strcmp(in_data_subdir_name, 'baseline') == 0)
            figure;
            axis equal;
            hold on;
                plot_sphere(hand_marker_radius, ...
                            start_xyz(1,1), ...
                            start_xyz(1,2), ...
                            start_xyz(1,3));
                plot_sphere(hand_marker_radius, ...
                            goal_xyz(1,1), ...
                            goal_xyz(1,2), ...
                            goal_xyz(1,3));
            %     plot_sphere(hand_marker_radius, ...
            %                 OBS_XYZ(end,1), ...
            %                 OBS_XYZ(end,2), ...
            %                 OBS_XYZ(end,3));
                if (strcmp(in_data_subdir_name, 'baseline') == 0)
                    for j=1:size(obs_markers_global_xyz,1)
                        plot_sphere(vicon_marker_radius,...
                                    obs_markers_global_xyz(j,1),...
                                    obs_markers_global_xyz(j,2),...
                                    obs_markers_global_xyz(j,3))
                    end
                end
                for k=1:size(end_eff_xyz_cell,2)
                    plot3(end_eff_xyz_cell{1,k}(:,1),...
                          end_eff_xyz_cell{1,k}(:,2),...
                          end_eff_xyz_cell{1,k}(:,3),'g');
                end
                xlabel('x');
                ylabel('y');
                zlabel('z');
            hold off;

            [ linespec_codes ]      = generateLinespecCodes();

            for order=1:3 % 1=position; 2=velocity; 3=acceleration
                new_traj_length     = size(end_eff_xyz_cell{order,1},1);
                figure;
                axis equal;
                for d=1:D
                    subplot(D,1,d)
                    if (d==1)
                        title(['order=',num2str(order)]);
                    end
                    hold on;
                        for k=1:size(end_eff_xyz_cell,2)
                            stretched_traj  = stretchTrajectory( end_eff_xyz_cell{order,k}(:,d), new_traj_length );
                            plot_handle{k}  = plot(stretched_traj, linespec_codes{1,k});
                            plot_legend{k}  = filenames{1,k};
                        end
                        legend([plot_handle{:}], plot_legend{:});
                    hold off;
                end
            end
%             cd('/home/gsutanto/amd_clmc_arm/workspace/src/catkin/control/dmp/matlab/dmp_coupling/learn_obs_avoid/SL_data_conversion_to_text_files_format/vicon_201607/');
            keyboard;
%         end
    end
end

