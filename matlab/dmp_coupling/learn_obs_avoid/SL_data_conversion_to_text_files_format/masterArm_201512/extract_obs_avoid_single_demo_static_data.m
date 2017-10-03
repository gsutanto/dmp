function extract_obs_avoid_single_demo_static_data(in_data_subdir_name, out_data_dir_name)
    % Author: Giovanni Sutanto
    % Date  : December 30, 2015
    
    fprintf(['Processing Obstacle Avoidance Demonstration Setting ', in_data_subdir_name, '...\n']);
    
    out_data_subdir_name        = strcat(out_data_dir_name, '/', in_data_subdir_name);
    if (exist(out_data_subdir_name, 'dir'))
        rmdir(out_data_subdir_name, 's');
    end
    mkdir(out_data_subdir_name);
    
    out_data_endeff_subdir_name = strcat(out_data_subdir_name, '/endeff_trajs/');
    if (exist(out_data_endeff_subdir_name, 'dir'))
        rmdir(out_data_endeff_subdir_name, 's');
    end
    mkdir(out_data_endeff_subdir_name);
    
    var_names   = {'time',...
                   'R_HAND_x','R_HAND_xd','R_HAND_xdd',...
                   'R_HAND_y','R_HAND_yd','R_HAND_ydd',...
                   'R_HAND_z','R_HAND_zd','R_HAND_zdd',...
                   'BLOB1_x','BLOB1_y','BLOB1_z',...
                   'BLOB2_x','BLOB2_y','BLOB2_z','BLOB3_x'};
                       
    file_count                  = 1;
    files = dir(strcat(in_data_subdir_name,'/','d*'));
    for file = files'
        [var_data]              = clmcplotGetNullClippedData(strcat(in_data_subdir_name,'/',file.name), var_names);
        time                    = var_data(:,1);
        EndEff_x                = var_data(:,2);
        EndEff_xd               = var_data(:,3);
        EndEff_xdd              = var_data(:,4);
        EndEff_y                = var_data(:,5);
        EndEff_yd               = var_data(:,6);
        EndEff_ydd              = var_data(:,7);
        EndEff_z                = var_data(:,8);
        EndEff_zd               = var_data(:,9);
        EndEff_zdd              = var_data(:,10);
        ObsCtr_x                = var_data(:,11);
        ObsCtr_y                = var_data(:,12);
        ObsCtr_z                = var_data(:,13);
        ObsSph_Radius           = var_data(:,14);
        ObsPositionSelection    = var_data(:,15);
        DoesObsExist            = var_data(:,16);
        DoesCollisionOccur      = var_data(:,17);
        if (~(all(DoesCollisionOccur==0)))  % If there is a collision...
            continue;                       % then skip the data file.
        end
        EndEff_xyz              = [EndEff_x, EndEff_y, EndEff_z];
        
        % Filter out abnormal trajectories:
        % (1) Discontinuous Trajectory
        path_abs_diff_threshold = 0.03; % path absolute diff threshold is 3 cm (for consecutive points in the path)
        if (detectPathDiscontinuity(EndEff_xyz, path_abs_diff_threshold) == 1)
            fprintf(['Path Discontinuity occurs in ', in_data_subdir_name,...
                     ', file ', file.name, '!\n']);
            continue;
        end
        % (2) Too-Short Trajectory
        path_shortness_threshold = 0.05; % path shortness threshold is 5 cm (overall path length)
        if (detectPathShortness(EndEff_xyz, path_shortness_threshold) == 1)
            fprintf(['Path Too-Short occurs in ', in_data_subdir_name,...
                     ', file ', file.name, '!\n']);
            continue;
        end
        % (3) Clip Start and End Trajectory (Eliminates Initial Motion Delay and Termination of Motion Delay)
        motion_threshold    = 0.015; % clipping threshold is 1.5 cm, both from start and end points of trajectory
        [start_idx, end_idx] = getTrajectoryStartEndClippingIndex(EndEff_xyz, motion_threshold);
%         fprintf(['Path Original Start, Chosen Start, Chosen End, Original End Indices are: ',...
%                  '1, ', num2str(start_idx), ', ', num2str(end_idx), ', ', num2str(size(EndEff_xyz,1)), ' in ', in_data_subdir_name,...
%                      ', file ', file.name, '\n']);
        
        if (strcmp(in_data_subdir_name, 'baseline'))
            if (~(all(DoesObsExist==0)) || ~(all(ObsPositionSelection==0)))
                continue;
            end
            
            fileID = fopen(strcat(out_data_subdir_name, '/does_obs_exist.txt'),'w');
            nbytes = fprintf(fileID, '%d\n', 0);        % obstacle does NOT exist
            fclose(fileID);
        else
            if (~(all(DoesObsExist==1)) || ~(all(ObsPositionSelection > 0)))
                continue;
            end
            
            fileID = fopen(strcat(out_data_subdir_name, '/does_obs_exist.txt'),'w');
            nbytes = fprintf(fileID, '%d\n', 1);        % obstacle exists
            fclose(fileID);
                    
            fileID  = fopen(strcat(out_data_subdir_name, '/obs_sph_center_coord.txt'),'w');
            nbytes  = fprintf(fileID, '%f\n', ObsCtr_x(1,1));
            nbytes  = fprintf(fileID, '%f\n', ObsCtr_y(1,1));
            nbytes  = fprintf(fileID, '%f\n', ObsCtr_z(1,1));
            fclose(fileID);
                    
            fileID  = fopen(strcat(out_data_subdir_name, '/obs_sph_radius.txt'),'w');
            nbytes  = fprintf(fileID, '%f\n', ObsSph_Radius(1,1));
            fclose(fileID);
        end
                    
        fileID = fopen(strcat(out_data_subdir_name, '/is_obs_static.txt'),'w');
        nbytes = fprintf(fileID, '%d\n', 1);        % obstacle is static (not moving)
        fclose(fileID);
        
        traj        = [(time-time(start_idx,1)), EndEff_x, EndEff_y, EndEff_z, EndEff_xd, EndEff_yd, EndEff_zd, EndEff_xdd, EndEff_ydd, EndEff_zdd];
        
        fileID      = fopen(strcat(out_data_endeff_subdir_name, num2str(file_count), '.txt'),'w');
        for j=start_idx:end_idx
            nbytes  = fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f\n',traj(j,:));
        end
        fclose(fileID);
        
        file_count              = file_count + 1;
    end
end

