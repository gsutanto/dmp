function plot_loa_static_obs_single_baseline_multi_demo_trajectories(  )
    % Author: Giovanni Sutanto
    % Date  : January 05, 2016
    close   all;
    clc;
    
    in_data_path    = '../../../../../../../data/dmp_coupling/learn_obs_avoid/static_obs/data_multi_demo_static/';
    out_data_path   = './';
    subdir          = {'baseline'};
    
    % count number of available static obstacle settings:
    i = 1;
    while (exist(strcat(in_data_path, num2str(i)), 'dir'))
        i                           = i + 1;
    end
    num_settings                    = i - 1;
    
    ColorSet        = varycolor(num_settings+1);
    
    training_trajectory                                     = cell(0, 0);
    obs_sph_center_coord                                    = cell(0, 0);
    obs_sph_radius                                          = cell(0, 0);
    for i = 1:(num_settings+1)
        if (i > 1)
            subdir{i}                                       = num2str(i-1);
            obs_sph_center_coord{i}                         = dlmread(strcat(in_data_path, subdir{i}, '/obs_sph_center_coord.txt'));
            obs_sph_radius{i}                               = dlmread(strcat(in_data_path, subdir{i}, '/obs_sph_radius.txt'));
        else
            obs_sph_center_coord{i}                         = [];
            obs_sph_radius{i}                               = [];
        end
        files       = dir(strcat(in_data_path, subdir{i}, '/endeff_trajs/*.txt'));
        j           = 0;
        for file = files'
            j                                               = j + 1;
            training_trajectory{i}{j}                       = dlmread(strcat(in_data_path, subdir{i}, '/endeff_trajs/', file.name));
        end
    end
    
    unrolled_trajectory                                     = cell(0, 0);
    for i = 1:(num_settings+1)
        unrolled_trajectory{i}                              = dlmread(strcat(out_data_path, subdir{i}, '/transform_sys_state_global_trajectory.txt'));
    end
    
    % plot altogether (all-settings):
    figure;
    hold            on;
    grid            on;
    px              = quiver3(0,0,0,1,0,0,'r');
    py              = quiver3(0,0,0,0,1,0,'g');
    pz              = quiver3(0,0,0,0,0,1,'b');
    for i = 1:length(training_trajectory)
        for j = 1:length(training_trajectory{i})
            plot3(training_trajectory{i}{j}(:,2)', training_trajectory{i}{j}(:,3)', training_trajectory{i}{j}(:,4)',...
                  ':', 'Color', ColorSet(i,:));
        end
        po_ur_traj{i}   = plot3(unrolled_trajectory{i}(:,2)', unrolled_trajectory{i}(:,3)', unrolled_trajectory{i}(:,4)',...
                                'Color', ColorSet(i,:), 'LineWidth', 3);
    end
    title('Plot of Learn-Obs-Avoid Training and Unrolled Trajectories');
    xlabel('x');
    ylabel('y');
    zlabel('z');
    legend([px, py, pz, po_ur_traj{:}], 'global x-axis', 'global y-axis', 'global z-axis', subdir{:});
    hold            off;
    
    % plot stand-alone per-setting:
    for i = 2:length(training_trajectory)
        figure;
        hold            on;
        grid            on;
        px              = quiver3(0,0,0,1,0,0,'r');
        py              = quiver3(0,0,0,0,1,0,'g');
        pz              = quiver3(0,0,0,0,0,1,'b');
        
        plot_sphere(obs_sph_radius{i}, obs_sph_center_coord{i}(1,1), obs_sph_center_coord{i}(2,1), obs_sph_center_coord{i}(3,1));
    
        for j = 1:length(training_trajectory{1})
            plot3(training_trajectory{1}{j}(:,2)', training_trajectory{1}{j}(:,3)', training_trajectory{1}{j}(:,4)',...
                  ':', 'Color', ColorSet(1,:));
        end
        po_ur_baseline_traj   = plot3(unrolled_trajectory{1}(:,2)', unrolled_trajectory{1}(:,3)', unrolled_trajectory{1}(:,4)',...
                                      'Color', ColorSet(1,:), 'LineWidth', 3);
                               
        for j = 1:length(training_trajectory{i})
            plot3(training_trajectory{i}{j}(:,2)', training_trajectory{i}{j}(:,3)', training_trajectory{i}{j}(:,4)',...
                  ':', 'Color', ColorSet(i,:));
        end
        po_ur_w_obs_traj{i}   = plot3(unrolled_trajectory{i}(:,2)', unrolled_trajectory{i}(:,3)', unrolled_trajectory{i}(:,4)',...
                                      'Color', ColorSet(i,:), 'LineWidth', 3);
        title(strcat('Plot of Learn-Obs-Avoid Training and Unrolled Trajectories for Setting #', subdir{i}));
        xlabel('x');
        ylabel('y');
        zlabel('z');
        legend([px, py, pz, po_ur_baseline_traj, po_ur_w_obs_traj{i}], 'global x-axis', 'global y-axis', 'global z-axis', subdir{1}, subdir{i});
        hold            off;
    end
end
