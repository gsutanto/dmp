function [  ] = visualizeSetting( varargin )
    setting_no                      = varargin{1};
    if (nargin > 1)
        is_visualizing_baseline     = varargin{2};
    else
        is_visualizing_baseline     = 0;
    end
    if (nargin > 2)
        if (ischar(varargin{3}) == 1)
            is_saving_figure        = 1;
            figure_path             = varargin{3};
        else
            is_saving_figure        = 0;
            figure_path             = '';
        end
    else
        is_saving_figure            = 0;
        figure_path                 = '';
    end
    if (nargin > 3)
        is_visualizing_2D_demo      = varargin{4};
    else
        is_visualizing_2D_demo      = 0;
    end
    if (nargin > 4)
        is_visualizing_unrolling    = 1;
        baseline_global_traj_unroll = varargin{5};
        obs_avoid_global_traj_unroll= varargin{6};
    else
        is_visualizing_unrolling    = 0;
    end
    if (setting_no == 0)
        point_obstacles_cart_position_global    = varargin{7};
    end

    if (exist('../utilities/', 'dir') == 7)
        addpath('../utilities/');
    elseif (exist('../../utilities/', 'dir') == 7)
        addpath('../../utilities/');
    end
    
    vicon_marker_radius             = 15.0/2000.0;  % in meter
    critical_position_marker_radius = 30/1000.0;    % in meter
    
    D                               = 3;
    
    %% Data Creation/Loading

    if (exist('../data', 'dir') == 7)
        data_dirpath    = '../data/';
    elseif (exist('../../data', 'dir') == 7)
        data_dirpath    = '../../data/';
    end
    data_filepath       = [data_dirpath, 'data_multi_demo_vicon_static_global_coord.mat'];
    
    % if input file is not exist yet, then create one (convert from C++ textfile format):
    if (~exist(data_filepath, 'file'))
        convert_loa_vicon_data_to_mat_format;
    end

    load(data_filepath);

    % end of Data Creation/Loading
    
    %% Plotting
    
    % 3D Plot:
    h_3D    = figure;
    axis equal;
    hold on;
        if (is_visualizing_unrolling)
            baseline_demo_line_code     = '-.c';
            if (is_visualizing_baseline)
                pbu  = plot3(baseline_global_traj_unroll{1,1}(:,1),...
                             baseline_global_traj_unroll{1,1}(:,2),...
                             baseline_global_traj_unroll{1,1}(:,3),...
                             'b*');
            end
            
            obs_avoid_demo_line_code    = '-.m';
            pou  = plot3(obs_avoid_global_traj_unroll{1,1}(:,1),...
                         obs_avoid_global_traj_unroll{1,1}(:,2),...
                         obs_avoid_global_traj_unroll{1,1}(:,3),...
                         'r*');
        else
            baseline_demo_line_code     = 'c';
            obs_avoid_demo_line_code    = 'm';
        end
        
        if (is_visualizing_baseline)
            for i=1:size(data_global_coord.baseline,2)
                pbd  = plot3(data_global_coord.baseline{1,i}(:,1),...
                             data_global_coord.baseline{1,i}(:,2),...
                             data_global_coord.baseline{1,i}(:,3),...
                             baseline_demo_line_code);
            end
        end
        
        plot_sphere(critical_position_marker_radius,...
                    data_global_coord.baseline{1,1}(end,1),...
                    data_global_coord.baseline{1,1}(end,2),...
                    data_global_coord.baseline{1,1}(end,3));
        
        if (setting_no ~= 0)
            for op=1:size(data_global_coord.obs_avoid{setting_no,1},1)
                plot_sphere(vicon_marker_radius,...
                            data_global_coord.obs_avoid{setting_no,1}(op,1),...
                            data_global_coord.obs_avoid{setting_no,1}(op,2),...
                            data_global_coord.obs_avoid{setting_no,1}(op,3));
            end

            for i=1:size(data_global_coord.obs_avoid{setting_no,2},2)
                pod  = plot3(data_global_coord.obs_avoid{setting_no,2}{1,i}(:,1),...
                             data_global_coord.obs_avoid{setting_no,2}{1,i}(:,2),...
                             data_global_coord.obs_avoid{setting_no,2}{1,i}(:,3),...
                             obs_avoid_demo_line_code);
            end
        else
            for op=1:size(point_obstacles_cart_position_global,1)
                plot_sphere(vicon_marker_radius,...
                            point_obstacles_cart_position_global(op,1),...
                            point_obstacles_cart_position_global(op,2),...
                            point_obstacles_cart_position_global(op,3));
            end
        end
        
        if (is_visualizing_unrolling)
            if (is_visualizing_baseline)
                if (setting_no ~= 0)
                    legend([pbd, pod, pbu, pou], ...
                           'baseline demo trajectories', ...
                           'obstacle avoidance demo trajectories', ...
                           'baseline unroll trajectory', ...
                           'obstacle avoidance unroll trajectory');
                else
                    legend([pbd, pbu, pou], ...
                           'baseline demo trajectories', ...
                           'baseline unroll trajectory', ...
                           'obstacle avoidance unroll trajectory');
                end
            else
                if (setting_no ~= 0)
                    legend([pod, pou], ...
                           'obstacle avoidance demo trajectories', ...
                           'obstacle avoidance unroll trajectory');
                else
                    legend([pou], ...
                           'obstacle avoidance unroll trajectory');
                end
            end
        else
            if (is_visualizing_baseline)
                if (setting_no ~= 0)
                    legend([pbd, pod], 'baseline demo trajectories', 'obstacle avoidance demo trajectories');
                else
                    legend([pbd], 'baseline demo trajectories');
                end
            else
                if (setting_no ~= 0)
                    legend([pod], 'obstacle avoidance demo trajectories');
                end
            end
        end
        xlabel('x');
        ylabel('y');
        zlabel('z');
    hold off;
    if ((is_saving_figure) && (strcmp(figure_path,'') ~= 1))
        savefig(h_3D, figure_path);
%         pause(10)
        close(h_3D);
    end
    
    if ((is_visualizing_2D_demo) && (setting_no ~= 0))
        % 2D Plot:
        for ord=1:3
            figure;
            axis equal;
            if (ord == 1)
                order_string    = 'position';
            elseif (ord == 2)
                order_string    = 'velocity';
            elseif (ord == 3)
                order_string    = 'acceleration';
            end
            for d=1:D
                for i=1:size(data_global_coord.obs_avoid{setting_no,2},2)
                    subplot(D,1,d);
                        hold on;
                            plot(data_global_coord.obs_avoid{setting_no,2}{ord,i}(:,d), obs_avoid_demo_line_code);
                            title(['Plot of ', order_string, ', dimension ', num2str(d)]);
                        hold off;
                end
            end
        end
    end
    % end of Plotting
end