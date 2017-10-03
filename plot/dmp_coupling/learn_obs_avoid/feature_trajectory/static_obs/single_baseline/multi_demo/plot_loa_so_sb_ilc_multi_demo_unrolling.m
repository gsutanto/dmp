function plot_loa_so_sb_ilc_multi_demo_unrolling( varargin )
    % Author: Giovanni Sutanto
    % Date  : March 16, 2016
    close   all;
    clc;
    
    in_data_path    = '../../../../../../../data/dmp_coupling/learn_obs_avoid/static_obs/data_multi_demo_static/';
    
    % count number of available static obstacle settings:
    i = 1;
    while (exist(strcat(in_data_path, num2str(i)), 'dir'))
        i                           = i + 1;
    end
    num_settings                    = i - 1;
    
    is_plotting_ilc_iterations      = 0;
    
    is_plotting_per_axis            = 0;
    
    if (nargin > 0)
        num_settings                = varargin{1};
    end
    if (nargin > 1)
        is_plotting_ilc_iterations  = varargin{2};
    end
    if (nargin > 2)
        is_plotting_per_axis        = varargin{3};
    end
    
    for n = 1:num_settings
        obs_sph_center_coord        = dlmread(strcat(in_data_path, num2str(n), '/obs_sph_center_coord.txt'));
        obs_sph_radius              = dlmread(strcat(in_data_path, num2str(n), '/obs_sph_radius.txt'));
        
        unrolled_traj_wo_obs        = dlmread('unroll_tests/baseline/transform_sys_state_global_trajectory.txt');
        averaged_traj_w_obs         = dlmread(strcat(num2str(n), '/transform_sys_state_global_trajectory.txt'));
        unrolled_traj_w_obs_ideal   = dlmread(strcat('unroll_tests/', num2str(n), '/ideal/transform_sys_state_global_trajectory.txt'));
        unrolled_traj_w_obs_real    = dlmread(strcat('unroll_tests/', num2str(n), '/real/transform_sys_state_global_trajectory.txt'));
        
        ilc_num_iter                = 1;
        while (exist(strcat('ILC/', num2str(ilc_num_iter)), 'dir'))
            ilc_unrolled_traj{ilc_num_iter} = dlmread(strcat('ILC/', num2str(ilc_num_iter), '/', num2str(n), '/transform_sys_state_global_trajectory.txt'));
            ilc_num_iter            = ilc_num_iter + 1;
        end
        ilc_num_iter                = ilc_num_iter - 1;

        figure;
        hold            on;
        grid            on;
        
        plot3(unrolled_traj_wo_obs(:,2),      unrolled_traj_wo_obs(:,3),      unrolled_traj_wo_obs(:,4),      'r*');
        legend_plot3{1}  = ['BASELINE (w/o coupling term)'];
        plot3(averaged_traj_w_obs(:,2),       averaged_traj_w_obs(:,3),       averaged_traj_w_obs(:,4),       'b*');
        legend_plot3{2}  = ['AVERAGED (w/ obstacle)'];
        plot3(unrolled_traj_w_obs_ideal(:,2), unrolled_traj_w_obs_ideal(:,3), unrolled_traj_w_obs_ideal(:,4), 'g*');
        legend_plot3{3}  = ['IDEALISTIC (w/ obstacle)'];
        plot3(unrolled_traj_w_obs_real(:,2),  unrolled_traj_w_obs_real(:,3),  unrolled_traj_w_obs_real(:,4),  'c*');
        legend_plot3{4}  = ['REALISTIC (w/ obstacle)'];
        
        if (is_plotting_ilc_iterations)
            ilc_iter_disp_idx= 1:4:ilc_num_iter;

            for i = 1:length(ilc_iter_disp_idx)
                plot3(ilc_unrolled_traj{ilc_iter_disp_idx(i)}(:,2),  ...
                      ilc_unrolled_traj{ilc_iter_disp_idx(i)}(:,3),  ...
                      ilc_unrolled_traj{ilc_iter_disp_idx(i)}(:,4), 'LineWidth', 3);
                legend_plot3{4+i}   = ['ILC iter ', num2str(ilc_iter_disp_idx(i))];
            end
        end
        
        plot_sphere(obs_sph_radius, obs_sph_center_coord(1,1),      obs_sph_center_coord(2,1),      obs_sph_center_coord(3,1));
        plot_sphere(0.01,           unrolled_traj_wo_obs(end,2),    unrolled_traj_wo_obs(end,3),    unrolled_traj_wo_obs(end,4));
        
        title(strcat('Plot of Learn-Obs-Avoid Averaged vs Unrolled Trajectories for Setting #', num2str(n)));
        xlabel('x');
        ylabel('y');
        zlabel('z');
        legend(legend_plot3{:});
        campos([1.1699   -3.3745    0.5115]);
        hold            off;
        
        if (is_plotting_per_axis)
            for ax = 1:3
                figure;
                hold        on;
                grid        on;

                plot(unrolled_traj_wo_obs(:,1),      unrolled_traj_wo_obs(:,(ax+1)),        'r*');
                legend_plot{1}  = ['BASELINE (w/o coupling term)'];
                plot(averaged_traj_w_obs(:,1),       averaged_traj_w_obs(:,(ax+1)),         'b*');
                legend_plot{2}  = ['AVERAGED (w/ obstacle)'];
                plot(unrolled_traj_w_obs_ideal(:,1), unrolled_traj_w_obs_ideal(:,(ax+1)),   'g*');
                legend_plot{3}  = ['IDEALISTIC (w/ obstacle)'];
                plot(unrolled_traj_w_obs_real(:,1),  unrolled_traj_w_obs_real(:,(ax+1)),    'c*');
                legend_plot{4}  = ['REALISTIC (w/ obstacle)'];

                if (is_plotting_ilc_iterations)
                    ilc_iter_disp_idx= 1:1:ilc_num_iter;

                    for i = 1:length(ilc_iter_disp_idx)
                        plot(ilc_unrolled_traj{ilc_iter_disp_idx(i)}(:,1),  ...
                             ilc_unrolled_traj{ilc_iter_disp_idx(i)}(:,(ax+1)), 'LineWidth', 1);
                        legend_plot{4+i}    = ['ILC iter ', num2str(ilc_iter_disp_idx(i))];
                    end
                end

                title(strcat('Plot of Learn-Obs-Avoid Averaged vs Unrolled Trajectories for Setting #', num2str(n), ', axis ', num2str(ax)));
                legend(legend_plot{:});
                hold            off;
            end
        end
    end
    
    if (is_plotting_ilc_iterations)
        Ct_mse_history  = dlmread('Ct_mse_history_transpose.txt');
        figure;
        hold            on;
        title('Mean Squared Error of Ct\_target - Ct\_actual');
        for ax = 1:3
            subplot(2,2,ax);
            plot(Ct_mse_history(:,ax));
            grid        on;
            title(strcat('axis ', num2str(ax)));
        end
        hold            off;
    end
end
